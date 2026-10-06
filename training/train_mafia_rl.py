#!/usr/bin/env python3
"""
Mafia RL Training Script (End-to-End Language RL via GRPO / Policy Gradient with LoRA)

Designed for single-GPU training (e.g. RTX 6000 / Pro 6000 with 24GB~48GB VRAM).
Plays multi-turn Mafia games in ChatArena, observes full game outcomes (winner, survival),
calculates rewards, and optimizes the policy using Group Relative Policy Optimization (GRPO).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

# Ensure local repository takes precedence over installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from chatarena.rl.mafia_rollout import MafiaEpisodeResult, MafiaRolloutManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mafia LLM with Reinforcement Learning")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base LLM to train (e.g. Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct)",
    )
    parser.add_argument(
        "--target_role",
        type=str,
        default="mafia",
        choices=["mafia", "villager", "doctor", "police"],
        help="Which role the learner policy is training for",
    )
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of training game episodes")
    parser.add_argument("--group_size", type=int, default=4, help="Number of rollouts per group (G in GRPO)")
    parser.add_argument("--num_players", type=int, default=None, help="Fixed player count (e.g. 4, 5, 6, 7). If None, uses dynamic range or default 4.")
    parser.add_argument("--min_players", type=int, default=4, help="Minimum player count when dynamic sampling is enabled")
    parser.add_argument("--max_players", type=int, default=6, help="Maximum player count when dynamic sampling is enabled")
    parser.add_argument("--dynamic_players", action="store_true", help="Randomize player count (min_players ~ max_players) each game episode")
    parser.add_argument("--max_discussion_messages", type=int, default=24)
    parser.add_argument("--max_intent_rounds", type=int, default=48)
    parser.add_argument("--max_total_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--max_new_tokens", type=int, default=80, help="Max generation tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--output_dir", type=str, default="./outputs/mafia_rl", help="Directory to save checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"),
    )
    parser.add_argument("--dry_run", action="store_true", help="Run a fast mock dry-run without loading full GPU weights")
    return parser.parse_args()


def build_dynamic_role_mapping(player_names: List[str], target_role: str, learner_name: str = "Player 1") -> Dict[str, str]:
    """Dynamically builds a balanced role mapping for any player count."""
    mapping = {learner_name: target_role}
    other_players = [p for p in player_names if p != learner_name]
    num_others = len(other_players)

    remaining_roles = []
    if target_role != "mafia":
        remaining_roles.append("mafia")
    if target_role != "doctor" and num_others >= 2:
        remaining_roles.append("doctor")
    if target_role != "police" and num_others >= 3:
        remaining_roles.append("police")
    if len(player_names) >= 7 and target_role == "mafia":
        remaining_roles.append("mafia")  # 2nd mafia in large games

    while len(remaining_roles) < num_others:
        remaining_roles.append("villager")
    remaining_roles = remaining_roles[:num_others]

    for p, r in zip(other_players, remaining_roles):
        mapping[p] = r
    return mapping


class MafiaPolicyTrainer:
    def __init__(self, args):
        self.args = args
        self.device = args.device

        print(f"=== Initializing Mafia RL Trainer on device: {self.device} ===")
        print(f"Model: {args.model_name_or_path} | Target Role: {args.target_role}")
        if args.dynamic_players:
            print(f"Player Count: Dynamic ({args.min_players} ~ {args.max_players} players per episode)")
        elif args.num_players:
            print(f"Player Count: Fixed {args.num_players} players")
        else:
            print(f"Player Count: Default 4 players")

        if not args.dry_run:
            from peft import LoraConfig, get_peft_model, TaskType
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        else:
            print("[Dry-Run Mode] Skipping actual weight initialization.")
            self.model = None
            self.tokenizer = None
            self.optimizer = None

        self.learner_player = "Player 1"
        self.rollout_manager = MafiaRolloutManager(
            num_players=args.num_players,
            min_players=args.min_players,
            max_players=args.max_players,
            dynamic_player_count=args.dynamic_players,
            max_discussion_messages=args.max_discussion_messages,
            max_intent_rounds=args.max_intent_rounds,
            max_total_steps=args.max_total_steps,
            seed=args.seed,
            max_days=3,
        )

    def generate_response(self, prompt: str) -> str:
        """Generate response from the learner model."""
        if self.args.dry_run or self.model is None:
            if "REQUEST intent:" in prompt:
                return "2"
            if "REQUEST speech:" in prompt:
                return "I suspect Player 2 because of the voting."
            # Mock generator for fast testing
            if "Mafia" in prompt:
                return "I choose to eliminate Player 2 <EOS>"
            elif "vote" in prompt.lower():
                return "I vote to eliminate Player 2 <EOS>"
            return "I am innocent and helping the village. <EOS>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        inputs = {key: value[:, -1536:] for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.args.max_new_tokens,
                temperature=self.args.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    def generate_opponent_response(self, player_name: str, prompt: str, role_map: Optional[Dict[str, str]] = None) -> str:
        """Baseline opponent response generator (simple rule/heuristic or fixed prompt)."""
        if "REQUEST intent:" in prompt:
            return "1"
        if "REQUEST speech:" in prompt:
            return "I want to hear the others before voting."
        role = role_map.get(player_name, "villager") if role_map else "villager"
        if role == "doctor":
            return "I choose to protect Player 2 <EOS>"
        elif role == "police":
            return "I choose to investigate Player 1 <EOS>"
        elif "vote" in prompt.lower():
            return "I vote to eliminate Player 1 <EOS>"
        return f"I am {player_name} and I am looking for the Mafia. <EOS>"

    def compute_sequence_log_probs(self, prompt: str, response: str) -> torch.Tensor:
        from training.mafia_policy_gradient import sequence_log_prob
        return sequence_log_prob(self.model, self.tokenizer, self.device, prompt, response)

    def train_step_grpo(self, group_results: List[MafiaEpisodeResult]):
        if self.args.dry_run or self.optimizer is None:
            return 0.0
        from training.mafia_policy_gradient import update_policy
        return update_policy(self.model, self.optimizer, self.device, self.learner_player,
                             group_results, self.compute_sequence_log_probs)

    def train(self):
        print(f"\n--- Starting Mafia RL Training ({self.args.num_episodes} episodes) ---")
        os.makedirs(self.args.output_dir, exist_ok=True)

        learner_wins = 0
        total_episodes = 0
        truncated_episodes = 0

        for epoch in range(self.args.num_episodes // self.args.group_size):
            group_results: List[MafiaEpisodeResult] = []

            # Rollout G games in the group
            for g in range(self.args.group_size):
                curr_names = self.rollout_manager.get_current_player_names()
                curr_role_map = build_dynamic_role_mapping(
                    curr_names, self.args.target_role, self.learner_player
                )
                self.rollout_manager.role_mapping = curr_role_map

                res = self.rollout_manager.rollout_episode(
                    policy_agent_names=[self.learner_player],
                    policy_fn=lambda p, prompt: self.generate_response(prompt),
                    opponent_fn=lambda p, prompt: self.generate_opponent_response(p, prompt, curr_role_map),
                    player_names=curr_names,
                )
                group_results.append(res)
                if res.truncated:
                    truncated_episodes += 1
                else:
                    if res.trajectories[self.learner_player].won:
                        learner_wins += 1
                    total_episodes += 1

            # Optimize policy
            loss = self.train_step_grpo(group_results)
            win_rate = (learner_wins / max(1, total_episodes)) * 100.0
            completed = [r for r in group_results if not r.truncated]
            avg_reward = sum(r.trajectories[self.learner_player].shaped_reward for r in completed) / max(1, len(completed))
            import json
            with open(os.path.join(self.args.output_dir, "episodes.jsonl"), "a", encoding="utf-8") as log:
                for result in group_results:
                    log.write(json.dumps({"winner": result.winner, "reason": result.termination_reason,
                                          "diagnostics": result.diagnostics,
                                          "discussion_endings": result.discussion_endings}) + "\n")

            print(
                f"Epoch {epoch + 1:3d} | "
                f"Completed: {total_episodes:4d} | Truncated: {truncated_episodes} | "
                f"Avg Group Reward: {avg_reward:+.3f} | "
                f"Cumulative Win Rate: {win_rate:5.1f}% | "
                f"Loss: {loss:.4f}"
            )

            # Checkpoint saving
            if (epoch + 1) % 10 == 0 and not self.args.dry_run:
                ckpt_path = os.path.join(self.args.output_dir, f"checkpoint_epoch_{epoch + 1}")
                self.model.save_pretrained(ckpt_path)
                self.tokenizer.save_pretrained(ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

        print("\n=== Training Completed Successfully ===")
        if not self.args.dry_run:
            final_path = os.path.join(self.args.output_dir, "final_model")
            self.model.save_pretrained(final_path)
            self.tokenizer.save_pretrained(final_path)
            print(f"Final model saved to: {final_path}")


def main():
    args = parse_args()
    trainer = MafiaPolicyTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
