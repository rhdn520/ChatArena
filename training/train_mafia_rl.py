#!/usr/bin/env python3
"""
Mafia RL Training Script (End-to-End Language RL via GRPO / Policy Gradient with LoRA)

Designed for single-GPU training (e.g. RTX 6000 / Pro 6000 with 24GB~48GB VRAM).
Plays multi-turn Mafia games in ChatArena, observes full game outcomes (winner, survival),
calculates rewards, and optimizes the policy using Group Relative Policy Optimization (GRPO).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List

# Ensure local repository takes precedence over installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False

from chatarena.environments.mafia import MAFIA, Mafia
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
            discussion_rounds=1,
            max_days=3,
        )

    def generate_response(self, prompt: str) -> str:
        """Generate response from the learner model."""
        if self.args.dry_run or self.model is None:
            # Mock generator for fast testing
            if "Mafia" in prompt:
                return "I choose to eliminate Player 2 <EOS>"
            elif "vote" in prompt.lower():
                return "I vote to eliminate Player 2 <EOS>"
            return "I am innocent and helping the village. <EOS>"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(self.device)
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
        role = role_map.get(player_name, "villager") if role_map else "villager"
        if role == "doctor":
            return "I choose to protect Player 2 <EOS>"
        elif role == "police":
            return "I choose to investigate Player 1 <EOS>"
        elif "vote" in prompt.lower():
            return "I vote to eliminate Player 1 <EOS>"
        return f"I am {player_name} and I am looking for the Mafia. <EOS>"

    def compute_sequence_log_probs(self, prompt: str, response: str) -> torch.Tensor:
        """Calculates token log probabilities for a (prompt, response) pair."""
        full_text = prompt + response
        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        full_enc = self.tokenizer(full_text, return_tensors="pt").to(self.device)

        prompt_len = prompt_enc["input_ids"].shape[1]
        input_ids = full_enc["input_ids"]
        target_ids = input_ids.clone()
        target_ids[:, :prompt_len] = -100  # Mask out prompt tokens

        logits = self.model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )
        return -loss.sum()  # Total log prob of response tokens

    def train_step_grpo(self, group_results: List[MafiaEpisodeResult]):
        """
        Updates the policy using Group Relative Policy Optimization (GRPO).
        Normalizes rewards within the group to estimate advantages without a critic.
        """
        if self.args.dry_run or self.optimizer is None:
            return 0.0

        rewards = torch.tensor(
            [res.trajectories[self.learner_player].shaped_reward for res in group_results],
            device=self.device,
            dtype=torch.float32,
        )
        mean_r = rewards.mean()
        std_r = rewards.std() + 1e-8
        advantages = (rewards - mean_r) / std_r

        self.optimizer.zero_grad()
        total_loss = 0.0

        for i, res in enumerate(group_results):
            adv = advantages[i]
            traj = res.trajectories[self.learner_player]

            for turn in traj.turns:
                log_prob = self.compute_sequence_log_probs(turn.prompt, turn.response)
                # Policy gradient loss: - advantage * log_prob
                loss = -adv * log_prob / len(traj.turns)
                loss.backward()
                total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss / len(group_results)

    def train(self):
        print(f"\n--- Starting Mafia RL Training ({self.args.num_episodes} episodes) ---")
        os.makedirs(self.args.output_dir, exist_ok=True)

        learner_wins = 0
        total_episodes = 0

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
                if res.trajectories[self.learner_player].won:
                    learner_wins += 1
                total_episodes += 1

            # Optimize policy
            loss = self.train_step_grpo(group_results)
            win_rate = (learner_wins / total_episodes) * 100.0
            avg_reward = sum(r.trajectories[self.learner_player].shaped_reward for r in group_results) / len(group_results)

            print(
                f"Epoch {epoch + 1:3d} | "
                f"Episodes: {total_episodes:4d} | "
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
