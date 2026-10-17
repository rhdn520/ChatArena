#!/usr/bin/env python3
"""Run the live workspace against a local chat API and retain readable logs."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def transcript(messages):
    lines = []
    for index, message in enumerate(messages, 1):
        audience = "공개" if message.visible_to == "all" else f"비밀 → {message.visible_to}"
        lines.extend([f"[{index:03d} | turn {message.turn} | {audience} | {message.agent_name}]",
                      message.content, ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=["mafia", "liar_game"], required=True)
    parser.add_argument("--base-url", required=True, help="Local OpenAI-compatible URL including /v1")
    parser.add_argument("--model", required=True, help="Exact served model ID")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", type=Path, required=True, help="A new directory; existing results are never overwritten")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--enable-thinking", action="store_true", help="Enable model reasoning for players and moderator")
    parser.add_argument("--thinking-token-budget", type=int)
    parser.add_argument("--control-thinking-token-budget", type=int)
    parser.add_argument("--request-timeout", type=float, default=60, help="API/controller timeout in seconds")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--players", type=int, help="Default: 7 for Mafia, 5 for Liar Game")
    parser.add_argument("--clue-rounds", type=int, help="Liar Game clue rounds; defaults to the existing game config")
    parser.add_argument("--no-moderator", action="store_true", help="Disable the Mafia LLM moderator for comparison")
    parser.add_argument("--moderator-model", help="Moderator model on the same API; defaults to the player model")
    args = parser.parse_args()
    sampling = {name: getattr(args, name) for name in (
        "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
        if getattr(args, name) is not None}
    moderator_sampling = dict(sampling, temperature=args.temperature if args.enable_thinking else 0.3)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    if args.max_steps < 1 or args.max_tokens < 1:
        parser.error("max-steps and max-tokens must be positive")
    if args.request_timeout <= 0:
        parser.error("request-timeout must be positive")
    for budget in (args.thinking_token_budget, args.control_thinking_token_budget):
        if budget is not None and (not args.enable_thinking or not 0 < budget < args.max_tokens):
            parser.error("thinking budgets require --enable-thinking and must be smaller than max-tokens")
    player_count = args.players or (7 if args.game == "mafia" else 5)
    if player_count < 3 or player_count > 10:
        parser.error("players must be between 3 and 10")
    if args.game != "mafia" and (args.no_moderator or args.moderator_model):
        parser.error("Moderator options apply only to Mafia")
    if args.no_moderator and args.moderator_model:
        parser.error("--moderator-model cannot be combined with --no-moderator")
    moderator_model = (args.moderator_model or args.model) if args.game == "mafia" and not args.no_moderator else None
    os.environ["OPENAI_BASE_URL"] = args.base_url
    os.environ.setdefault("OPENAI_API_KEY", "local-model")
    from chatarena.arena import Arena
    from chatarena.config import ArenaConfig
    from chatarena.backends import openai as backend

    args.output.mkdir(parents=True, exist_ok=False)
    backend.client = backend.client.with_options(timeout=args.request_timeout, max_retries=0)
    create = backend.client.chat.completions.create
    root = Path(__file__).resolve().parents[1]
    for seed in args.seeds:
        out = args.output / f"seed_{seed}"
        out.mkdir()
        config = ArenaConfig.load(str(root / "examples" / f"{args.game}.json"))
        while len(config.players) < player_count:
            player = copy.deepcopy(config.players[-1])
            old_name = player.name
            player.name = f"Player {len(config.players) + 1}"
            player.role_desc = player.role_desc.replace(old_name, player.name)
            config.players.append(player)
        config.players = config.players[:player_count]
        config.environment["seed" if args.game == "mafia" else "random_seed"] = seed
        if args.game == "liar_game" and args.clue_rounds is not None:
            config.environment["clue_rounds"] = args.clue_rounds
        for player in config.players:
            player.backend.update(model=args.model, max_tokens=args.max_tokens,
                                  **sampling, enable_thinking=args.enable_thinking,
                                  thinking_token_budget=args.thinking_token_budget,
                                  control_thinking_token_budget=args.control_thinking_token_budget,
                                  system_messages_at_start=True)
        if args.game == "mafia":
            config.environment["discussion_moderator"] = None if moderator_model is None else {
                "name": "Moderator",
                "role_desc": "당신은 마피아 게임의 낮 토론 종료 여부를 판단하는 사회자입니다.",
                "backend": {"backend_type": "openai-chat", "model": moderator_model,
                            **moderator_sampling, "max_tokens": args.max_tokens, "enable_thinking": args.enable_thinking,
                            "thinking_token_budget": args.thinking_token_budget,
                            "control_thinking_token_budget": args.control_thinking_token_budget,
                            "system_messages_at_start": True},
            }
        arena = Arena.from_config(config)
        if arena.discussion is not None:
            # Intent batches can include more players than executor workers.
            # Allow every queued request its configured generation timeout.
            arena.discussion.request_timeout = args.request_timeout * player_count
        lock = threading.Lock()
        calls = []

        def logged_create(**kwargs):
            record = {"phase": arena.environment.phase, "started": time.time(),
                      "request": kwargs}
            with lock:
                record["call"] = len(calls) + 1
                calls.append(record)
            try:
                result = create(**kwargs)
                record["response"] = result.model_dump(mode="json")
                return result
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                record["elapsed_seconds"] = time.time() - record["started"]
                with lock, (out / "api_calls.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

        backend.client.chat.completions.create = logged_create
        error = None
        steps = 0
        try:
            while not arena.current_timestep.terminal and steps < args.max_steps:
                arena.step()
                steps += 1
                messages = arena.environment.message_pool.get_all_messages()
                (out / "conversation.txt").write_text(transcript(messages), encoding="utf-8")
            if not arena.current_timestep.terminal:
                raise RuntimeError(f"Game did not terminate within {args.max_steps} steps")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            arena.close()
            backend.client.chat.completions.create = create
            messages = arena.environment.message_pool.get_all_messages()
            (out / "conversation.txt").write_text(transcript(messages), encoding="utf-8")
            (out / "public_conversation.txt").write_text(
                transcript([m for m in messages if m.visible_to == "all"]), encoding="utf-8")
            write_json(out / "messages.json", [dataclasses.asdict(m) for m in messages])
            diagnostics = [] if arena.discussion is None else [
                dataclasses.asdict(r) for r in arena.discussion.records]
            for record in diagnostics:
                # Controller eligibility is for RL bookkeeping, not expert IL labels.
                record["format_eligible"] = record.pop("trainable")
                record["expert_approved"] = False
            write_json(out / "decisions.json", diagnostics)
            moderator_lines = []
            for record in diagnostics:
                if record["request"]["kind"] == "moderation":
                    moderator_lines.extend([
                        f"[{record['request']['phase']} | version {record['request']['version']}]",
                        record["response"],
                        f"valid={record['valid']} stale={record['stale']} error={record['error']}", "",
                    ])
            (out / "moderator_decisions.txt").write_text("\n".join(moderator_lines), encoding="utf-8")
            readable_calls = []
            for call in calls:
                readable_calls.append(f"=== API call {call['call']} | {call['phase']} ===")
                for msg in call["request"]["messages"]:
                    readable_calls.extend([f"[{msg['role']} / {msg.get('name', '')}]", msg["content"]])
                readable_calls.extend(["[원문 API 응답]", json.dumps(
                    call.get("response", {"error": call.get("error")}), ensure_ascii=False, indent=2), ""])
            (out / "model_io.txt").write_text("\n".join(readable_calls), encoding="utf-8")
            reasoning_lines = []
            reasoning_calls = 0
            for call in sorted(calls, key=lambda c: c["call"]):
                for choice in call.get("response", {}).get("choices", []):
                    message = choice.get("message", {})
                    reasoning = message.get("reasoning") or message.get("reasoning_content")
                    if reasoning:
                        reasoning_calls += 1
                        reasoning_lines.extend([
                            f"=== API call {call['call']} | {call['phase']} ===",
                            "[reasoning]", reasoning, "[최종 응답]", message.get("content") or "", "",
                        ])
            (out / "reasoning.txt").write_text("\n".join(reasoning_lines), encoding="utf-8")
            write_json(out / "summary.json", {
                "game": args.game, "seed": seed, "model": args.model,
                "players": player_count, "enable_thinking": args.enable_thinking,
                "clue_rounds": getattr(arena.environment, "clue_rounds", None),
                "moderator_enabled": moderator_model is not None,
                "moderator_model": moderator_model,
                "temperature": args.temperature, "max_tokens": args.max_tokens,
                "sampling_parameters": sampling,
                "request_timeout": args.request_timeout, "reasoning_calls": reasoning_calls,
                "thinking_token_budget": args.thinking_token_budget,
                "control_thinking_token_budget": args.control_thinking_token_budget,
                "terminal": arena.environment.is_terminal(), "steps": steps,
                "error": error, "rewards": arena.environment.get_rewards(),
                "winner": getattr(arena.environment, "winner", None),
                "termination_reason": getattr(arena.environment, "termination_reason", None),
                "api_calls": len(calls),
                "rejected_actions": [],
                "decision_schema_version": 2,
                "training_labels": "format_eligible checks format/freshness only; expert_approved is false until independently reviewed. No automatic expert/IL labels.",
                "note": "Seed controls game randomness; API sampling is not seeded. Target recognition is separate from strict response format compliance.",
            })
        print(f"{args.game} seed={seed}: {out / 'conversation.txt'}", flush=True)
        if error:
            raise RuntimeError(error)


if __name__ == "__main__":
    main()
