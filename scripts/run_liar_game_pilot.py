#!/usr/bin/env python3
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatarena.arena import Arena
from chatarena.config import ArenaConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Run one complete Liar Game pilot")
    parser.add_argument("--config", default="examples/liar_game.json")
    parser.add_argument("--history", default="outputs/liar_game_history.json")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    config = ArenaConfig.load(args.config)
    if args.seed is not None:
        config.environment["random_seed"] = args.seed
    arena = Arena.from_config(config)
    timestep = arena.current_timestep
    steps = 0
    while not timestep.terminal and steps < args.max_steps:
        timestep = arena.step()
        steps += 1

    if not timestep.terminal:
        raise RuntimeError(
            f"Liar Game did not terminate within {args.max_steps} steps"
        )

    history_dir = os.path.dirname(os.path.abspath(args.history))
    os.makedirs(history_dir, exist_ok=True)
    arena.save_history(args.history)
    env = arena.environment
    summary_path = args.summary or os.path.splitext(args.history)[0] + "_summary.json"
    summary_dir = os.path.dirname(os.path.abspath(summary_path))
    os.makedirs(summary_dir, exist_ok=True)
    summary = {
        "seed": args.seed,
        "liar": env.liar_name,
        "topic": env.topic,
        "secret_word": env.secret_word,
        "reveal_topic": env.reveal_topic,
        "clue_rounds": env.clue_rounds,
        "votes": env.votes,
        "accused_player": env.accused_player,
        "word_guess": env.word_guess,
        "winner": env.winner,
        "rewards": env.get_rewards(),
        "steps": steps,
        "history": os.path.abspath(args.history),
        "malformed_votes": sum(target is None for target in env.votes.values()),
    }
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)

    print(f"Seed: {args.seed}")
    print(f"Liar: {env.liar_name}")
    print(f"Topic: {env.topic}")
    print(f"Secret word: {env.secret_word}")
    print(f"Winner: {env.winner}")
    print(f"Rewards: {env.get_rewards()}")
    print(f"History: {os.path.abspath(args.history)}")
    print(f"Summary: {os.path.abspath(summary_path)}")


if __name__ == "__main__":
    main()
