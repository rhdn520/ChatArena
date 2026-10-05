#!/usr/bin/env python3
"""Launch the multiplayer Liar Game UI from a Slurm job."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatarena.ui.liar_game_gradio import build_demo


def main():
    parser = argparse.ArgumentParser(description="Play Liar Game with humans and AI")
    parser.add_argument("--config", default="examples/liar_game_qwen3_5_9b.json")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")

    demo = build_demo(args.config)
    print(f"Gradio listening on compute-node localhost:{args.port}", flush=True)
    demo.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
