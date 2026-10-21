"""Run the dedicated room-based Liar Game UI from the mixed-games Slurm job."""
import argparse
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from chatarena.ui.liar_game_gradio import build_demo
from chatarena.ui.runtime_logging import configure_ui_logging


def main():
    configure_ui_logging("Liar")
    parser = argparse.ArgumentParser(description="라이어게임 전용 웹 UI")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    demo = build_demo(str(ROOT / "examples/liar_game.json"))
    _, _, share_url = demo.launch(
        server_name="127.0.0.1", server_port=args.port,
        share=True, show_error=True, prevent_thread_lock=True,
    )
    print(f"LIAR_PUBLIC_URL={share_url}", flush=True)
    print(f"라이어게임 주소: {share_url}", flush=True)
    demo.block_thread()


if __name__ == "__main__":
    main()
