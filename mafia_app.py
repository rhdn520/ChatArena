"""Run the dedicated Mafia UI: python -u mafia_app.py --port 8081."""
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from chatarena.ui.mafia_app import MafiaUI


def main():
    parser = argparse.ArgumentParser(description="마피아 전용 웹 UI")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    print(f"마피아 UI 시작 중: http://localhost:{args.port}", flush=True)
    ui = MafiaUI()
    ui.demo.queue()
    ui.demo.launch(server_name="127.0.0.1", server_port=args.port, show_error=True, share=True)


if __name__ == "__main__":
    main()
