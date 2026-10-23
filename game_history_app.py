"""Browse saved games without starting a game or contacting a model."""
import argparse
from pathlib import Path

from chatarena.ui.game_history import build_demo


def main():
    parser = argparse.ArgumentParser(description="저장된 게임 대화 조회")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--share", action="store_true", help="전체 기록 조회용 공유 링크 생성")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / "outputs"
    demo = build_demo(root)
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                share=args.share, show_error=True)


if __name__ == "__main__":
    main()
