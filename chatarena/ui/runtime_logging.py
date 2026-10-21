"""Timestamped UI and model-request diagnostics without game contents."""
import logging
import os
import sys

from .mixed_lineup import make_gpt_backend


def configure_ui_logging(game):
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, force=True,
        format=f"%(asctime)s %(levelname)s [{game} pid=%(process)d] %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    gpt = make_gpt_backend()
    logging.info("UI_START game=%s", game)
    logging.info("GPT_CONFIG model=%s endpoint=%s key_env=%s key_present=%s status=not_probed",
                 gpt["model"], gpt["base_url"], gpt["api_key_env"],
                 bool(os.environ.get(gpt["api_key_env"])))
