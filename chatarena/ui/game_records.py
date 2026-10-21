"""Write completed UI games without credentials or private observations."""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from ..backends.human import Human


def save_finished_game(arena, game, session_id, game_number=1):
    env = arena.environment
    if not env.is_terminal():
        return None
    root = Path(__file__).resolve().parents[2]
    job = os.environ.get("SLURM_JOB_ID", "local")
    path = (root / "outputs" / "mixed_games" / f"job_{job}" / game
            / f"{session_id}-game-{game_number:04d}.json")
    record = {
        "game": game, "session_id": session_id, "game_number": game_number,
        "job_id": job, "finished_at": datetime.now(timezone.utc).isoformat(),
        "players": [{"name": p.name,
                     "type": "human" if isinstance(p.backend, Human) else "ai",
                     "model": None if isinstance(p.backend, Human) else getattr(p.backend, "model", None)}
                    for p in arena.players],
        "rewards": env.get_rewards(),
        "messages": [{"speaker": m.agent_name, "turn": m.turn,
                      "timestamp": m.timestamp, "type": m.msg_type, "content": m.content}
                     for m in env.message_pool.get_all_messages() if m.visible_to == "all"],
    }
    if game == "liar":
        record["result"] = {name: getattr(env, name) for name in (
            "winner", "liar_name", "topic", "secret_word", "word_guess", "votes")}
        record["clue_rounds"] = env.clue_rounds
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        logging.info("GAME_SAVED game=%s session=%s game_number=%d path=%s roster=%s",
                     game, session_id, game_number, path, record["players"])
        return str(path)
    except OSError:
        logging.exception("GAME_SAVE_FAILED game=%s session=%s path=%s", game, session_id, path)
        return None
