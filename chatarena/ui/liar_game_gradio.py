"""Multi-human Gradio controller for the existing Liar Game environment."""

import copy
import html
import secrets
from threading import RLock

from ..arena import Arena
from ..config import AgentConfig, ArenaConfig, BackendConfig


MAX_PLAYERS = 10
MAX_CLUE_ROUNDS = 10
RECENT_MESSAGES = 10


def validate_player_counts(human_count, ai_count):
    try:
        humans, ais = int(human_count), int(ai_count)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Enter whole-number human and AI counts.") from exc
    if humans != human_count or ais != ai_count:
        raise ValueError("Enter whole-number human and AI counts.")
    if humans < 1 or ais < 0 or not 3 <= humans + ais <= MAX_PLAYERS:
        raise ValueError("Use at least 1 human, 0 or more AI, and 3–10 players total.")
    return humans, ais


def validate_clue_rounds(value):
    try:
        rounds = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Enter a whole-number clue round count.") from exc
    if rounds != value or not 1 <= rounds <= MAX_CLUE_ROUNDS:
        raise ValueError(f"Use 1–{MAX_CLUE_ROUNDS} clue rounds.")
    return rounds


def validate_nickname(value):
    nickname = (value or "").strip()
    if len(nickname) > 32 or any(ord(char) < 32 for char in nickname):
        raise ValueError("Use a nickname of at most 32 printable characters.")
    return nickname


class HumanLiarGame:
    """Advance AI turns while leaving human turns for their own browsers."""

    def __init__(self, arena: Arena, human_names):
        if isinstance(human_names, str):
            human_names = [human_names]
        if not human_names or any(
            name not in arena.environment.player_names for name in human_names
        ):
            raise ValueError("Every human must be a configured player")
        self.arena = arena
        self.human_names = frozenset(human_names)
        self.timestep = arena.current_timestep
        self.pending_votes = {}

    @property
    def environment(self):
        return self.arena.environment

    @property
    def next_is_human(self) -> bool:
        if self.environment.phase == "vote":
            return any(name not in self.pending_votes for name in self.human_names)
        return (
            not self.timestep.terminal
            and self.environment.get_next_player() in self.human_names
        )

    def is_human_turn(self, human_name: str) -> bool:
        if self.environment.phase == "vote":
            return (
                not self.timestep.terminal
                and human_name in self.human_names
                and human_name not in self.pending_votes
            )
        return self.next_is_human and self.environment.get_next_player() == human_name

    def _commit_votes(self):
        if len(self.pending_votes) != self.environment.num_players:
            return
        for player in self.environment.player_names:
            self.timestep = self.environment.step(player, self.pending_votes[player])
        self.arena.current_timestep = self.timestep

    def step_ai_once(self):
        if self.timestep.terminal:
            return False
        if self.environment.phase == "vote":
            if any(name not in self.pending_votes for name in self.human_names):
                return False
            ai_name = next(
                (name for name in self.environment.player_names
                 if name not in self.human_names and name not in self.pending_votes),
                None,
            )
            if ai_name is None:
                self._commit_votes()
                return False
            player = self.arena.name_to_player[ai_name]
            self.pending_votes[ai_name] = player(self.environment.get_observation(ai_name))
            self._commit_votes()
            return True
        if self.next_is_human:
            return False
        self.timestep = self.arena.step()
        self.arena.current_timestep = self.timestep
        return True

    def advance_ai(self):
        """Yield after each AI action so the browser can show the live log."""
        limit = self.environment.num_players * (self.environment.clue_rounds + 1) + 1
        for _ in range(limit):
            if not self.step_ai_once():
                return
            yield self.timestep
        raise RuntimeError("AI turns exceeded the maximum expected game length")

    def submit(self, human_name: str, text: str = "", vote: str = ""):
        if human_name not in self.human_names or not self.is_human_turn(human_name):
            raise ValueError("Wait for your turn before submitting an action.")
        phase = self.environment.phase
        if phase == "vote":
            if vote not in self.environment.player_names or vote == human_name:
                raise ValueError("Choose one other player for your vote.")
            action = f"I vote for {vote}"
            self.pending_votes[human_name] = action
            self._commit_votes()
            return self.timestep
        else:
            action = (text or "").strip()
            if not action:
                raise ValueError("Enter a clue or a final word guess.")
            if phase == "liar_guess" and "\n" in action:
                raise ValueError("Enter only one final word guess.")
        self.timestep = self.environment.step(human_name, action)
        self.arena.current_timestep = self.timestep
        return self.timestep

    def snapshot(self, human_name: str):
        if human_name not in self.human_names:
            raise ValueError("Unknown human player")
        env = self.environment
        visible = env.get_observation(human_name)
        role_message = next(
            (
                msg.content
                for msg in visible
                if msg.agent_name == "Moderator"
                and msg.turn == 0
                and msg.content.startswith(("You are a non-liar.", "You are the Liar."))
            ),
            "Role card is not available.",
        )
        public_messages = [msg for msg in visible if msg.visible_to == "all"]
        next_player = None if self.timestep.terminal else env.get_next_player()
        human_turn = self.is_human_turn(human_name)
        current_instruction = next(
            (msg.content for msg in reversed(visible)
             if msg.agent_name == "Moderator" and msg.msg_type == "instruction"),
            None,
        ) if human_turn else None
        conversation_messages = [
            (msg.agent_name, msg.content)
            for msg in visible
            if (
                (msg.agent_name == "Moderator" and msg.msg_type == "instruction")
                or (msg.agent_name != "Moderator" and msg.msg_type != "vote")
            )
        ]
        return {
            "phase": env.phase,
            "round": env.current_round,
            "rounds": env.clue_rounds,
            "human_name": human_name,
            "human_turn": human_turn,
            "current_instruction": current_instruction,
            "next_player": "other players" if env.phase == "vote" else next_player,
            "terminal": self.timestep.terminal,
            "winner": env.winner,
            "role_message": role_message,
            "public_topic": env.topic if env.reveal_topic or self.timestep.terminal else None,
            "rules": self.arena.global_prompt or "",
            "public_messages": [
                (msg.agent_name, msg.content) for msg in public_messages
            ],
            "conversation_messages": conversation_messages,
            "players": list(env.player_names),
            "votes": dict(env.votes) if self.timestep.terminal else {},
            "votes_cast": len(self.pending_votes) if env.phase == "vote" else len(env.votes),
            "my_vote_cast": human_name in self.pending_votes,
            "accused_player": env.accused_player,
            "rewards": env.get_rewards() if self.timestep.terminal else None,
            "result": {
                "topic": env.topic,
                "secret_word": env.secret_word,
                "liar_name": env.liar_name,
                "word_guess": env.word_guess,
            } if self.timestep.terminal else None,
        }


def create_human_game(
    config_path: str, human_count: int, ai_count: int, reveal_topic: bool,
    clue_rounds=None, human_names=None,
):
    human_count, ai_count = validate_player_counts(human_count, ai_count)
    config = ArenaConfig.load(config_path)
    if clue_rounds is not None:
        config.environment["clue_rounds"] = validate_clue_rounds(clue_rounds)
    ai_backend = copy.deepcopy(config.players[0]["backend"])
    config["global_prompt"] = (
        (config.get("global_prompt") or "")
        + " Voting is a private simultaneous ballot: cast one vote without "
        "seeing anyone else's choice or the running totals. Ballot choices "
        "and totals are disclosed only with the final result."
    )
    config.environment["reveal_topic"] = bool(reveal_topic)
    player_names = [f"Player {index + 1}" for index in range(human_count + ai_count)]
    if human_names is None:
        human_names = player_names[:human_count]
    human_names = list(human_names)
    if len(human_names) != human_count or len(set(human_names)) != human_count or any(
        name not in player_names for name in human_names
    ):
        raise ValueError("Human seats must be distinct configured players.")
    config.players = []
    for name in player_names:
        config.players.append(
            AgentConfig(
                name=name,
                role_desc=(
                    f"You are {name}. Follow the latest Moderator instruction. "
                    "Do not vote until voting begins."
                ),
                backend=(
                    BackendConfig(backend_type="human")
                    if name in human_names
                    else copy.deepcopy(ai_backend)
                ),
            )
        )
    return HumanLiarGame(Arena.from_config(config), human_names)


class LiarGameRoom:
    """One game shared by several browser sessions, with seat-specific access."""

    def __init__(self, config_path, room_code, human_count, ai_count, reveal_topic, clue_rounds=2):
        self.human_count, self.ai_count = validate_player_counts(human_count, ai_count)
        self.config_path = config_path
        self.room_code = room_code
        self.reveal_topic = bool(reveal_topic)
        self.clue_rounds = validate_clue_rounds(clue_rounds)
        self.seats = [f"Seat {index + 1}" for index in range(human_count)]
        self.ai_seats = [f"AI {index + 1}" for index in range(ai_count)]
        self.player_for_seat = {}
        self.player_for_ai = {}
        self.cumulative_scores = {
            identity: 0.0 for identity in self.seats + self.ai_seats
        }
        self.scored_game_number = 0
        self.nicknames = {}
        self.seat_tokens = {}
        self.seat_codes = {}
        self.tokens = {}
        self.game = None
        self.game_number = 0
        self.restart_ready = set()
        self.lock = RLock()

    def _start_game(self):
        player_names = [
            f"Player {index + 1}" for index in range(self.human_count + self.ai_count)
        ]
        selected = secrets.SystemRandom().sample(player_names, self.human_count)
        self.player_for_seat = dict(zip(self.seats, selected))
        self.player_for_ai = dict(zip(
            self.ai_seats, (name for name in player_names if name not in selected)
        ))
        self.game = create_human_game(
            self.config_path, self.human_count, self.ai_count,
            self.reveal_topic, self.clue_rounds, selected,
        )
        self.game_number += 1
        self.restart_ready.clear()

    def _record_score(self):
        if (
            self.game is None or not self.game.timestep.terminal
            or self.scored_game_number == self.game_number
        ):
            return
        rewards = self.game.environment.get_rewards()
        for identity, player in {**self.player_for_seat, **self.player_for_ai}.items():
            self.cumulative_scores[identity] += rewards[player]
        self.scored_game_number = self.game_number

    def join(self, seat_code="", nickname=""):
        with self.lock:
            nickname = validate_nickname(nickname)
            code = (seat_code or "").strip().upper()
            if code:
                seat = next(
                    (seat for seat, value in self.seat_codes.items() if value == code),
                    None,
                )
                if seat is None:
                    raise ValueError("Invalid personal seat code for this room.")
            else:
                seat = next(
                    (seat for seat in self.seats if seat not in self.seat_tokens),
                    None,
                )
                if seat is None:
                    raise ValueError("All human seats are taken; use your seat code to rejoin.")
            if nickname and any(
                existing != seat and current.casefold() == nickname.casefold()
                for existing, current in self.nicknames.items() if current
            ):
                raise ValueError("That nickname is already in use in this room.")
            if code:
                self.tokens.pop(self.seat_tokens[seat], None)
            else:
                self.seat_codes[seat] = secrets.token_hex(8).upper()
            if nickname or seat not in self.nicknames:
                self.nicknames[seat] = nickname
            token = secrets.token_urlsafe(24)
            self.seat_tokens[seat] = token
            self.tokens[token] = seat
            if len(self.seat_tokens) == self.human_count and self.game is None:
                self._start_game()
            return token

    def view(self, token):
        with self.lock:
            seat = self.tokens.get(token)
            if seat is None:
                raise ValueError("Join the room again using your personal seat code.")
            self._record_score()
            scoreboard = [
                {"identity": identity,
                 "label": self.nicknames.get(identity) or identity,
                 "type": "Human" if identity in self.seats else "AI",
                 "score": self.cumulative_scores[identity]}
                for identity in self.seats + self.ai_seats
            ]
            common = {
                "room_code": self.room_code,
                "seat_code": self.seat_codes[seat],
                "seat": seat,
                "nickname": self.nicknames[seat],
                "human_count": self.human_count,
                "ai_count": self.ai_count,
                "rounds": self.clue_rounds,
                "joined_humans": len(self.seat_tokens),
                "game_number": self.game_number,
                "restart_ready": len(self.restart_ready),
                "i_am_ready": seat in self.restart_ready,
                "my_score": self.cumulative_scores[seat],
                "scoreboard": scoreboard,
            }
            if self.game is None:
                return {"lobby": True, "human_name": seat, **common}
            name = self.player_for_seat[seat]
            snapshot = self.game.snapshot(name)
            if snapshot["terminal"]:
                snapshot["participants"] = {
                    player: {
                        "type": "Human" if player in self.player_for_seat.values() else "AI",
                        "nickname": next(
                            (self.nicknames[owner] for owner, assigned in self.player_for_seat.items()
                             if assigned == player), "",
                        ),
                    }
                    for player in snapshot["players"]
                }
            return {"lobby": False, **snapshot, **common}

    def submit(self, token, text="", vote=""):
        with self.lock:
            seat = self.tokens.get(token)
            if seat is None:
                raise ValueError("Your seat is no longer active; rejoin with your seat code.")
            if self.game is None:
                raise ValueError("Waiting for all human players to join.")
            self.game.submit(self.player_for_seat[seat], text, vote)
            return self.view(token)

    def ready_for_restart(self, token):
        with self.lock:
            seat = self.tokens.get(token)
            if seat is None:
                raise ValueError("Your seat is no longer active; rejoin with your seat code.")
            if self.game is None or not self.game.timestep.terminal:
                raise ValueError("Finish the current game before restarting.")
            self._record_score()
            self.restart_ready.add(seat)
            if len(self.restart_ready) == self.human_count:
                self._start_game()
            return self.view(token)

    def step_ai_once(self):
        with self.lock:
            return bool(self.game and self.game.step_ai_once())


class RoomRegistry:
    def __init__(self, config_path):
        self.config_path = config_path
        self.rooms = {}
        self.lock = RLock()

    def create(self, human_count, ai_count, reveal_topic, clue_rounds=2, nickname=""):
        humans, ais = validate_player_counts(human_count, ai_count)
        rounds = validate_clue_rounds(clue_rounds)
        nickname = validate_nickname(nickname)
        with self.lock:
            if len(self.rooms) >= 32:
                raise ValueError("Too many active rooms; restart the UI job to clear them.")
            code = secrets.token_hex(6).upper()
            while code in self.rooms:
                code = secrets.token_hex(6).upper()
            room = LiarGameRoom(self.config_path, code, humans, ais, reveal_topic, rounds)
            self.rooms[code] = room
            token = room.join(nickname=nickname)
            return room, token

    def get(self, room_code):
        code = (room_code or "").strip().upper()
        with self.lock:
            room = self.rooms.get(code)
        if room is None:
            raise ValueError("Room code not found.")
        return room


def _safe(value):
    return html.escape(str(value), quote=True)


def render_snapshot(snapshot, gr, clear_input=False):
    """Render only the human's private card and public conversation."""
    if snapshot is None:
        return (
            "Choose player counts to create a room, or join an existing room.",
            "<div class='liar-card'>Your private role will appear here.</div>",
            "<div class='liar-empty'>The public conversation will appear here.</div>",
            "<div class='liar-empty'>Voting has not started.</div>",
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=None),
            gr.update(visible=False, interactive=False),
            "<div class='liar-empty'>Game information will appear here.</div>",
            gr.update(visible=False, interactive=False),
        )

    if snapshot["lobby"]:
        status = (
            f"### Room {_safe(snapshot['room_code'])} · "
            f"waiting for humans ({snapshot['joined_humans']}/{snapshot['human_count']})"
            f" · {snapshot['rounds']} clue rounds"
        )
        card = (
            "<div class='liar-card'><b>Your seat: "
            + _safe(snapshot["human_name"])
            + (f" ({_safe(snapshot['nickname'])})" if snapshot["nickname"] else "")
            + "</b><p>Share the room code with other players. "
            "Keep this personal rejoin code private: <b>"
            + _safe(snapshot["seat_code"])
            + "</b>. Your role appears when all humans have joined.</p></div>"
        )
        return (
            status,
            card,
            "<div class='liar-empty'>The game will start when all human players join.</div>",
            "<div class='liar-empty'>Voting has not started.</div>",
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=None),
            gr.update(visible=False, interactive=False),
            "<div class='liar-empty'>Game information appears when everyone joins.</div>",
            gr.update(visible=False, interactive=False),
        )

    phase = snapshot["phase"]
    if snapshot["terminal"]:
        result = "Liar wins" if snapshot["winner"] == "liar" else "Non-liars win"
        status = f"### Game over · {result}"
    elif snapshot["human_turn"]:
        phase_label = {
            "clue": f"Clue round {snapshot['round']}/{snapshot['rounds']}",
            "vote": "Vote for the Liar",
            "liar_guess": "Final word guess",
        }[phase]
        status = f"### Your turn · {phase_label}"
    else:
        status = f"### Waiting for {_safe(snapshot['next_player'])} · {phase.replace('_', ' ')}"

    status += (
        f"  \nRoom {_safe(snapshot['room_code'])} · Game {snapshot.get('game_number', 1)} · "
        f"{snapshot['human_count']} human + {snapshot['ai_count']} AI"
        f" · {snapshot['rounds']} clue rounds"
    )
    if "my_score" in snapshot:
        status += f" · Your cumulative score: {snapshot['my_score']:.1f}"
    if snapshot["terminal"]:
        status += (
            f"  \nNext game: {snapshot.get('restart_ready', 0)}/{snapshot['human_count']} "
            "humans ready. A new game starts when everyone is ready."
        )
    if snapshot["human_turn"] and snapshot["current_instruction"]:
        status += f"  \n**Your instruction:** {_safe(snapshot['current_instruction'])}"
    topic = snapshot["public_topic"] or "Hidden until the game ends"
    public_info = (
        "<div class='liar-info'><b>Game information</b>"
        f"<p>Topic: <b>{_safe(topic)}</b> · {snapshot['rounds']} clue rounds · "
        "one private vote per player. Votes are revealed only at the final result.</p>"
        "<p>Non-Liars give indirect clues and try to identify the Liar. "
        "The Liar blends in and tries to infer the secret word. "
        "A wrong accusation gives the Liar the win; a correct accusation "
        "gives the Liar one final word guess. Correct guess: Liar wins; "
        "wrong guess: Non-Liars win. Ties use player order.</p>"
        f"<details><summary>Full shared rules</summary><p>{_safe(snapshot['rules'])}</p></details>"
        "</div>"
    )
    nickname_label = f" ({_safe(snapshot['nickname'])})" if snapshot.get("nickname") else ""
    card = (
        "<div class='liar-card'><strong>Private role card · "
        + _safe(snapshot["human_name"])
        + nickname_label
        + "</strong><p>"
        + _safe(snapshot["role_message"])
        + "</p><small>Personal rejoin code: "
        + _safe(snapshot["seat_code"])
        + "</small></div>"
    )
    player_messages = snapshot.get("conversation_messages", [
        (speaker, content)
        for speaker, content in snapshot["public_messages"]
        if speaker != "Moderator"
    ])
    def message_html(speaker, content):
        kind = "moderator" if speaker == "Moderator" else "player"
        label = "Moderator · to you" if speaker == "Moderator" else speaker
        return (
            f"<div class='liar-message {kind}'><b>{_safe(label)}</b>"
            f"<div>{_safe(content)}</div></div>"
        )
    recent = list(reversed(player_messages[-RECENT_MESSAGES:]))
    transcript = "<div class='liar-log'><small>Newest first</small>"
    transcript += "".join(message_html(*message) for message in recent)
    if len(player_messages) > RECENT_MESSAGES:
        older = reversed(player_messages[:-RECENT_MESSAGES])
        transcript += (
            "<details><summary>Show earlier messages</summary>"
            + "".join(message_html(*message) for message in older)
            + "</details>"
        )
    if not player_messages:
        transcript += "<div class='liar-empty'>No player messages yet.</div>"
    transcript += "</div>"

    votes = snapshot["votes"]
    if not snapshot["terminal"]:
        vote_html = (
            "<div class='liar-votes'><b>Secret ballot</b><p>"
            + (f"{snapshot['votes_cast']}/{len(snapshot['players'])} ballots submitted. "
               if phase == "vote" else
               "All ballots are sealed; awaiting the final guess. "
               if phase == "liar_guess" else "Voting starts after the clue rounds. ")
            + ("Your vote is locked in." if snapshot.get("my_vote_cast") else
               "Choices and totals stay hidden until the final result.")
            + "</p></div>"
        )
    else:
        counts = {name: 0 for name in snapshot["players"]}
        for target in votes.values():
            if target in counts:
                counts[target] += 1
        rows = "".join(
            "<tr><td>" + _safe(name) + "</td><td>"
            + _safe(votes[name] if votes[name] is not None else "Abstain")
            + "</td></tr>"
            if name in votes
            else "<tr><td>" + _safe(name) + "</td><td>—</td></tr>"
            for name in snapshot["players"]
        )
        totals = " · ".join(f"{_safe(name)}: {counts[name]}" for name in counts)
        accused = snapshot["accused_player"]
        vote_html = (
            "<div class='liar-votes'><b>Final votes</b>"
            "<table><tr><th>Voter</th><th>Choice</th></tr>"
            + rows + "</table><p><b>Totals:</b> " + totals + "</p>"
            + (f"<p><b>Accused:</b> {_safe(accused)}</p>" if accused else "")
            + "</div>"
        )
    if snapshot["rewards"] is not None:
        result = snapshot["result"]
        rewards = " · ".join(
            f"{_safe(name)}: {reward:.1f}"
            for name, reward in snapshot["rewards"].items()
        )
        result_label = "Liar wins" if snapshot["winner"] == "liar" else "Non-liars win"
        guess = result["word_guess"] if result["word_guess"] is not None else "none"
        vote_html = (
            vote_html[:-6]
            + f"<div class='liar-result'><b>{result_label}</b>"
            + f"<p>Actual Liar: {_safe(result['liar_name'])}<br>"
            + f"Topic: {_safe(result['topic'])}<br>"
            + f"Secret word: {_safe(result['secret_word'])}<br>"
            + f"Final guess: {_safe(guess)}</p>"
            + f"<p><b>Rewards:</b> {rewards}</p></div></div>"
        )
        participants = "".join(
            "<tr><td>" + _safe(name) + "</td><td>"
            + _safe(snapshot["participants"][name]["type"]) + "</td><td>"
            + _safe(snapshot["participants"][name]["nickname"] or "—")
            + "</td></tr>"
            for name in snapshot["players"]
        ) if snapshot.get("participants") else ""
        if participants:
            vote_html += (
                "<div class='liar-votes'><b>Players revealed</b>"
                "<table><tr><th>Player</th><th>Type</th><th>Nickname</th></tr>"
                + participants + "</table></div>"
            )
    if snapshot.get("scoreboard"):
        score_rows = "".join(
            "<tr><td>" + _safe(entry["label"]) + "</td><td>"
            + _safe(entry["type"]) + "</td><td>"
            + f"{entry['score']:.1f}" + "</td></tr>"
            for entry in snapshot["scoreboard"]
        )
        vote_html += (
            "<div class='liar-votes'><b>Room cumulative scores</b>"
            "<table><tr><th>Participant</th><th>Type</th><th>Points</th></tr>"
            + score_rows + "</table></div>"
        )

    action_visible = snapshot["human_turn"] and phase in ("clue", "liar_guess")
    vote_visible = snapshot["human_turn"] and phase == "vote"
    action_label = "Your indirect clue" if phase == "clue" else "Your final word guess"
    vote_choices = [
        name for name in snapshot["players"] if name != snapshot["human_name"]
    ]
    action_update = {"visible": action_visible, "label": action_label}
    vote_update = {"visible": vote_visible, "choices": vote_choices}
    if clear_input:
        action_update["value"] = ""
        vote_update["value"] = None
    return (
        status,
        card,
        transcript,
        vote_html,
        gr.update(**action_update),
        gr.update(**vote_update),
        gr.update(
            visible=not snapshot["terminal"],
            interactive=snapshot["human_turn"],
            value={
                "clue": "Submit clue",
                "vote": "Cast vote",
                "liar_guess": "Submit final guess",
            }.get(phase, "Submit"),
        ),
        public_info,
        gr.update(
            visible=snapshot["terminal"],
            interactive=snapshot["terminal"] and not snapshot.get("i_am_ready", False),
            value=("Waiting for others" if snapshot.get("i_am_ready") else "Ready for next game"),
        ),
    )


def build_demo(config_path: str):
    import gradio as gr

    config = ArenaConfig.load(config_path)
    default_reveal = config.environment.get("reveal_topic", True)
    default_rounds = config.environment.get("clue_rounds", 2)
    registry = RoomRegistry(config_path)

    def display(identity, snapshot, error="", clear_input=False):
        return (identity, *render_snapshot(snapshot, gr, clear_input), error)

    def resolve(identity):
        if not identity:
            raise ValueError("Create or join a room first.")
        room = registry.get(identity["room"])
        return room, identity["token"]

    def create_room(human_count, ai_count, reveal_topic, clue_rounds, nickname):
        try:
            room, token = registry.create(human_count, ai_count, reveal_topic, clue_rounds, nickname)
            identity = {"room": room.room_code, "token": token}
            yield display(identity, room.view(token), clear_input=True)
            while room.step_ai_once():
                yield display(identity, room.view(token))
        except ValueError as exc:
            yield display(None, None, _safe(exc))

    def join_room(room_code, seat_code, nickname):
        try:
            room = registry.get(room_code)
            token = room.join(seat_code, nickname)
            identity = {"room": room.room_code, "token": token}
            yield display(identity, room.view(token), clear_input=True)
            while room.step_ai_once():
                yield display(identity, room.view(token))
        except ValueError as exc:
            yield display(None, None, _safe(exc))

    def refresh(identity):
        if not identity:
            return display(None, None)
        try:
            room, token = resolve(identity)
            room.step_ai_once()
            return display(identity, room.view(token))
        except ValueError as exc:
            return display(None, None, _safe(exc))

    def make_move(identity, action, vote):
        try:
            room, token = resolve(identity)
            snapshot = room.submit(token, action, vote)
            yield display(identity, snapshot, clear_input=True)
            while room.step_ai_once():
                yield display(identity, room.view(token))
        except ValueError as exc:
            if identity:
                try:
                    room, token = resolve(identity)
                    yield display(identity, room.view(token), _safe(exc))
                except ValueError:
                    yield display(None, None, _safe(exc))
            else:
                yield display(None, None, _safe(exc))
        except Exception as exc:
            yield display(identity, None, f"Game error: {_safe(exc)}")

    def restart_game(identity):
        try:
            room, token = resolve(identity)
            snapshot = room.ready_for_restart(token)
            yield display(identity, snapshot, clear_input=True)
            while room.step_ai_once():
                yield display(identity, room.view(token))
        except ValueError as exc:
            if identity:
                try:
                    room, token = resolve(identity)
                    yield display(identity, room.view(token), _safe(exc))
                except ValueError:
                    yield display(None, None, _safe(exc))
            else:
                yield display(None, None, _safe(exc))
        except Exception as exc:
            yield display(identity, None, f"Game error: {_safe(exc)}")

    css = """
    .liar-card {background:#eef6ff;border:1px solid #8bb9ed;border-radius:12px;padding:16px;white-space:pre-wrap}
    .liar-info {background:#f1f5f9;border:1px solid #cbd5e1;border-radius:12px;padding:14px;margin-bottom:10px}
    .liar-info p {margin:8px 0}
    .liar-info details p {white-space:pre-wrap}
    .liar-card p {margin-bottom:0}
    .liar-log {padding:8px;background:#f7f8fa;border-radius:12px}
    .liar-message {padding:10px 14px;margin:8px 0;border-radius:10px;background:white;white-space:pre-wrap;border:1px solid #e2e8f0}
    .liar-message b {display:block;margin-bottom:4px}
    .liar-message.moderator {background:#fff7e6;border-color:#f3ce84}
    .liar-log summary {cursor:pointer;padding:8px;font-weight:600}
    .liar-result {padding:12px;margin-top:12px;background:#e7f7ed;border-radius:10px}
    .liar-votes {padding:12px;background:#f7f8fa;border-radius:12px}
    .liar-votes table {width:100%;margin-top:8px;border-collapse:collapse}
    .liar-votes th,.liar-votes td {text-align:left;border-bottom:1px solid #dce1e7;padding:6px}
    .liar-empty {padding:14px;color:#64748b}
    """
    with gr.Blocks(title="Liar Game · Humans and AI", css=css) as demo:
        gr.Markdown("# Liar Game · humans and AI")
        gr.Markdown(
            "The host chooses human and AI counts. Each human opens this page in "
            "their own browser and joins using the room code. Player numbers are "
            "randomized for each game. Keep your personal rejoin code private."
        )
        session_state = gr.State(value=None)
        with gr.Row():
            human_count = gr.Number(value=1, minimum=1, maximum=MAX_PLAYERS, precision=0, label="Humans")
            ai_count = gr.Number(value=4, minimum=0, maximum=MAX_PLAYERS - 1, precision=0, label="AI players")
            clue_rounds = gr.Number(value=default_rounds, minimum=1, maximum=MAX_CLUE_ROUNDS, precision=0, label="Clue rounds (each player speaks once)")
            reveal = gr.Checkbox(value=default_reveal, label="Reveal topic to everyone")
            host_nickname = gr.Textbox(label="Your nickname (optional)", max_lines=1)
            create = gr.Button("Create room", variant="primary")
        with gr.Row():
            room_code = gr.Textbox(label="Room code to join")
            seat_code = gr.Textbox(label="Personal rejoin code (only if reconnecting)")
            join_nickname = gr.Textbox(label="Your nickname (optional; blank keeps previous on rejoin)", max_lines=1)
            join = gr.Button("Join room")
        timer = gr.Timer(2)
        status = gr.Markdown("Choose player counts to create a room, or join an existing room.")
        public_info = gr.HTML("<div class='liar-empty'>Game information will appear here.</div>")
        role_card = gr.HTML("<div class='liar-card'>Your private role will appear here.</div>")
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Player conversation")
                transcript = gr.HTML(
                    "<div class='liar-empty'>The public conversation will appear here.</div>"
                )
            with gr.Column(scale=2):
                gr.Markdown("### Voting and result")
                vote_view = gr.HTML("<div class='liar-empty'>Voting has not started.</div>")
                action_box = gr.Textbox(label="Your indirect clue", lines=2, visible=False)
                vote_choice = gr.Dropdown(label="Vote for another player", visible=False)
                submit = gr.Button("Submit", visible=False)
                restart = gr.Button("Ready for next game", visible=False)
                error = gr.Markdown("")

        outputs = [
            session_state,
            status,
            role_card,
            transcript,
            vote_view,
            action_box,
            vote_choice,
            submit,
            public_info,
            restart,
            error,
        ]
        create.click(
            create_room,
            inputs=[human_count, ai_count, reveal, clue_rounds, host_nickname],
            outputs=outputs,
            concurrency_id="liar_game",
            concurrency_limit=1,
            api_name=False,
        )
        join.click(
            join_room,
            inputs=[room_code, seat_code, join_nickname],
            outputs=outputs,
            concurrency_id="liar_game",
            concurrency_limit=1,
            api_name=False,
        )
        submit.click(
            make_move,
            inputs=[session_state, action_box, vote_choice],
            outputs=outputs,
            concurrency_id="liar_game",
            concurrency_limit=1,
            api_name=False,
        )
        restart.click(
            restart_game,
            inputs=[session_state],
            outputs=outputs,
            concurrency_id="liar_game",
            concurrency_limit=1,
            api_name=False,
        )
        timer.tick(
            refresh,
            inputs=[session_state],
            outputs=outputs,
            queue=False,
            api_name=False,
        )
    return demo.queue(default_concurrency_limit=1, max_size=16)
