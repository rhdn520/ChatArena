import random
import re
from typing import Dict, List, Optional, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env
from .liar_game_words import DEFAULT_WORD_SETS


@register_env
class LiarGame(Environment):
    """Fixed-order clue game with one uninformed Liar."""

    type_name = "liar_game"

    def __init__(
        self,
        player_names: List[str],
        word_sets: Optional[Dict[str, List[str]]] = None,
        clue_rounds: int = 2,
        reveal_topic: bool = True,
        random_seed: Optional[int] = None,
        liar_name: Optional[str] = None,
        topic: Optional[str] = None,
        secret_word: Optional[str] = None,
        **kwargs,
    ):
        if len(player_names) < 3:
            raise ValueError("Liar Game requires at least three players")
        configured_sets = {
            name: list(words)
            for name, words in (
                DEFAULT_WORD_SETS if word_sets is None else word_sets
            ).items()
        }
        if not configured_sets or any(not words for words in configured_sets.values()):
            raise ValueError("Every word set must contain at least one word")
        if liar_name is not None and liar_name not in player_names:
            raise ValueError("Configured liar_name must be one of player_names")
        if topic is not None and topic not in configured_sets:
            raise ValueError("Configured topic must be in word_sets")
        if secret_word is not None:
            if topic is None:
                raise ValueError("Configured secret_word requires a configured topic")
            if secret_word not in configured_sets[topic]:
                raise ValueError("Configured secret_word must belong to its topic")
        if int(clue_rounds) < 1:
            raise ValueError("clue_rounds must be at least one")

        super().__init__(
            player_names=player_names,
            word_sets=configured_sets,
            clue_rounds=clue_rounds,
            reveal_topic=reveal_topic,
            random_seed=random_seed,
            liar_name=liar_name,
            topic=topic,
            secret_word=secret_word,
            **kwargs,
        )
        self.word_sets = configured_sets
        self.clue_rounds = int(clue_rounds)
        self.reveal_topic = reveal_topic
        self.random_seed = random_seed
        self._rng = random.Random(random_seed)
        self._configured_liar_name = liar_name
        self._configured_topic = topic
        self._configured_secret_word = secret_word
        self.message_pool = MessagePool()

        self.liar_name = ""
        self.non_liar_names: List[str] = []
        self.topic = ""
        self.secret_word = ""
        self.phase = "clue"
        self.current_round = 1
        self._clue_index = 0
        self._vote_index = 0
        self._current_turn = 0
        self.votes: Dict[str, Optional[str]] = {}
        self.accused_player: Optional[str] = None
        self.word_guess: Optional[str] = None
        self.winner: Optional[str] = None
        self._terminal = False
        self._terminal_rewards = self.get_zero_rewards()
        self._vote_summary = ""
        self.reset()

    def _moderator_speak(
        self, text: str, visible_to: Union[str, List[str]] = "all",
        msg_type: str = "text",
    ) -> None:
        self.message_pool.append_message(
            Message(
                agent_name="Moderator",
                content=text,
                turn=self._current_turn,
                visible_to=visible_to,
                msg_type=msg_type,
            )
        )

    def _announce_clue_turn(self) -> None:
        player = self.player_names[self._clue_index]
        if self._clue_index == 0:
            self._moderator_speak(
                f"Clue round {self.current_round}/{self.clue_rounds}. "
                f"Public speaking order: {', '.join(self.player_names)}."
            )
        task = (
            "You were not given the word. Use only the public clues so far to "
            "give a plausible indirect clue while concealing that uncertainty. "
            "If no clues have been spoken yet, use only the topic if it was revealed. "
            "Do not simply copy another player's sentence."
            if player == self.liar_name else
            "You know the word privately; the other non-liars already know it too. "
            "Give one understated, truthful association through an everyday "
            "situation, reaction, habit, or preference. Leave several words plausible. "
            "Avoid explaining its purpose, recognizable parts, or signature facts. "
            "If earlier clues already narrow the answer, add no further identifying "
            "detail. Keep a real connection to your word rather than an empty "
            "statement that fits anything."
        )
        self._moderator_speak(
            f"Round {self.current_round}/{self.clue_rounds}: {player}, "
            + task + " Your clue is PUBLIC. Output one short clue only; "
            "do not name the secret word or declare your role.",
            visible_to=[player], msg_type="instruction",
        )

    def reset(self) -> TimeStep:
        """Initialize an independent episode and clear the previous history."""
        self.message_pool.reset()
        self.topic = self._configured_topic or self._rng.choice(list(self.word_sets))
        self.secret_word = self._configured_secret_word or self._rng.choice(
            self.word_sets[self.topic]
        )
        self.liar_name = self._configured_liar_name or self._rng.choice(
            self.player_names
        )
        self.non_liar_names = [
            name for name in self.player_names if name != self.liar_name
        ]
        self.phase = "clue"
        self.current_round = 1
        self._clue_index = 0
        self._vote_index = 0
        self._current_turn = 0
        self.votes = {}
        self.accused_player = None
        self.word_guess = None
        self.winner = None
        self._terminal = False
        self._terminal_rewards = self.get_zero_rewards()
        self._vote_summary = ""

        self._moderator_speak(
            f"Liar Game begins with {len(self.player_names)} players and exactly "
            f"one Liar. There will be {self.clue_rounds} clue round(s), followed "
            "by one private vote from each player for another player. "
            "If the group accuses a non-liar, the Liar wins. If it accuses the "
            "Liar, the Liar gets one final word guess and still wins if correct; "
            "otherwise the non-liars win. Non-liars must both identify the Liar "
            "and keep the word difficult to guess. All clues are public. "
            "Malformed or self-votes abstain; ties accuse the earliest tied "
            "player in the configured player order."
        )
        if self.reveal_topic:
            self._moderator_speak(f"Topic: {self.topic}.")
        self._moderator_speak(
            "You are a non-liar. "
            f"The secret word is {self.secret_word}. "
            "Keep it secret from the Liar. The other non-liars already know it, "
            "so there is no need to teach them its definition or identifying features. "
            "Use a short, understated association with an everyday situation, "
            "reaction, habit, or preference. Do not explicitly say the secret word.",
            visible_to=self.non_liar_names,
        )
        self._moderator_speak(
            "You are the Liar. You do not know the secret word. Infer it from "
            "others' clues while blending in. Give a plausible clue on your turn.",
            visible_to=[self.liar_name],
        )
        self._announce_clue_turn()
        return TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

    def print(self) -> None:
        self.message_pool.print()

    def get_next_player(self) -> str:
        if self.phase == "clue":
            return self.player_names[self._clue_index]
        if self.phase == "vote":
            return self.player_names[self._vote_index]
        if self.phase == "liar_guess":
            return self.liar_name
        raise RuntimeError("The Liar Game episode is already terminal")

    def get_observation(self, player_name: Optional[str] = None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        return self.message_pool.get_visible_messages(
            player_name, turn=self._current_turn + 1
        )

    def get_agent_context(self, player_name: str) -> str:
        """Refresh identity and phase without revealing the word to the Liar."""
        if player_name not in self.player_names:
            raise ValueError("Unknown player")
        is_liar = player_name == self.liar_name
        lines = [
            "[Private engine state; do not copy this block into your answer]",
            f"You are {player_name}. Your role is {'Liar' if is_liar else 'non-liar'}.",
            f"Current phase: {self.phase}. Clue round: {min(self.current_round, self.clue_rounds)}/{self.clue_rounds}.",
            f"Public speaking order in every round: {', '.join(self.player_names)}. "
            "Each speaker has heard the earlier clues in that order, including previous rounds.",
        ]
        if self.reveal_topic:
            lines.append(f"Public topic: {self.topic}.")
        if is_liar:
            lines.append("You were not given the secret word. Infer it only from public clues; blend in to avoid being accused. If accused, you still win by guessing the word correctly.")
        else:
            lines.append(f"Your private secret word: {self.secret_word}. You win only if the group accuses the Liar AND the Liar fails the final word guess. Identify the Liar without giving away the word.")
        lines.append("History marked [Public] is known to everyone. History marked [Private] is available to you but is not public knowledge; it does not reveal who else received it.")
        if self.phase == "clue":
            lines.append(f"Your position this round: {self._clue_index + 1}/{len(self.player_names)}.")
            lines.append("Your next answer is one short PUBLIC indirect clue, not a role declaration, word guess, or explanation. Do not explicitly say the secret word or copy another clue verbatim.")
            if is_liar:
                lines.append("Respond as an informed participant would, using only public clues already spoken. Do not claim that the engine gave you the word. Add a plausible association rather than repeating the last speaker.")
            else:
                lines.append("Speak to people who already know the word: give an understated association with an everyday situation, reaction, habit, or preference. Keep it truthful and word-related, but leave several possible words open. Do not describe its characteristic function, parts, materials, geography, characters, or memorable quotes. Earlier clues may already reveal too much; do not complete their identifying pattern. Do not confirm someone else's guessed word or copy the style-example sentences.")
        elif self.phase == "vote":
            candidates = [p for p in self.player_names if p != player_name]
            lines += ["Your next answer is a PRIVATE vote. Other players' private ballots are unknown.",
                      f"Eligible targets: {', '.join(candidates)}.",
                      "Output only 'I vote for ' followed by one eligible name. No brackets, channel labels or explanation. Malformed or self-votes abstain."]
            if is_liar:
                lines.append("Choose another player to divert the accusation away from you; you already know that you are the Liar.")
            else:
                lines.append("Compare each player's clues with the public information available BEFORE each of their turns. An informed player is deliberately indirect: vagueness alone is not evidence of being the Liar. A precise, correct clue is not proof of private knowledge either: the speaker may have inferred the word from earlier clues. Consider contradictions and how a player's clues develop as information accumulates, not specificity or writing style alone. Choose one suspect using the whole sequence; the evidence can remain uncertain.")
        elif self.phase == "liar_guess":
            lines.append("Your next answer is your one final PUBLIC guess. Output only the secret word you infer, matching the topic and all clues, not just one part or feature.")
        return "\n".join(lines)

    def check_action(self, action: str, player_name: str) -> bool:
        if not isinstance(action, str):
            return False
        if self.phase != "clue":
            # Malformed ballots are legal submissions but resolve as abstentions.
            return True
        word_pattern = r"(?<!\w)" + r"[\W_]+".join(
            re.escape(part) for part in self.secret_word.split()
        ) + r"(?!\w)"
        return bool(action.strip()) and re.search(word_pattern, action, re.I) is None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _parse_vote(self, text: str, voter: str) -> Optional[str]:
        text = text.strip().strip("`\"'").strip()
        for candidate in self.player_names:
            name = re.escape(candidate)
            if candidate != voter and re.fullmatch(
                r"\s*I vote for\s+(?:" + name + r"|\[" + name + r"\])\s*[.!]?\s*", text, re.I
            ):
                return candidate
        return None

    def _is_correct_word_guess(self, text: str) -> bool:
        normalized_text = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        normalized_word = re.sub(
            r"[^a-z0-9]+", " ", self.secret_word.lower()
        ).strip()
        if normalized_text == normalized_word:
            return True
        if not normalized_text.endswith(" " + normalized_word):
            return False
        prefix = normalized_text[: -(len(normalized_word) + 1)]
        return bool(
            re.fullmatch(
                r"(?:i (?:think|guess|believe)(?: (?:the )?(?:secret )?word is)?"
                r"|(?:my )?(?:final )?guess(?: is)?"
                r"|(?:the )?(?:secret )?word is)",
                prefix,
            )
        )

    def _start_voting(self) -> None:
        self.phase = "vote"
        self._vote_index = 0
        self.votes = {}
        self._moderator_speak("Clues are complete. All players vote privately now.")
        for player in self.player_names:
            self._moderator_speak(
                f"{player}, vote privately for one other player as the Liar "
                "by writing 'I vote for ' followed by their exact name. "
                "Do not add brackets, a channel label or explanation. Malformed or self-votes "
                "count as abstentions.",
                visible_to=[player], msg_type="instruction",
            )

    def _resolve_votes(self) -> None:
        counts = {name: 0 for name in self.player_names}
        for target in self.votes.values():
            if target is not None:
                counts[target] += 1
        highest_count = max(counts.values())
        tied = [name for name in self.player_names if counts[name] == highest_count]
        self.accused_player = tied[0]

        vote_summary = ", ".join(
            f"{voter} -> {target if target is not None else 'abstain'}"
            for voter, target in self.votes.items()
        )
        count_summary = ", ".join(
            f"{name}: {counts[name]}" for name in self.player_names
        )
        tie_note = (
            " A tie was resolved by choosing the earliest tied player in the "
            "configured player order."
            if len(tied) > 1
            else ""
        )
        self._vote_summary = (
            f"Votes: {vote_summary}. Totals: {count_summary}. "
            f"The accused player is {self.accused_player}.{tie_note}"
        )
        if self.accused_player != self.liar_name:
            self._finish(
                winner="liar",
                reason="The group accused the wrong player, so the Liar wins.",
            )
        else:
            self.phase = "liar_guess"
            self._moderator_speak(
                f"The accusation is correct: {self.liar_name} is the Liar."
            )
            self._moderator_speak(
                f"{self.liar_name}, make one final guess of the secret word.",
                visible_to=[self.liar_name], msg_type="instruction",
            )

    def _finish(self, winner: str, reason: str) -> None:
        self.winner = winner
        self.phase = "terminal"
        self._terminal = True
        self._terminal_rewards = {
            name: float(
                (winner == "liar" and name == self.liar_name)
                or (winner == "non_liars" and name != self.liar_name)
            )
            for name in self.player_names
        }
        reward_summary = ", ".join(
            f"{name}: {reward:.1f}"
            for name, reward in self._terminal_rewards.items()
        )
        winner_label = "Liar" if winner == "liar" else "Non-liars"
        topic_reveal = "on" if self.reveal_topic else "off"
        self._moderator_speak(
            f"{self._vote_summary} {reason} Topic reveal: {topic_reveal}. Topic: {self.topic}. "
            f"Secret word: {self.secret_word}. Actual Liar: {self.liar_name}. "
            f"Final guess: {self.word_guess if self.word_guess is not None else 'none'}. "
            f"Winner: {winner_label}. Rewards: {reward_summary}"
        )

    def get_rewards(self) -> Dict[str, float]:
        if not self._terminal:
            return self.get_zero_rewards()
        return dict(self._terminal_rewards)

    def is_terminal(self) -> bool:
        if self._terminal:
            return True
        last_message = self.message_pool.last_message
        return bool(
            last_message
            and last_message.content.startswith(SIGNAL_END_OF_CONVERSATION)
        )

    def step(self, player_name: str, action: str) -> TimeStep:
        if self._terminal:
            raise RuntimeError("Cannot step a terminal Liar Game episode")
        expected_player = self.get_next_player()
        assert player_name == expected_player, (
            f"Wrong player: expected {expected_player}, got {player_name}"
        )
        if not self.check_action(action, player_name):
            # Do not publish the rejected clue or reveal the forbidden word.
            raise ValueError("Invalid clue: use a nonempty indirect clue without naming the secret word")
        self._current_turn += 1
        self.message_pool.append_message(
            Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name] if self.phase == "vote" else "all",
                msg_type="vote" if self.phase == "vote" else "text",
            )
        )

        if self.phase == "clue":
            self._clue_index += 1
            if self._clue_index == len(self.player_names):
                self._clue_index = 0
                self.current_round += 1
            if self.current_round > self.clue_rounds:
                self._start_voting()
            else:
                self._announce_clue_turn()
        elif self.phase == "vote":
            self.votes[player_name] = self._parse_vote(action, player_name)
            self._vote_index += 1
            if self._vote_index == len(self.player_names):
                self._resolve_votes()
        elif self.phase == "liar_guess":
            self.word_guess = action
            if self._is_correct_word_guess(action):
                self._finish(
                    winner="liar",
                    reason="The Liar guessed the secret word correctly.",
                )
            else:
                self._finish(
                    winner="non_liars",
                    reason="The Liar guessed the secret word incorrectly.",
                )
        else:
            raise ValueError(f"Unknown Liar Game phase: {self.phase}")

        return TimeStep(
            observation=self.get_observation(),
            reward=self.get_rewards(),
            terminal=self.is_terminal(),
        )
