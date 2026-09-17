from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional, Set, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

# Standard role names
MAFIA = "mafia"
DOCTOR = "doctor"
POLICE = "police"
VILLAGER = "villager"

DEFAULT_ROLE_DESCRIPTIONS = {
    MAFIA: "Eliminate citizens during the night and blend in during the day to avoid detection.",
    DOCTOR: "Protect one player from being eliminated each night.",
    POLICE: "Investigate one player each night to determine whether they are a member of the Mafia.",
    VILLAGER: "Discuss clues with other players during the day and vote to eliminate suspected Mafia members.",
}


@register_env
class Mafia(Environment):
    """
    Multi-Agent Mafia (Werewolf) Game Environment.

    Supports configurable roles (Mafia, Doctor, Police, Villager),
    asymmetric information visibility, day/night cycles, night actions,
    day discussions, and democratic voting for elimination.
    """

    type_name = "mafia"

    def __init__(
        self,
        player_names: List[str],
        role_counts: Optional[Dict[str, int]] = None,
        role_mapping: Optional[Dict[str, str]] = None,
        role_descriptions: Optional[Dict[str, str]] = None,
        discussion_rounds: int = 1,
        max_days: int = 5,
        reveal_role_on_death: bool = True,
        **kwargs,
    ):
        super().__init__(
            player_names=player_names,
            role_counts=role_counts,
            role_mapping=role_mapping,
            role_descriptions=role_descriptions,
            discussion_rounds=discussion_rounds,
            max_days=max_days,
            reveal_role_on_death=reveal_role_on_death,
            **kwargs,
        )

        self.custom_role_counts = role_counts
        self.custom_role_mapping = role_mapping
        self.role_descriptions = (
            role_descriptions
            if role_descriptions is not None
            else DEFAULT_ROLE_DESCRIPTIONS.copy()
        )
        self.discussion_rounds = max(1, int(discussion_rounds))
        self.max_days = max(1, int(max_days))
        self.reveal_role_on_death = bool(reveal_role_on_death)

        self.message_pool = MessagePool()

        # Game state tracking
        self.player_roles: Dict[str, str] = {}
        self.alive_players: Set[str] = set()
        self.day: int = 1
        self.phase: str = "NIGHT_MAFIA"
        self._current_turn: int = 0

        # Night action temporary storage
        self.night_kills: List[str] = []
        self.night_heal: Optional[str] = None
        self.night_investigation_target: Optional[str] = None

        # Turn sequence queues within current phase
        self._action_queue: List[str] = []
        self._current_actor: Optional[str] = None
        self._discussion_count: int = 0
        self.votes: Dict[str, str] = {}

        self._terminal: bool = False
        self._initialized: bool = False

        self.reset()

    def _assign_roles(self):
        """Assign roles to players based on configuration or smart defaults."""
        num_players = len(self.player_names)
        if self.custom_role_mapping:
            for name in self.player_names:
                role = self.custom_role_mapping.get(name, VILLAGER).lower()
                self.player_roles[name] = role
            return

        roles_list: List[str] = []
        if self.custom_role_counts:
            for role, count in self.custom_role_counts.items():
                roles_list.extend([role.lower()] * int(count))

            # Fill remaining with villagers or truncate
            if len(roles_list) < num_players:
                roles_list.extend([VILLAGER] * (num_players - len(roles_list)))
            else:
                roles_list = roles_list[:num_players]
        else:
            # Smart defaults based on player count
            if num_players <= 3:
                roles_list = [MAFIA, DOCTOR] + [VILLAGER] * (num_players - 2)
            elif num_players == 4:
                roles_list = [MAFIA, DOCTOR, POLICE, VILLAGER]
            elif num_players in (5, 6):
                roles_list = [MAFIA, DOCTOR, POLICE] + [VILLAGER] * (num_players - 3)
            else:
                # 7 or more players: 2 mafias
                roles_list = [MAFIA, MAFIA, DOCTOR, POLICE] + [VILLAGER] * (num_players - 4)

        random.shuffle(roles_list)
        self.player_roles = {
            player: roles_list[i] for i, player in enumerate(self.player_names)
        }

    def _get_players_by_role(self, role: str, alive_only: bool = True) -> List[str]:
        return [
            name
            for name, r in self.player_roles.items()
            if r == role and (not alive_only or name in self.alive_players)
        ]

    def _moderator_speak(self, text: str, visible_to: Union[str, List[str]] = "all"):
        message = Message(
            agent_name="Moderator",
            content=text,
            turn=self._current_turn,
            visible_to=visible_to,
        )
        self.message_pool.append_message(message)

    def reset(self) -> TimeStep:
        self.message_pool.reset()
        self.alive_players = set(self.player_names)
        self.day = 1
        self._current_turn = 0
        self._terminal = False
        self.night_kills = []
        self.night_heal = None
        self.night_investigation_target = None
        self.votes = {}

        self._assign_roles()

        # Build public intro
        role_counts_summary = {}
        for r in self.player_roles.values():
            role_counts_summary[r] = role_counts_summary.get(r, 0) + 1
        summary_str = ", ".join(
            f"{count} {role}(s)" for role, count in role_counts_summary.items()
        )

        intro = (
            f"Welcome to Mafia! There are {len(self.player_names)} players in this game: "
            f"{', '.join(self.player_names)}.\n"
            f"Role distribution: {summary_str}.\n"
            "Rules:\n"
            "- The game alternates between Night and Day.\n"
            "- At Night, the Mafia chooses a player to eliminate. Special roles (Doctor, Police) may perform actions.\n"
            "- During the Day, players discuss clues and vote to eliminate a suspected Mafia member.\n"
            "- Citizens win if all Mafia are eliminated. Mafia wins when Mafia count equals or exceeds Citizens."
        )
        self._moderator_speak(intro, visible_to="all")

        # Send private role assignments to each player
        mafia_players = self._get_players_by_role(MAFIA, alive_only=False)
        for player, role in self.player_roles.items():
            desc = self.role_descriptions.get(role, "")
            role_msg = f"Your secret role is: **{role.upper()}**.\nDescription: {desc}"
            if role == MAFIA:
                fellows = [m for m in mafia_players if m != player]
                if fellows:
                    role_msg += f"\nYour fellow Mafia member(s): {', '.join(fellows)}."
                else:
                    role_msg += "\nYou are the sole Mafia member."
            self._moderator_speak(role_msg, visible_to=[player])

        # Start Night 1
        self._start_night_phase()

        self._initialized = True
        return TimeStep(
            observation=self.get_observation(),
            reward=self.get_zero_rewards(),
            terminal=False,
        )

    def _start_night_phase(self):
        self._moderator_speak(
            f"--- Day {self.day}: Night Falls ---\n"
            "Night has fallen. All players close their eyes and go to sleep.",
            visible_to="all",
        )
        self.night_kills = []
        self.night_heal = None
        self.night_investigation_target = None

        # Queue alive mafia players
        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        if alive_mafia:
            self.phase = "NIGHT_MAFIA"
            self._action_queue = list(alive_mafia)
            self._current_actor = self._action_queue.pop(0)
            valid_targets = [p for p in self.alive_players if p not in alive_mafia]
            if not valid_targets:
                valid_targets = list(self.alive_players)
            prompt = (
                f"{self._current_actor} (Mafia), choose a living citizen to eliminate tonight.\n"
                f"Valid living targets: {', '.join(valid_targets)}.\n"
                "Format: 'I choose to eliminate [Player Name]'."
            )
            self._moderator_speak(prompt, visible_to=alive_mafia)
        else:
            self._advance_night_roles()

    def _advance_night_roles(self):
        """Move from Mafia -> Doctor -> Police -> Day."""
        if self.phase == "NIGHT_MAFIA":
            alive_doctors = self._get_players_by_role(DOCTOR, alive_only=True)
            if alive_doctors:
                self.phase = "NIGHT_DOCTOR"
                self._action_queue = list(alive_doctors)
                self._current_actor = self._action_queue.pop(0)
                prompt = (
                    f"{self._current_actor} (Doctor), choose a living player to protect tonight.\n"
                    f"Living players: {', '.join(sorted(self.alive_players))}.\n"
                    "Format: 'I choose to protect [Player Name]'."
                )
                self._moderator_speak(prompt, visible_to=[self._current_actor])
                return

        if self.phase in ("NIGHT_MAFIA", "NIGHT_DOCTOR"):
            alive_police = self._get_players_by_role(POLICE, alive_only=True)
            if alive_police:
                self.phase = "NIGHT_POLICE"
                self._action_queue = list(alive_police)
                self._current_actor = self._action_queue.pop(0)
                other_living = [p for p in self.alive_players if p != self._current_actor]
                prompt = (
                    f"{self._current_actor} (Police), choose a living player to investigate tonight.\n"
                    f"Candidate players: {', '.join(sorted(other_living))}.\n"
                    "Format: 'I choose to investigate [Player Name]'."
                )
                self._moderator_speak(prompt, visible_to=[self._current_actor])
                return

        # All night actions finished, start day
        self._start_day_phase()

    def _start_day_phase(self):
        # Resolve night actions
        killed_target = self.night_kills[-1] if self.night_kills else None
        healed_target = self.night_heal

        self._moderator_speak(
            f"--- Day {self.day}: Daybreak ---\n"
            "The sun rises. The village awakens to find out what happened during the night.",
            visible_to="all",
        )

        if killed_target and killed_target != healed_target:
            victim = killed_target
            self.alive_players.discard(victim)
            msg = f"Tragedy struck! **{victim}** was eliminated during the night!"
            if self.reveal_role_on_death:
                victim_role = self.player_roles.get(victim, "unknown").upper()
                msg += f" {victim} was a **{victim_role}**."
            self._moderator_speak(msg, visible_to="all")
        elif killed_target and killed_target == healed_target:
            self._moderator_speak(
                f"A miracle occurred! The Doctor successfully protected the victim. Nobody died tonight!",
                visible_to="all",
            )
        else:
            self._moderator_speak(
                "Peaceful night! Nobody was eliminated tonight.", visible_to="all"
            )

        # Check win condition immediately after night casualty
        if self._check_win_condition():
            return

        # Start Day Discussion
        alive_list = sorted(list(self.alive_players))
        self.phase = "DAY_DISCUSSION"
        self._action_queue = []
        for _ in range(self.discussion_rounds):
            self._action_queue.extend(alive_list)

        self._current_actor = self._action_queue.pop(0)
        self._moderator_speak(
            f"Day discussion begins. Living players: {', '.join(alive_list)}.\n"
            f"Discuss suspicions, share information, and defend yourselves.\n"
            f"We begin with {self._current_actor}.",
            visible_to="all",
        )

    def _start_voting_phase(self):
        self.phase = "DAY_VOTING"
        self.votes = {}
        alive_list = sorted(list(self.alive_players))
        self._action_queue = list(alive_list)
        self._current_actor = self._action_queue.pop(0)
        self._moderator_speak(
            f"Discussion has ended. It is time for democratic voting.\n"
            f"Every living player must vote to eliminate one suspect.\n"
            f"Eligible targets: {', '.join(alive_list)}.\n"
            f"Starting vote with {self._current_actor}. Format: 'I vote to eliminate [Player Name]'.",
            visible_to="all",
        )

    def _resolve_voting(self):
        # Tally votes
        tally: Dict[str, int] = {p: 0 for p in self.alive_players}
        for voter, target in self.votes.items():
            if target in tally:
                tally[target] += 1

        tally_breakdown = ", ".join(f"{p}: {count}" for p, count in tally.items())
        self._moderator_speak(
            f"Voting Results: {tally_breakdown}", visible_to="all"
        )

        max_votes = max(tally.values()) if tally else 0
        top_targets = [p for p, count in tally.items() if count == max_votes]

        if max_votes > 0 and len(top_targets) == 1:
            eliminated = top_targets[0]
            self.alive_players.discard(eliminated)
            msg = f"**{eliminated}** received the most votes ({max_votes}) and has been eliminated."
            if self.reveal_role_on_death:
                r = self.player_roles.get(eliminated, "unknown").upper()
                msg += f" {eliminated} was a **{r}**."
            self._moderator_speak(msg, visible_to="all")
        else:
            self._moderator_speak(
                "There was a tie in votes or no votes cast. Nobody is eliminated today.",
                visible_to="all",
            )

        if self._check_win_condition():
            return

        # Advance to next night
        self.day += 1
        if self.day > self.max_days:
            self._moderator_speak(
                f"Maximum days ({self.max_days}) reached. The game ends in a draw!",
                visible_to="all",
            )
            self._terminal = True
            return

        self._start_night_phase()

    def _check_win_condition(self) -> bool:
        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        alive_citizens = [
            p for p in self.alive_players if self.player_roles[p] != MAFIA
        ]

        if len(alive_mafia) == 0:
            self._moderator_speak(
                "All Mafia members have been eliminated! **The Citizens win the game!**",
                visible_to="all",
            )
            self._terminal = True
            return True

        if len(alive_mafia) >= len(alive_citizens):
            mafia_names = ", ".join(self._get_players_by_role(MAFIA, alive_only=False))
            self._moderator_speak(
                f"Mafia equals or outnumbers citizens ({len(alive_mafia)} vs {len(alive_citizens)})! "
                f"**The Mafia ({mafia_names}) wins the game!**",
                visible_to="all",
            )
            self._terminal = True
            return True

        return False

    def _parse_target(self, text: str, candidates: List[str]) -> Optional[str]:
        """Extract target player name from action text using pattern matching."""
        if not text or not candidates:
            return None

        # 1. Regex check for explicit structured format: [eliminate/vote/protect/target] Player X
        regex_pattern = r"(?:vote|eliminate|kill|protect|heal|investigate|target)(?:\s+to|\s+for)?\s*:?\s*(?:\[|\()?([a-zA-Z0-9_\s]+)(?:\]|\))?"
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        for m in matches:
            m_clean = m.strip().lower()
            for cand in candidates:
                if cand.lower() == m_clean or cand.lower().replace(" ", "") == m_clean.replace(" ", ""):
                    return cand

        # 2. Substring matching against candidates
        text_lower = text.lower()
        matched = []
        for cand in candidates:
            cand_variants = [
                cand.lower(),
                cand.lower().replace(" ", ""),
                cand.lower().replace(" ", "_"),
            ]
            if any(v in text_lower for v in cand_variants):
                matched.append(cand)

        if matched:
            # Pick the candidate mentioned latest in the text if multiple
            return max(matched, key=lambda c: text_lower.rfind(c.lower()))

        # 3. Fallback: random candidate from valid list
        return random.choice(candidates) if candidates else None

    def get_next_player(self) -> str:
        """Returns the player whose turn it is to act."""
        if self._current_actor is not None:
            return self._current_actor
        alive = sorted(list(self.alive_players))
        return alive[0] if alive else self.player_names[0]

    def get_observation(self, player_name: Optional[str] = None) -> List[Message]:
        if player_name is None:
            return self.message_pool.get_all_messages()
        return self.message_pool.get_visible_messages(
            player_name, turn=self._current_turn + 1
        )

    def get_rewards(self) -> Dict[str, float]:
        """
        Compute scalar rewards for each player:
        - Winning team: +1.0
        - Losing team: -1.0
        - Draw: 0.0
        """
        rewards = {name: 0.0 for name in self.player_names}
        if not self._terminal:
            return rewards

        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        alive_citizens = [
            p for p in self.alive_players if self.player_roles[p] != MAFIA
        ]

        if len(alive_mafia) == 0:
            # Citizens win
            for name, role in self.player_roles.items():
                rewards[name] = 1.0 if role != MAFIA else -1.0
        elif len(alive_mafia) >= len(alive_citizens):
            # Mafia win
            for name, role in self.player_roles.items():
                rewards[name] = 1.0 if role == MAFIA else -1.0

        return rewards

    def is_terminal(self) -> bool:
        if self._terminal:
            return True
        last_msg = self.message_pool.last_message
        if last_msg and last_msg.content.startswith(SIGNAL_END_OF_CONVERSATION):
            return True
        return False

    def step(self, player_name: str, action: str) -> TimeStep:
        assert (
            player_name == self.get_next_player()
        ), f"Turn error! Expected {self.get_next_player()}, got {player_name}."

        self._current_turn += 1

        if self.phase == "NIGHT_MAFIA":
            # Record Mafia message (visible only to Mafia)
            alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
            msg = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=alive_mafia,
            )
            self.message_pool.append_message(msg)

            valid_targets = [p for p in self.alive_players if p not in alive_mafia]
            if not valid_targets:
                valid_targets = list(self.alive_players)
            target = self._parse_target(action, valid_targets)
            if target:
                self.night_kills.append(target)

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._advance_night_roles()

        elif self.phase == "NIGHT_DOCTOR":
            # Record Doctor message (private to Doctor)
            msg = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name],
            )
            self.message_pool.append_message(msg)

            target = self._parse_target(action, list(self.alive_players))
            self.night_heal = target

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._advance_night_roles()

        elif self.phase == "NIGHT_POLICE":
            # Record Police message (private to Police)
            msg = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to=[player_name],
            )
            self.message_pool.append_message(msg)

            candidates = [p for p in self.alive_players if p != player_name]
            target = self._parse_target(action, candidates)
            if target:
                is_mafia = self.player_roles.get(target) == MAFIA
                verdict = "MAFIA" if is_mafia else "NOT MAFIA"
                self._moderator_speak(
                    f"Investigation report: **{target}** is **{verdict}**.",
                    visible_to=[player_name],
                )

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._advance_night_roles()

        elif self.phase == "DAY_DISCUSSION":
            # Discussion is public to all
            msg = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to="all",
            )
            self.message_pool.append_message(msg)

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._start_voting_phase()

        elif self.phase == "DAY_VOTING":
            # Vote is publicly registered
            msg = Message(
                agent_name=player_name,
                content=action,
                turn=self._current_turn,
                visible_to="all",
            )
            self.message_pool.append_message(msg)

            candidates = list(self.alive_players)
            target = self._parse_target(action, candidates)
            if target:
                self.votes[player_name] = target

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._resolve_voting()

        else:
            raise ValueError(f"Unknown game phase: {self.phase}")

        terminal = self.is_terminal()
        rewards = self.get_rewards()

        return TimeStep(
            observation=self.get_observation(),
            reward=rewards,
            terminal=terminal,
        )
