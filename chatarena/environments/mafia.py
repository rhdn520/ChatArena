from __future__ import annotations

import math
import random
import re
import uuid
import warnings
from typing import Dict, List, Optional, Set, Union

from ..agent import SIGNAL_END_OF_CONVERSATION
from ..message import Message, MessagePool
from .base import Environment, TimeStep, register_env

# Standard role names
MAFIA = "mafia"
DOCTOR = "doctor"
POLICE = "police"
VILLAGER = "villager"

ROLE_NAMES = {MAFIA: "마피아", DOCTOR: "의사", POLICE: "경찰", VILLAGER: "시민"}

DEFAULT_ROLE_DESCRIPTIONS = {
    MAFIA: "밤에는 시민을 제거하고, 낮에는 정체를 숨기며 시민 사이에 섞여 행동하세요.",
    DOCTOR: "매일 밤 한 명을 선택해 마피아의 공격으로부터 보호하세요.",
    POLICE: "밤의 경찰 차례에 한 명을 조사해 마피아인지 확인하세요. 낮에는 새 조사를 할 수 없으며, 지난밤 결과를 공개할지는 스스로 결정하세요.",
    VILLAGER: "낮에 다른 참가자와 단서를 토론하고, 마피아로 의심되는 사람에게 투표하세요.",
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
        discussion_rounds: Optional[int] = None,
        max_days: int = 5,
        reveal_role_on_death: bool = True,
        max_discussion_messages: int = 24,
        max_intent_rounds: int = 48,
        discussion_seconds: float = 180,
        silence_seconds: float = 10,
        intent_weights=None,
        repeat_speaker_factor: float = 0.5,
        seed: Optional[int] = None,
        discussion_moderator=None,
        mafia_discussion_messages: int = 8,
        mafia_intent_rounds: int = 16,
        mafia_discussion_seconds: float = 60,
        **kwargs,
    ):
        super().__init__(
            player_names=player_names,
            role_counts=role_counts,
            role_mapping=role_mapping,
            role_descriptions=role_descriptions,
            max_discussion_messages=max_discussion_messages,
            max_intent_rounds=max_intent_rounds,
            discussion_seconds=discussion_seconds,
            silence_seconds=silence_seconds,
            intent_weights=intent_weights,
            repeat_speaker_factor=repeat_speaker_factor,
            seed=seed,
            discussion_moderator=discussion_moderator,
            mafia_discussion_messages=mafia_discussion_messages,
            mafia_intent_rounds=mafia_intent_rounds,
            mafia_discussion_seconds=mafia_discussion_seconds,
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
        if discussion_rounds is not None:
            warnings.warn("discussion_rounds is ignored; Mafia uses intent-based discussion",
                          FutureWarning, stacklevel=2)
        self.max_discussion_messages = int(max_discussion_messages)
        self.max_intent_rounds = int(max_intent_rounds)
        self.discussion_seconds = float(discussion_seconds)
        self.silence_seconds = float(silence_seconds)
        self.intent_weights = tuple(intent_weights if intent_weights is not None else (0, 1, 4, 9))
        self.repeat_speaker_factor = float(repeat_speaker_factor)
        if (self.max_discussion_messages < 1 or self.max_intent_rounds < 1
                or not math.isfinite(self.discussion_seconds) or self.discussion_seconds <= 0
                or not math.isfinite(self.silence_seconds) or self.silence_seconds < 0
                or len(self.intent_weights) != 4 or self.intent_weights[0] != 0
                or any(not math.isfinite(w) or w <= 0 for w in self.intent_weights[1:])
                or not 0 < self.repeat_speaker_factor <= 1):
            raise ValueError("Invalid discussion limits or weights")
        self.seed = seed
        self.rng = random.Random(seed)
        self.presentation_rng = random.Random(seed)
        self.discussion_moderator = discussion_moderator
        self.mafia_discussion_messages = int(mafia_discussion_messages)
        self.mafia_intent_rounds = int(mafia_intent_rounds)
        self.mafia_discussion_seconds = float(mafia_discussion_seconds)
        if (self.mafia_discussion_messages < 1 or self.mafia_intent_rounds < 1
                or not math.isfinite(self.mafia_discussion_seconds) or self.mafia_discussion_seconds <= 0):
            raise ValueError("Invalid Mafia discussion limits")
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

        self.rng.shuffle(roles_list)
        self.player_roles = {
            player: roles_list[i] for i, player in enumerate(self.player_names)
        }

    def randomized_names(self, names):
        """Randomize prompt order independently of roles and gameplay draws."""
        ordered = sorted(names)
        self.presentation_rng.shuffle(ordered)
        return ordered

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
        self.session_id = uuid.uuid4().hex
        self.version = 0
        self.discussion_messages = 0
        self.last_speaker = None
        self.discussion_end_reason = None
        self.discussion_endings = []
        self.termination_reason = None
        self.phase = "NIGHT_MAFIA"
        self._current_actor = None
        self._action_queue = []
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
            f"{ROLE_NAMES.get(role, role)} {count}명" for role, count in role_counts_summary.items()
        )

        intro = (
            f"마피아 게임에 오신 것을 환영합니다! 참가자는 총 {len(self.player_names)}명입니다: "
            f"{', '.join(self.randomized_names(self.player_names))}.\n"
            f"역할 구성: {summary_str}.\n"
            "게임 규칙:\n"
            "- 밤과 낮이 번갈아 진행됩니다.\n"
            "- 밤에는 마피아가 제거할 대상을 선택하고, 의사는 보호, 경찰은 조사를 수행합니다.\n"
            "- 낮에는 단서를 토론한 뒤 마피아로 의심되는 사람에게 투표해 탈락시킵니다.\n"
            "- 마피아가 모두 탈락하면 시민 팀이 승리합니다. 생존 마피아 수가 나머지 생존자 수 이상이면 마피아 팀이 승리합니다."
        )
        self._moderator_speak(intro, visible_to="all")

        # Send private role assignments to each player
        mafia_players = self._get_players_by_role(MAFIA, alive_only=False)
        for player, role in self.player_roles.items():
            desc = self.role_descriptions.get(role, "")
            role_msg = f"당신의 비밀 역할은 **{ROLE_NAMES.get(role, role)}**입니다.\n역할 설명: {desc}"
            if role == MAFIA:
                fellows = [m for m in mafia_players if m != player]
                if fellows:
                    role_msg += f"\n동료 마피아: {', '.join(fellows)}."
                else:
                    role_msg += "\n당신은 유일한 마피아입니다."
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
            f"--- {self.day}일차: 밤 ---\n"
            "밤이 되었습니다. 모두 눈을 감아주세요.",
            visible_to="all",
        )
        self.night_kills = []
        self.night_heal = None
        self.night_investigation_target = None

        # Queue alive mafia players
        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        if len(alive_mafia) > 1:
            self.phase = "NIGHT_MAFIA_DISCUSSION"
            self._current_actor = None
            self._action_queue = []
            self.discussion_messages = 0
            self.last_speaker = None
            self.discussion_end_reason = None
            self._moderator_speak(
                "마피아 비공개 의논을 시작합니다. 동료와 오늘 밤 공격할 대상을 상의하세요. "
                "이 대화는 생존 마피아에게만 보입니다. 의논 후 각자 한 표씩 투표하며, "
                "최다 득표 대상을 공격합니다. 동률이면 공동 최다 득표자 중 무작위로 결정합니다.",
                visible_to=alive_mafia,
            )
        else:
            self._start_mafia_selection()

    def _start_mafia_selection(self):
        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        if alive_mafia:
            self.phase = "NIGHT_MAFIA"
            self._action_queue = list(alive_mafia)
            self._current_actor = self._action_queue.pop(0)
            valid_targets = [p for p in self.alive_players if p not in alive_mafia]
            if not valid_targets:
                valid_targets = list(self.alive_players)
            prompt = (
                "공격 대상 투표를 시작합니다. 생존 마피아가 순서대로 한 표씩 투표합니다.\n"
                f"{self._current_actor}(마피아), 오늘 밤 제거할 생존 시민을 선택하세요.\n"
                f"선택 가능한 생존자: {', '.join(self.randomized_names(valid_targets))}.\n"
                "한국어로 응답하세요. 형식: '[대상 이름]을 제거하겠습니다.' 대상 이름은 표시된 그대로 쓰세요."
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
                    f"{self._current_actor}(의사), 오늘 밤 보호할 생존자를 선택하세요.\n"
                    f"생존자: {', '.join(self.randomized_names(self.alive_players))}.\n"
                    "한국어로 응답하세요. 형식: '[대상 이름]을 보호하겠습니다.' 대상 이름은 표시된 그대로 쓰세요."
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
                    f"{self._current_actor}(경찰), 오늘 밤 조사할 생존자를 선택하세요.\n"
                    f"조사 가능한 생존자: {', '.join(self.randomized_names(other_living))}.\n"
                    "한국어로 응답하세요. 형식: '[대상 이름]을 조사하겠습니다.' 대상 이름은 표시된 그대로 쓰세요."
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
            f"--- {self.day}일차: 아침 ---\n"
            "아침이 되었습니다. 지난밤의 결과를 발표합니다.",
            visible_to="all",
        )

        if killed_target and killed_target != healed_target:
            victim = killed_target
            self.alive_players.discard(victim)
            msg = f"지난밤 **{victim}**님이 사망했습니다."
            if self.reveal_role_on_death:
                victim_role = ROLE_NAMES.get(self.player_roles.get(victim), "알 수 없음")
                msg += f" {victim}님의 역할은 **{victim_role}**였습니다."
            self._moderator_speak(msg, visible_to="all")
        elif killed_target and killed_target == healed_target:
            self._moderator_speak(
                f"의사가 공격 대상을 보호했습니다. 지난밤 사망자는 없습니다!",
                visible_to="all",
            )
        else:
            self._moderator_speak(
                "평화로운 밤이었습니다. 지난밤 사망자는 없습니다.", visible_to="all"
            )

        # Check win condition immediately after night casualty
        if self._check_win_condition():
            return

        # Start Day Discussion
        alive_list = self.randomized_names(self.alive_players)
        self.phase = "DAY_DISCUSSION"
        self._action_queue = []
        self._current_actor = None
        self.discussion_messages = 0
        self.last_speaker = None
        self.discussion_end_reason = None
        self._moderator_speak(
            f"낮 토론을 시작합니다. 생존자: {', '.join(alive_list)}.\n"
            "낮에는 조사·보호·야간 공격을 할 수 없습니다. 플레이어의 주장은 공식 조사 결과가 아닙니다.\n"
            "의견이 있으면 한국어로 발언하세요. 말하지 않고 듣기만 해도 됩니다.",
        )

    def timestep(self) -> TimeStep:
        return TimeStep(list(self.get_observation()), self.get_rewards(), self.is_terminal())

    @property
    def is_discussion(self):
        return self.phase in ("DAY_DISCUSSION", "NIGHT_MAFIA_DISCUSSION")

    @property
    def discussion_participants(self):
        if self.phase == "NIGHT_MAFIA_DISCUSSION":
            return set(self._get_players_by_role(MAFIA))
        return set(self.alive_players) if self.phase == "DAY_DISCUSSION" else set()

    def discussion_speak(self, player_name: str, text: str) -> TimeStep:
        if (self.is_terminal() or not self.is_discussion
                or player_name not in self.discussion_participants or not text.strip()):
            raise ValueError("Only living players can speak during discussion")
        self._current_turn += 1
        self.version += 1
        self.discussion_messages += 1
        self.last_speaker = player_name
        visible_to = (sorted(self.discussion_participants)
                      if self.phase == "NIGHT_MAFIA_DISCUSSION" else "all")
        self.message_pool.append_message(Message(player_name, text.strip(), self._current_turn,
                                                 visible_to=visible_to))
        return self.timestep()

    def discussion_announce(self, text: str):
        if self.phase != "DAY_DISCUSSION" or self.is_terminal():
            raise ValueError("Discussion is not active")
        self._current_turn += 1
        self.version += 1
        self._moderator_speak(text)

    def end_discussion(self, reason: str):
        if not self.is_discussion or self.is_terminal():
            raise ValueError("Discussion is not active")
        self.discussion_end_reason = reason
        self.discussion_endings.append({"day": self.day, "phase": self.phase, "reason": reason})
        self.version += 1
        if self.phase == "NIGHT_MAFIA_DISCUSSION":
            self._start_mafia_selection()
        else:
            self._start_voting_phase()

    def print(self):
        self.message_pool.print()

    def _start_voting_phase(self):
        self.phase = "DAY_VOTING"
        self.votes = {}
        alive_list = self.randomized_names(self.alive_players)
        self._action_queue = list(alive_list)
        self._current_actor = self._action_queue.pop(0)
        self._moderator_speak(
            f"토론이 끝났습니다. 투표를 시작합니다.\n"
            f"모든 생존자는 탈락시킬 용의자 한 명에게 투표해야 합니다.\n"
            f"투표 가능한 대상: {', '.join(alive_list)}.\n"
            f"{self._current_actor}부터 투표합니다. 한국어로 응답하세요. 형식: '[대상 이름]에게 투표합니다.' 대상 이름은 표시된 그대로 쓰세요.",
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
            f"투표 결과: {tally_breakdown}", visible_to="all"
        )

        max_votes = max(tally.values()) if tally else 0
        top_targets = [p for p, count in tally.items() if count == max_votes]

        if max_votes > 0 and len(top_targets) == 1:
            eliminated = top_targets[0]
            self.alive_players.discard(eliminated)
            msg = f"**{eliminated}**님이 최다 득표({max_votes}표)로 탈락했습니다."
            if self.reveal_role_on_death:
                r = ROLE_NAMES.get(self.player_roles.get(eliminated), "알 수 없음")
                msg += f" {eliminated}님의 역할은 **{r}**였습니다."
            self._moderator_speak(msg, visible_to="all")
        else:
            self._moderator_speak(
                "동률이거나 유효한 투표가 없어 오늘은 아무도 탈락하지 않습니다.",
                visible_to="all",
            )

        if self._check_win_condition():
            return

        # Advance to next night
        self.day += 1
        if self.day > self.max_days:
            self._moderator_speak(
                f"최대 진행 일수({self.max_days}일)에 도달했습니다. 무승부로 게임을 종료합니다!",
                visible_to="all",
            )
            self._terminal = True
            self.termination_reason = "max_days"
            return

        self._start_night_phase()

    def _check_win_condition(self) -> bool:
        alive_mafia = self._get_players_by_role(MAFIA, alive_only=True)
        alive_citizens = [
            p for p in self.alive_players if self.player_roles[p] != MAFIA
        ]

        if len(alive_mafia) == 0:
            self._moderator_speak(
                "마피아가 모두 탈락했습니다! **시민 팀이 승리했습니다!**",
                visible_to="all",
            )
            self._terminal = True
            self.termination_reason = "citizens_win"
            return True

        if len(alive_mafia) >= len(alive_citizens):
            mafia_names = ", ".join(self._get_players_by_role(MAFIA, alive_only=False))
            self._moderator_speak(
                f"생존 마피아({len(alive_mafia)}명)가 나머지 생존자({len(alive_citizens)}명) 이상입니다! "
                f"**마피아 팀({mafia_names})이 승리했습니다!**",
                visible_to="all",
            )
            self._terminal = True
            self.termination_reason = "mafia_win"
            return True

        return False

    def _parse_target(self, text: str, candidates: List[str]) -> Optional[str]:
        """Match whole names/aliases and use the last actual mention's position.

        A numeric suffix cannot be a prefix of another player (1 vs 10).
        Korean particles may directly follow a name. Never invent a target
        when the model did not name an eligible player.
        """
        matches = []
        for candidate in candidates:
            parts = re.split(r"[\s_]+", candidate.strip())
            alias = r"[\s_]*".join(re.escape(part) for part in parts)
            pattern = r"(?<!\w)" + alias + r"(?![A-Za-z0-9_])"
            for match in re.finditer(pattern, text or "", re.IGNORECASE):
                matches.append((match.start(), len(match.group()), candidate))
        return max(matches)[2] if matches else None

    def get_next_player(self) -> str:
        """Returns the player whose turn it is to act."""
        if self.is_discussion:
            raise RuntimeError("Discussion speakers are selected by MafiaDiscussionController")
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
        if self.is_terminal():
            raise ValueError("Game has ended")
        if self.is_discussion:
            return self.discussion_speak(player_name, action)
        assert (
            player_name == self.get_next_player()
        ), f"Turn error! Expected {self.get_next_player()}, got {player_name}."

        candidates = list(self.alive_players)
        if self.phase == "NIGHT_MAFIA":
            candidates = [p for p in candidates if self.player_roles[p] != MAFIA]
        elif self.phase == "NIGHT_POLICE":
            candidates = [p for p in candidates if p != player_name]
        if self._parse_target(action, candidates) is None:
            raise ValueError("선택 가능한 대상 이름을 정확히 포함해주세요. 임의로 대상을 선택하지 않습니다.")

        self.version += 1
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
                if self.night_kills:
                    counts = {name: self.night_kills.count(name) for name in set(self.night_kills)}
                    top = max(counts.values())
                    tied = sorted(name for name, count in counts.items() if count == top)
                    chosen = self.rng.choice(tied)
                    self.night_kills = [chosen]
                    self._moderator_speak(
                        f"공격 대상 투표 결과: " + ", ".join(f"{n}: {counts[n]}표" for n in sorted(counts))
                        + (". 동률이므로 무작위로 결정했습니다." if len(tied) > 1 else ".")
                        + f" 최종 공격 대상은 {chosen}입니다.", visible_to=alive_mafia)
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
                verdict = "마피아" if is_mafia else "마피아가 아님"
                self._moderator_speak(
                    f"조사 결과: **{target}** — **{verdict}**.",
                    visible_to=[player_name],
                )

            if self._action_queue:
                self._current_actor = self._action_queue.pop(0)
            else:
                self._current_actor = None
                self._advance_night_roles()

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
