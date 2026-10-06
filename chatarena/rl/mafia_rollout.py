from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Union

from ..environments.mafia import MAFIA, Mafia
from ..message import Message
from ..mafia_discussion import MafiaDiscussionController, DiscussionError


@dataclasses.dataclass
class MafiaTurnRecord:
    turn: int
    phase: str
    prompt: str
    response: str
    action_valid: bool = True
    kind: str = "game_action"
    version: int = 0
    session_id: str = ""
    score: Optional[int] = None
    selected: bool = False
    trainable: bool = True
    stale: bool = False
    retry: bool = False
    error: Optional[str] = None


@dataclasses.dataclass
class MafiaTrajectory:
    player_name: str
    role: str
    turns: List[MafiaTurnRecord] = dataclasses.field(default_factory=list)
    raw_reward: float = 0.0
    shaped_reward: float = 0.0
    won: bool = False


@dataclasses.dataclass
class MafiaEpisodeResult:
    trajectories: Dict[str, MafiaTrajectory]
    winner: str  # "mafia", "citizens", "draw", or "truncated"
    rewards: Dict[str, float]
    all_messages: List[Message]
    day: int
    truncated: bool = False
    termination_reason: str = ""
    diagnostics: Dict = dataclasses.field(default_factory=dict)
    discussion_endings: List[Dict] = dataclasses.field(default_factory=list)

    def summary(self) -> str:
        """Returns a clean, high-level summary of the game outcome."""
        lines = [
            "================ MAFIA EPISODE RESULT ================",
            f"Winner: {self.winner.upper()} | Day: {self.day} | Reason: {self.termination_reason}",
            "------------------------------------------------------",
            "Players & Outcomes:",
        ]
        for name, traj in sorted(self.trajectories.items()):
            status = "INCOMPLETE" if self.truncated else ("DRAW" if self.winner == "draw" else ("WON" if traj.won else "LOST"))
            lines.append(
                f"  - {name} [{traj.role.upper()}]: {status} | "
                f"Reward: {traj.raw_reward:+.1f} (Shaped: {traj.shaped_reward:+.2f}) | "
                f"{len(traj.turns)} actions taken"
            )
        lines.append("======================================================")
        return "\n".join(lines)

    def render(self, include_dialogue: bool = True, include_prompts: bool = False) -> str:
        """
        Returns a beautifully formatted, human-readable play-by-play transcript of the game.
        """
        lines = [self.summary()]

        if include_dialogue:
            lines.append("\n================ GAME TRANSCRIPT ================")
            for msg in self.all_messages:
                sender = msg.agent_name
                receiver = f" (-> {msg.visible_to})" if msg.visible_to != "all" else ""
                lines.append(f"[{sender}{receiver}]")
                for line in msg.content.strip().split("\n"):
                    lines.append(f"  {line}")
                lines.append("")
            lines.append("=================================================")

        if include_prompts:
            lines.append("\n================ TRAJECTORY DETAILS ================")
            for name, traj in self.trajectories.items():
                if traj.turns:
                    lines.append(f"\n--- {name} ({traj.role.upper()}) ---")
                    for t in traj.turns:
                        lines.append(f"Turn {t.turn} [{t.phase}/{t.kind}] "
                                     f"score={t.score} selected={t.selected} trainable={t.trainable}:")
                        lines.append(f"  Prompt: {t.prompt}")
                        lines.append(f"  Response: {t.response}")
            lines.append("====================================================")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(include_dialogue=True, include_prompts=False)

    def save_log(self, filepath: str):
        """Save human-readable game transcript to a file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.render(include_dialogue=True, include_prompts=True))



def format_observation_as_prompt(
    observation: List[Message],
    system_prompt: Optional[str] = None,
) -> str:
    """Formats a list of ChatArena messages into a prompt string for LLMs."""
    formatted_lines = []
    if system_prompt:
        formatted_lines.append(f"System: {system_prompt}\n")

    for msg in observation:
        sender = msg.agent_name
        content = msg.content.strip()
        formatted_lines.append(f"{sender}: {content}")

    formatted_lines.append("You: ")
    return "\n".join(formatted_lines)


def compute_mafia_reward(
    player_name: str,
    role: str,
    base_reward: float,
    won: bool,
    turns: List[MafiaTurnRecord],
    format_bonus: float = 0.1,
    format_penalty: float = 0.2,
) -> float:
    """Compatibility helper: only the team outcome is rewarded, never participation."""
    return float(base_reward)


class MafiaRolloutManager:
    """
    Manages multi-turn game rollouts between RL Policy agent(s) and opponent agent(s).
    Supports dynamic player counts (e.g. 4 to 8 players, configurable or randomized per episode).
    Collects trajectories with prompts, token/text actions, and final outcome rewards.
    """

    def __init__(
        self,
        player_names: Optional[List[str]] = None,
        num_players: Optional[int] = None,
        min_players: int = 4,
        max_players: int = 6,
        dynamic_player_count: bool = False,
        role_mapping: Optional[Dict[str, str]] = None,
        role_counts: Optional[Dict[str, int]] = None,
        discussion_rounds: Optional[int] = None,
        max_days: int = 5,
        max_total_steps: int = 500,
        system_prompts: Optional[Dict[str, str]] = None,
        moderator_fn=None,
        **discussion_options,
    ):
        self.dynamic_player_count = dynamic_player_count
        self.min_players = max(3, min_players)
        self.max_players = max(self.min_players, max_players)

        if player_names is not None:
            self.player_names = player_names
        elif num_players is not None:
            self.player_names = [f"Player {i + 1}" for i in range(num_players)]
        else:
            self.player_names = [f"Player {i + 1}" for i in range(min_players)]

        self.role_mapping = role_mapping
        self.role_counts = role_counts
        self.discussion_rounds = discussion_rounds
        self.max_days = max_days
        self.max_total_steps = max_total_steps
        self.system_prompts = system_prompts or {}
        self.moderator_fn = moderator_fn
        self.discussion_options = discussion_options

    def get_current_player_names(self) -> List[str]:
        """Returns the list of player names for an episode (dynamically sampled if enabled)."""
        if self.dynamic_player_count:
            import random
            n = random.randint(self.min_players, self.max_players)
            return [f"Player {i + 1}" for i in range(n)]
        return self.player_names

    def create_env(self, player_names: Optional[List[str]] = None) -> Mafia:
        names = player_names if player_names is not None else self.get_current_player_names()
        return Mafia(
            player_names=names,
            role_mapping=self.role_mapping if (self.role_mapping and set(names) <= set(self.role_mapping.keys())) else None,
            role_counts=self.role_counts,
            discussion_rounds=self.discussion_rounds,
            max_days=self.max_days,
            **self.discussion_options,
        )

    def rollout_episode(
        self,
        policy_agent_names: Union[str, List[str]],
        policy_fn: Callable[[str, str], str],  # fn(player_name, prompt) -> response
        opponent_fn: Callable[[str, str], str],  # fn(player_name, prompt) -> response
        player_names: Optional[List[str]] = None,
    ) -> MafiaEpisodeResult:
        """
        Executes one complete Mafia game episode and collects trajectories for training.

        Args:
            policy_agent_names: The player(s) being trained via RL (e.g. ["Player 1"]).
            policy_fn: Function to generate action text from the policy model.
            opponent_fn: Function to generate action text for the other players.
            player_names: Optional list of players for this specific episode.
        """
        if isinstance(policy_agent_names, str):
            policy_agent_names = [policy_agent_names]
        policy_agents_set = set(policy_agent_names)

        env = self.create_env(player_names)
        trajectories: Dict[str, MafiaTrajectory] = {}
        for name in env.player_names:
            role = env.player_roles.get(name, "villager")
            trajectories[name] = MafiaTrajectory(player_name=name, role=role)

        def query(request):
            callback = policy_fn if request.player_name in policy_agents_set else opponent_fn
            return callback(request.player_name, request.prompt)

        moderator_query = None
        if self.moderator_fn:
            moderator_query = lambda req: self.moderator_fn(req.player_name, req.prompt)
        elif env.discussion_moderator:
            from ..agent import Player
            from ..config import AgentConfig
            cfg = dict(env.discussion_moderator)
            cfg.setdefault("name", "Moderator")
            cfg.setdefault("role_desc", "Facilitate a fair Mafia discussion.")
            moderator = Player.from_config(AgentConfig(cfg))
            moderator_query = lambda req: moderator.act(
                list(req.observation), Message("System", req.instruction, -1))

        # Callbacks can share a single GPU model. Run them serially; the scheduling
        # and observation semantics are identical to the parallel Arena adapter.
        controller = MafiaDiscussionController(
            env, query, moderator_query=moderator_query,
            system_prompts=self.system_prompts, parallel=False,
        )
        reason = "step_limit"
        try:
            for _ in range(self.max_total_steps):
                if env.is_terminal():
                    break
                controller.advance()
            if env.is_terminal():
                reason = env.termination_reason or "game_end"
        except DiscussionError as exc:
            reason = "model_error: " + str(exc)
        finally:
            controller.close()

        truncated = not env.is_terminal()
        for i, record in enumerate(controller.records):
            req = record.request
            if req.player_name not in trajectories:  # Moderator is not a policy player.
                continue
            trajectories[req.player_name].turns.append(MafiaTurnRecord(
                turn=i, phase=req.phase, prompt=req.prompt, response=record.response,
                action_valid=record.valid, kind=req.kind, version=req.version,
                session_id=req.session_id, score=record.score, selected=record.selected,
                trainable=record.trainable and not truncated, stale=record.stale,
                retry=req.retry, error=record.error,
            ))

        raw_rewards = env.get_rewards()

        # Determine winner
        alive_mafia = env._get_players_by_role(MAFIA, alive_only=True)
        alive_citizens = [p for p in env.alive_players if env.player_roles[p] != MAFIA]
        if truncated:
            winner = "truncated"
        elif len(alive_mafia) == 0:
            winner = "citizens"
        elif len(alive_mafia) >= len(alive_citizens):
            winner = "mafia"
        else:
            winner = "draw"

        # Finalize trajectories with rewards
        for name, traj in trajectories.items():
            traj.raw_reward = raw_rewards.get(name, 0.0)
            traj.won = (
                (traj.role == MAFIA and winner == "mafia")
                or (traj.role != MAFIA and winner == "citizens")
            )
            traj.shaped_reward = compute_mafia_reward(
                player_name=name,
                role=traj.role,
                base_reward=traj.raw_reward,
                won=traj.won,
                turns=traj.turns,
            )

        return MafiaEpisodeResult(
            trajectories=trajectories,
            winner=winner,
            rewards=raw_rewards,
            all_messages=env.get_observation(None),
            day=env.day,
            truncated=truncated,
            termination_reason=reason,
            diagnostics=controller.diagnostics(),
            discussion_endings=list(env.discussion_endings),
        )

