from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..backends.base import IntelligenceBackend
from ..environments.mafia import MAFIA, Mafia
from ..message import Message


@dataclasses.dataclass
class MafiaTurnRecord:
    turn: int
    phase: str
    prompt: str
    response: str
    action_valid: bool = True


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
    winner: str  # "mafia", "citizens", or "draw"
    rewards: Dict[str, float]
    all_messages: List[Message]
    day: int

    def summary(self) -> str:
        """Returns a clean, high-level summary of the game outcome."""
        lines = [
            "================ MAFIA EPISODE RESULT ================",
            f"Winner: {self.winner.upper()} | Finished on Day: {self.day}",
            "------------------------------------------------------",
            "Players & Outcomes:",
        ]
        for name, traj in sorted(self.trajectories.items()):
            status = "WON" if traj.won else "LOST"
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
                        lines.append(f"Turn {t.turn} [{t.phase}]:")
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
    """
    Computes shaped reward for RL training:
    - Win/Loss base reward: +1.0 (win) / -1.0 (loss) / 0.0 (draw)
    - Format adherence bonus/penalty: checks if model responses were non-empty and well-formed
    """
    total_reward = base_reward

    # Small shaping: reward format adherence / penalize empty turns
    for t in turns:
        if len(t.response.strip()) == 0:
            total_reward -= format_penalty
        elif not t.action_valid:
            total_reward -= format_penalty * 0.5
        else:
            total_reward += format_bonus * 0.2

    return total_reward


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
        discussion_rounds: int = 1,
        max_days: int = 5,
        max_total_steps: int = 30,
        system_prompts: Optional[Dict[str, str]] = None,
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

        step_count = 0
        while not env.is_terminal() and step_count < self.max_total_steps:
            current_player = env.get_next_player()
            current_phase = env.phase
            obs_messages = env.get_observation(current_player)

            sys_prompt = self.system_prompts.get(
                current_player,
                "You are participating in a game of Mafia. Respond clearly and stay in character.",
            )
            prompt = format_observation_as_prompt(obs_messages, system_prompt=sys_prompt)

            # Query policy or opponent
            if current_player in policy_agents_set:
                response = policy_fn(current_player, prompt)
            else:
                response = opponent_fn(current_player, prompt)

            # Record turn in trajectory
            record = MafiaTurnRecord(
                turn=step_count,
                phase=current_phase,
                prompt=prompt,
                response=response,
                action_valid=len(response.strip()) > 0,
            )
            trajectories[current_player].turns.append(record)

            env.step(current_player, response)
            step_count += 1

        raw_rewards = env.get_rewards()

        # Determine winner
        alive_mafia = env._get_players_by_role(MAFIA, alive_only=True)
        alive_citizens = [p for p in env.alive_players if env.player_roles[p] != MAFIA]
        if len(alive_mafia) == 0:
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
        )

