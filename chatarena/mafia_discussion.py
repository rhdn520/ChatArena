"""Shared, nonblocking Mafia orchestration. Only this owner commits model results.

Callbacks run outside the state lock. One bounded batch is outstanding at a time;
invalidating a batch never queues replacement work until its futures drain.
"""
from __future__ import annotations

import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .agent import SIGNAL_END_OF_CONVERSATION
from .message import Message

INTENT_PROMPT = (
    "REQUEST intent: 현재 얼마나 발언하고 싶은지 비공개로 판단하세요. "
    "0=듣기, 1=기다려도 됨, 2=말하고 싶음, 3=지금 답변하거나 반박하고 싶음. "
    "대화와 자신의 역할을 고려하세요. 침묵해도 괜찮습니다. "
    "0, 1, 2, 3 중 숫자 하나만 출력하고 이유는 쓰지 마세요."
)
SPEECH_PROMPT = (
    "REQUEST speech: 지금 토론에 참여할 수 있습니다. "
    "현재는 낮 토론입니다. 조사·보호·야간 공격은 밤의 해당 역할 차례에만 가능합니다. "
    "낮에 새 조사를 실행하거나 새 결과를 요청하지 마세요. 지난밤의 결과를 공개하거나 "
    "자신의 추측·주장을 말할 수는 있지만, 플레이어의 말이 공식 조사 결과가 되는 것은 아닙니다. "
    "최근 대화에 자연스러운 한국어로 짧게 1~3문장으로 답하세요. "
    "듣기로 마음을 바꿨다면 따옴표나 문장부호 없이 PASS만 출력하세요. "
    "CONTINUE와 END_DISCUSSION은 사회자 전용 명령입니다. 다른 사람을 사칭하지 마세요."
)
_CALL_SLOTS = threading.BoundedSemaphore(32)

MODERATION_PROMPT = (
    "REQUEST moderation: 토론을 이끌지 말고 종료가 필요한지만 조용히 판단하세요. "
    "기본 응답은 CONTINUE입니다. 새로운 근거·질문·답변·반박이 오가면 개입하지 마세요. "
    "단순히 같은 사람을 의심하거나 발언이 짧다는 이유로 종료하지 마세요. "
    "충분한 답변·반론 기회가 있었고 새 정보 없이 같은 주장만 계속 반복되거나, "
    "논의가 충분히 끝나 투표할 시점일 때만 END_DISCUSSION을 출력하세요. "
    "특정인을 지목하거나 발언을 요구하지 마세요. 발언 순서 배정, 의견 유도, 요약, "
    "평가, 역할 추측, 안내문은 금지합니다. CONTINUE 또는 END_DISCUSSION 중 하나만 출력하세요."
)


def observation_prompt(name, observation, instruction, system_prompt=""):
    lines = [f"System: {system_prompt}\nYou are {name}."]
    lines.extend(f"{m.agent_name}: {m.content.strip()}" for m in observation)
    lines.extend([instruction, "You: "])
    return "\n".join(lines)


def control_token(response):
    """Recognize standalone control replies without changing ordinary speech."""
    token = response.strip(" \t\r\n`*\"'“”‘’.!。").upper()
    if re.fullmatch(r"I(?:'LL| WILL) (?:JUST )?LISTEN(?: FOR NOW)?[.!]?\s+PASS", token):
        return "PASS"
    return token if token in {"PASS", "CONTINUE", "END_DISCUSSION"} else None


@dataclass
class DecisionRequest:
    player_name: str
    kind: str
    observation: Tuple[Message, ...]
    instruction: str
    prompt: str
    session_id: str
    version: int
    phase: str
    generation: int
    retry: bool = False


@dataclass
class DecisionRecord:
    request: DecisionRequest
    response: str = ""
    valid: bool = False
    trainable: bool = False
    score: Optional[int] = None
    selected: bool = False
    stale: bool = False
    error: Optional[str] = None


class DiscussionError(RuntimeError):
    pass


class MafiaDiscussionController:
    def __init__(self, env, query: Callable[[DecisionRequest], str], human_names=(),
                 moderator_query=None, system_prompts=None, realtime=False,
                 parallel=True, max_workers=4, request_timeout=60.0,
                 clock=time.monotonic):
        self.env = env
        self.query = query
        self.human_names = set(human_names)
        self.moderator_query = moderator_query
        self.system_prompts = system_prompts or {}
        self.realtime = realtime
        self.clock = clock
        self.request_timeout = request_timeout
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=max_workers if parallel else 1)
        self.records: List[DecisionRecord] = []
        self.pending = []
        self.generation = 0
        self.paused = False
        self.closed = False
        self.error = None
        self.status = "running"
        self.rng = random.Random(env.seed)
        self._key = None
        self._stage = "intent"
        self._scores = {}
        self._score_records = {}
        self._retry_names = []
        self._rounds = 0
        self._silence_prompted = False
        self._silence_deadline = None
        self._elapsed = 0.0
        self._last_clock = clock()
        self._batch_started = 0.0
        self._moderating_silence = False

    @property
    def _use_moderator(self):
        return self.moderator_query is not None and self.env.phase == "DAY_DISCUSSION"

    def _sync_day(self):
        key = (self.env.session_id, self.env.day, self.env.phase)
        if key != self._key:
            self._key = key
            self._stage = "intent"
            self._scores = {}
            self._score_records = {}
            self._retry_names = []
            self._rounds = 0
            self._silence_prompted = False
            self._silence_deadline = None
            self._elapsed = 0.0

    def _request(self, name, kind, retry=False):
        env = self.env
        if kind == "moderation":
            # Do not use get_observation('Moderator'): MessagePool grants it omniscience.
            obs = tuple(m for m in env.get_observation() if m.visible_to == "all")
            instruction = MODERATION_PROMPT + (
                f"\n생존자: {', '.join(env.randomized_names(env.alive_players))}. "
                f"플레이어 발언 수: {env.discussion_messages}. "
                f"의향 수집 횟수: {self._rounds}. 경과 시간: {self._elapsed:.1f}초."
            )
            if self._moderating_silence:
                instruction += "\n모두 침묵하고 있으며 입력 대기도 끝났습니다. 발언을 강요하지 말고 END_DISCUSSION으로 투표로 넘기세요."
        else:
            obs = tuple(env.get_observation(name))
            instruction = {"intent": INTENT_PROMPT, "speech": SPEECH_PROMPT}.get(
                kind, f"REQUEST game_action: {name}의 현재 단계는 {env.phase}입니다. 현재 단계의 지시를 따르고 한국어로 응답하세요. 대상 이름은 표시된 그대로 쓰세요."
            )
        if env.phase == "NIGHT_MAFIA_DISCUSSION":
            context = ("현재는 밤의 마피아 전용 비공개 의논입니다. 생존 동료 마피아에게만 전달됩니다. "
                       "오늘 밤 공격할 대상을 함께 상의하세요. 이 단계의 발언은 공격 확정이 아닙니다. ")
            if kind == "speech":
                instruction = ("REQUEST speech: " + context
                               + "동료의 의견에 한국어 1~3문장으로 답하세요. 말하지 않으려면 PASS만 출력하세요.")
            elif kind == "intent":
                instruction += "\n" + context
        if retry:
            instruction += "\n이전 응답의 형식이 잘못되었습니다. 0~3 중 숫자 하나만 출력하세요."
        return DecisionRequest(name, kind, obs, instruction,
                               observation_prompt(name, obs, instruction, self.system_prompts.get(name, "")),
                               env.session_id, env.version, env.phase, self.generation, retry)

    def _submit(self, names, kind, retry=False):
        assert not self.pending
        if len(names) > 32:
            self._fail("At most 32 simultaneous AI participants are supported")
            return False
        reserved = 0
        for _ in names:
            if not _CALL_SLOTS.acquire(blocking=False):
                for _ in range(reserved):
                    _CALL_SLOTS.release()
                self.status = "waiting_capacity"
                return False
            reserved += 1
        callback = self.moderator_query if kind == "moderation" else self.query
        self._batch_started = self.clock()
        for name in names:
            req = self._request(name, kind, retry)
            future = self.executor.submit(callback, req)
            future.add_done_callback(lambda _: _CALL_SLOTS.release())
            self.pending.append((req, future))
        return True

    def _fresh(self, req):
        return (not self.closed and req.generation == self.generation
                and req.session_id == self.env.session_id
                and req.version == self.env.version and req.phase == self.env.phase)

    def _fail(self, reason):
        self.error = reason
        self.paused = True
        self.status = "error"
        self.generation += 1
        self._stage = "intent"
        self._scores = {}
        self._retry_names = []

    def pause(self):
        with self.lock:
            self._advance_clock()
            self.paused = True
            self.generation += 1
            self.status = "paused"
            self._silence_deadline = None
            # Resume reevaluates the latest conversation, not an old selection.
            if self._stage not in ("moderation", "silence_moderation"):
                self._stage = "intent"
            self._scores = {}
            self._retry_names = []

    def resume(self):
        with self.lock:
            self.paused = False
            self.error = None
            self.status = "running"
            self._last_clock = self.clock()

    def close(self):
        with self.lock:
            self.closed = True
            self.generation += 1
            for _, future in self.pending:
                future.cancel()
            self.executor.shutdown(wait=False)

    def reset(self):
        with self.lock:
            self.generation += 1
            self.env.reset()
            self._key = None
            self._sync_day()
            self.records.clear()
            self.error = None
            self.paused = False
            self._last_clock = self.clock()

    def _advance_clock(self):
        now = self.clock()
        if not self.paused and self.env.is_discussion:
            self._elapsed += max(0, now - self._last_clock)
        self._last_clock = now

    def _limit(self):
        env = self.env
        private = env.phase == "NIGHT_MAFIA_DISCUSSION"
        messages = env.mafia_discussion_messages if private else env.max_discussion_messages
        rounds = env.mafia_intent_rounds if private else env.max_intent_rounds
        seconds = env.mafia_discussion_seconds if private else env.discussion_seconds
        if env.discussion_messages >= messages:
            return "message_limit"
        # The last allowed round may still select and publish a speech.
        if self._rounds >= rounds and self._stage == "intent" and not self.pending:
            return "intent_limit"
        if self.realtime and self._elapsed >= seconds:
            return "time_limit"
        return None

    def _end(self, reason):
        self.env.end_discussion(reason)
        self.generation += 1
        self._stage = "intent"
        self._silence_deadline = None

    def submit_human(self, name, text):
        with self.lock:
            if self.closed or name not in self.human_names or name not in self.env.alive_players:
                raise ValueError("Not a living human player")
            if self.env.is_terminal() or not text.strip():
                raise ValueError("Game ended or empty input")
            self._sync_day()
            self._advance_clock()
            if self.env.is_discussion:
                if name not in self.env.discussion_participants:
                    raise ValueError("현재 비공개 의논에는 참여할 수 없습니다.")
                reason = self._limit()
                if reason:
                    self._end(reason)
                    raise ValueError("의논이 끝났습니다. 대상 선택 차례를 확인해주세요.")
                self.env.discussion_speak(name, text)
                self._stage = "moderation" if self._use_moderator else "intent"
                self._silence_deadline = None
                self._scores = {}
                self._retry_names = []
            else:
                if name != self.env.get_next_player():
                    raise ValueError("Wait for your night/voting turn")
                self.env.step(name, text)
                self._last_clock = self.clock()
            self.generation += 1
            return self.env.timestep()

    def selection_weights(self):
        return {name: self.env.intent_weights[score] * (
            self.env.repeat_speaker_factor if name == self.env.last_speaker else 1.0
        ) for name, score in sorted(self._scores.items()) if score > 0}

    def _silence(self):
        humans_alive = self.human_names & self.env.discussion_participants
        if self.realtime and humans_alive:
            if self._silence_deadline is None:
                self._silence_deadline = self.clock() + self.env.silence_seconds
            if self.clock() < self._silence_deadline:
                self.status = "waiting_human"
                return
        self._silence_deadline = None
        if self._use_moderator and not self._silence_prompted:
            self._silence_prompted = True
            self._stage = "silence_moderation"
        else:
            self._end("silence")

    def _drain(self):
        if not self.pending:
            return
        if not all(f.done() for _, f in self.pending):
            if self.clock() - self._batch_started > self.request_timeout and not self.error:
                self._fail("Model request timed out; outstanding calls must drain before retry")
            return
        batch, self.pending = self.pending, []
        results = []
        for req, future in batch:
            record = DecisionRecord(req)
            try:
                value = future.result()
                if not isinstance(value, str):
                    raise ValueError("Model response must be text")
                record.response = value.strip()
                if record.response.startswith(SIGNAL_END_OF_CONVERSATION):
                    raise ValueError("Backend failed to generate a response")
                record.valid = bool(record.response)
                command = control_token(record.response)
                allowed = {"speech": {"PASS"},
                           "moderation": {"CONTINUE", "END_DISCUSSION"}}
                if command and command not in allowed.get(req.kind, set()):
                    record.valid = False
                if req.kind == "moderation" and command not in allowed["moderation"]:
                    record.valid = False
                if req.kind == "intent":
                    record.valid = bool(re.fullmatch(r"[0-3]", record.response))
                    if record.valid:
                        record.score = int(record.response)
                if not record.valid:
                    record.error = "Invalid response format"
            except Exception as exc:
                record.error = f"{type(exc).__name__}: {exc}"
            record.stale = not self._fresh(req)
            record.trainable = (record.valid and not record.stale and not req.retry
                                and req.kind != "moderation")
            if control_token(record.response) and record.response != control_token(record.response):
                # Keep the raw output for diagnosis, but do not train on a
                # control reply that failed the requested exact output format.
                record.trainable = False
            self.records.append(record)
            if not record.stale:
                results.append(record)
        if not results:
            return
        kind = results[0].request.kind
        if kind == "intent":
            retry_names = []
            for rec in results:
                name = rec.request.player_name
                if rec.valid:
                    self._scores[name] = rec.score
                    self._score_records[name] = rec
                elif not rec.request.retry:
                    retry_names.append(name)
            if retry_names:
                self._retry_names = retry_names
                self._stage = "retry_intent"
            elif not self._scores:
                self._fail("All AI intent requests failed")
            else:
                self._stage = "choose"
        elif kind == "speech":
            rec = results[0]
            if not rec.valid:
                self._fail("Speech generation failed: " + str(rec.error))
            elif control_token(rec.response) == "PASS":
                self._scores.pop(rec.request.player_name, None)
                self._stage = "choose"
            else:
                self.env.discussion_speak(rec.request.player_name, rec.response)
                self._stage = "moderation" if self._use_moderator else "intent"
        elif kind == "moderation":
            rec = results[0]
            if not rec.valid:
                if rec.response and rec.error == "Invalid response format":
                    # Never publish unsolicited instructions to individual players.
                    # Retain the invalid decision internally, without another call
                    # on the same conversation or interruption of active discussion.
                    if self._moderating_silence:
                        self._end("silence")
                    else:
                        self._stage = "intent"
                else:
                    self._fail("Moderator failed: " + str(rec.error))
            elif control_token(rec.response) == "END_DISCUSSION":
                self._end("moderator")
            elif control_token(rec.response) == "CONTINUE":
                if self._moderating_silence:
                    self._end("silence")
                else:
                    self._stage = "intent"
        else:
            rec = results[0]
            if rec.valid:
                try:
                    self.env.step(rec.request.player_name, rec.response)
                except ValueError as exc:
                    rec.valid = rec.trainable = False
                    rec.error = str(exc)
                    self._fail("Game action failed: " + rec.error)
                    return
                self._last_clock = self.clock()
            else:
                self._fail("Game action failed: " + str(rec.error))

    def tick(self):
        """Perform a short nonblocking poll; callers may safely submit human input meanwhile."""
        with self.lock:
            if self.closed:
                return "closed"
            self._sync_day()
            self._advance_clock()
            if not self.paused and self.env.is_discussion:
                reason = self._limit()
                if reason:
                    self._end(reason)
            self._drain()
            # A completed night action can enter a new discussion immediately.
            # Initialize its counters before submitting the first intent batch.
            self._sync_day()
            if self.paused:
                return self.status
            if self.env.is_terminal():
                self.status = "terminal"
                return self.status
            self.status = "running"
            if self.pending:
                return self.status
            if not self.env.is_discussion:
                name = self.env.get_next_player()
                if name in self.human_names:
                    self.status = "waiting_human"
                else:
                    self._submit([name], "game_action")
                return self.status
            reason = self._limit()
            if reason:
                self._end(reason)
                return self.status
            if self._stage == "intent":
                names = sorted(self.env.discussion_participants - self.human_names)
                self._scores = {}
                self._score_records = {}
                if names:
                    if self._submit(names, "intent"):
                        self._rounds += 1
                else:
                    self._rounds += 1
                    self._stage = "silence"
            elif self._stage == "retry_intent":
                self._submit(self._retry_names, "intent", retry=True)
            elif self._stage == "choose":
                weights = self.selection_weights()
                if weights:
                    names = list(weights)
                    name = self.rng.choices(names, weights=list(weights.values()), k=1)[0]
                    if self._submit([name], "speech"):
                        self._score_records[name].selected = True
                else:
                    self._stage = "silence"
            elif self._stage in ("moderation", "silence_moderation"):
                self._moderating_silence = self._stage == "silence_moderation"
                self._submit(["Moderator"], "moderation")
            elif self._stage == "silence":
                self._silence()
            return self.status

    def advance(self):
        """Synchronous adapter for Arena/CLI/RL; returns after one public state change."""
        version = self.env.version
        while True:
            status = self.tick()
            if status == "error":
                raise DiscussionError(self.error)
            if status in ("terminal", "closed", "paused", "waiting_human") or self.env.version != version:
                return self.env.timestep()
            time.sleep(0.001)

    def diagnostics(self):
        with self.lock:
            result = {}
            for name in self.env.player_names:
                records = [r for r in self.records if r.request.player_name == name and not r.stale]
                intents = [r for r in records if r.request.kind == "intent" and r.valid]
                speeches = [r for r in records if r.request.kind == "speech" and r.valid]
                result[name] = {
                    "intent_counts": {str(i): sum(r.score == i for r in intents) for i in range(4)},
                    "speeches": sum(control_token(r.response) != "PASS" for r in speeches),
                    "passes": sum(control_token(r.response) == "PASS" for r in speeches),
                    "failures": sum(not r.valid for r in records),
                    "failure_rate": sum(not r.valid for r in records) / max(1, len(records)),
                    "pass_rate": sum(control_token(r.response) == "PASS" for r in speeches) / max(1, len(speeches)),
                }
            total = sum(v["speeches"] for v in result.values())
            for values in result.values():
                values["speech_share"] = values["speeches"] / max(1, total)
            return result
