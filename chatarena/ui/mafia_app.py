"""Mafia-only Gradio UI, using the shared game session and discussion controller."""
from __future__ import annotations

import html
from copy import deepcopy

import gradio as gr

from ..arena import Arena
from ..config import ArenaConfig
from .mafia_session import SessionRegistry

ROLE_NAMES = {"mafia": "마피아", "doctor": "의사", "police": "경찰", "villager": "시민"}
DEFAULT_MODEL = "gpt-6-luna"
PHASE_NAMES = {
    "NIGHT_MAFIA": "밤 · 마피아 행동",
    "NIGHT_DOCTOR": "밤 · 의사 보호",
    "NIGHT_POLICE": "밤 · 경찰 조사",
    "DAY_DISCUSSION": "낮 · 자유 토론",
    "DAY_VOTING": "낮 · 투표",
}
MODERATOR_GUIDE = "토론을 중립적으로 진행하세요. 새로운 의견을 유도하되, 같은 주장이 반복되거나 충분히 토론했다면 투표로 넘어가세요."
GLOBAL_PROMPT = (
    "마피아 게임에 참여하고 있습니다. 비밀 역할과 현재 단계의 지시를 따르세요. "
    "밤 행동, 투표, 토론을 포함한 모든 대사는 자연스러운 한국어로 짧게 1~3문장으로 말하세요. "
    "대상을 지목할 때는 표시된 플레이어 이름을 그대로 사용하세요. "
    "발언 의향 요청에는 0~3 중 숫자 하나만, 발언 요청에는 대사 또는 PASS만 출력하세요. "
    "밤 행동과 투표에서는 선택 가능한 대상 한 명을 명확히 지목하세요. "
    "다른 플레이어나 사회자를 사칭하지 마세요."
)

CSS = """
.gradio-container {max-width: 1220px !important; margin: auto;}
#mafia-header {padding: 18px 0 10px;}
#mafia-status {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px;}
"""


def make_config(player_count, participate, model, temperature, max_tokens,
                moderator_enabled, moderator_model, moderator_guide,
                max_messages, max_rounds, seconds, silence_seconds, max_days, reveal_roles):
    count = int(player_count)
    if not 3 <= count <= 10:
        raise ValueError("참가자 수는 3~10명이어야 합니다.")
    if not model.strip():
        raise ValueError("AI 플레이어의 모델 이름을 입력해주세요.")
    if moderator_enabled and not moderator_model.strip():
        raise ValueError("사회자 모델 이름을 입력해주세요.")
    backend = {"backend_type": "openai-chat", "model": model.strip(),
               "temperature": float(temperature), "max_tokens": int(max_tokens)}
    players = []
    for i in range(count):
        name = f"Player {i + 1}"
        players.append({
            "name": name, "role_desc": f"당신은 {name}입니다. 자신의 팀이 승리하도록 행동하세요.",
            "backend": {"backend_type": "human"} if participate and i == 0 else dict(backend),
        })
    moderator = None
    if moderator_enabled:
        moderator = {
            "name": "Moderator", "role_desc": moderator_guide.strip() or MODERATOR_GUIDE,
            "global_prompt": "진행 안내는 한국어로 하세요. 제어 명령은 요청된 영문 그대로 출력하세요.",
            "backend": dict(backend, model=moderator_model.strip(), temperature=0.3),
        }
    return ArenaConfig(
        players=players, global_prompt=GLOBAL_PROMPT,
        environment={
            "env_type": "mafia", "max_days": int(max_days),
            "max_discussion_messages": int(max_messages), "max_intent_rounds": int(max_rounds),
            "discussion_seconds": float(seconds), "silence_seconds": float(silence_seconds),
            "reveal_role_on_death": bool(reveal_roles), "discussion_moderator": moderator,
        },
    )


class MafiaBlocks(gr.Blocks):
    def get_api_info(self, all_endpoints=False):
        info = super().get_api_info(all_endpoints=all_endpoints)
        # Gradio 4.44's browser submits numeric function IDs, but its server
        # advertises only named endpoints. Supply the numeric lookup as well
        # so the browser can validate polling inputs against the right schema.
        for fn_id, fn in self.fns.items():
            endpoint = info["named_endpoints"].get("/" + str(fn.api_name))
            if endpoint is not None:
                info["unnamed_endpoints"][str(fn_id)] = deepcopy(endpoint)
        return info


class MafiaUI:
    def __init__(self, arena_factory=Arena.from_config):
        self.arena_factory = arena_factory
        self.registry = SessionRegistry()
        with MafiaBlocks(title="Mafia · 자유 토론", css=CSS, analytics_enabled=False) as self.demo:
            # Only an opaque session handle is sent; game state stays server-side.
            self.key = gr.Textbox(value="", visible=False)
            gr.Markdown("# MAFIA\n발언할 때를 스스로 고르는 AI들과 함께하는 마피아 게임", elem_id="mafia-header")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 새 게임 설정")
                    count = gr.Slider(3, 10, value=4, step=1, label="전체 참가자 수")
                    human = gr.Checkbox(True, label="직접 참여 · Player 1", info="끄면 비밀 정보를 포함한 전체 관전 모드입니다.")
                    gr.Markdown("역할은 무작위로 배정합니다. 4~6명: 마피아 1 · 의사 1 · 경찰 1 · 나머지 시민. 7명 이상은 마피아 2명입니다.")
                    model = gr.Textbox(value=DEFAULT_MODEL, label="AI 플레이어 모델", info="OpenAI API 모델 이름 · API 키는 .env의 OPENAI_API_KEY 사용")
                    with gr.Accordion("AI 응답 설정", open=False):
                        temperature = gr.Slider(0, 2, value=0.7, step=0.1, label="응답 다양성")
                        tokens = gr.Slider(32, 16384, value=4096, step=1, label="최대 생성 토큰",
                                           info="추론 모델은 내부 추론 토큰도 포함합니다. 토큰 한도로 빈 응답이 발생하면 늘려주세요.")
                    moderator_enabled = gr.Checkbox(False, label="토론 진행자 LLM 사용", info="꺼져 있어도 게임 안내와 투표 진행은 자동으로 처리됩니다.")
                    with gr.Group(visible=False) as moderator_settings:
                        moderator_model = gr.Textbox(value=DEFAULT_MODEL, label="진행자 모델")
                        moderator_guide = gr.Textbox(value=MODERATOR_GUIDE, lines=3, label="토론 진행 지침")
                    with gr.Accordion("토론과 게임 규칙", open=False):
                        max_messages = gr.Slider(1, 60, value=24, step=1, label="하루 최대 발언 수")
                        seconds = gr.Slider(15, 600, value=180, step=15, label="하루 토론 시간 (초)")
                        silence = gr.Slider(0, 60, value=10, step=1, label="침묵 시 인간 입력 대기 (초)")
                        max_rounds = gr.Slider(1, 100, value=48, step=1, label="최대 발언 의향 수집 횟수")
                        days = gr.Slider(1, 15, value=5, step=1, label="최대 게임 일수")
                        reveal = gr.Checkbox(True, label="사망 시 역할 공개")
                    gr.Markdown("설정은 새 게임에 적용됩니다. 진행 중 변경하려면 먼저 **초기화**하세요.")
                with gr.Column(scale=2, min_width=400):
                    self.status = gr.Markdown("**대기 중** · 설정을 확인하고 게임을 시작하세요.", elem_id="mafia-status")
                    with gr.Row():
                        self.start_button = gr.Button("게임 시작", variant="primary")
                        self.pause_button = gr.Button("일시정지", interactive=False)
                        self.clear_button = gr.Button("초기화")
                    self.identity = gr.Markdown("참여하면 자신의 비밀 역할과 메시지만 표시됩니다.")
                    self.roster = gr.Markdown("")
                    # Gradio 4's Chatbot label renders an invalid label for="".
                    gr.Markdown("### 게임 대화")
                    # Gradio 4 groups consecutive messages with the same role.
                    # One tuple per utterance keeps every speaker's bubble separate.
                    self.chat = gr.Chatbot(type="tuples", height=480, label="게임 대화", show_label=False, elem_id="mafia-chat",
                                           placeholder="게임을 시작하면 역할 배정과 밤 안내가 표시됩니다.",
                                           show_copy_button=True)
                    with gr.Row():
                        self.input = gr.Textbox(label="내 발언 / 행동", placeholder="게임을 시작해주세요.",
                                                scale=5, interactive=False)
                        self.send_button = gr.Button("전송", scale=1, interactive=False)
                    self.feedback = gr.Markdown("")
                    gr.Markdown("낮 토론에는 언제든 끼어들 수 있습니다. 밤과 투표에는 자신의 차례에 행동하세요.")
            # Never make Chatbot an output of a polled backend event: Gradio 4
            # toggles its pending_message even when show_progress is hidden.
            self.chat_payload = gr.JSON(value=[], visible=False)
            self.chat_payload.change(
                fn=None, inputs=self.chat_payload, outputs=self.chat,
                js="(messages) => [messages || []]", queue=False, show_progress="hidden",
            )
            self.settings = [count, human, model, temperature, tokens, moderator_enabled,
                             moderator_model, moderator_guide, max_messages, max_rounds,
                             seconds, silence, days, reveal]
            self.live_outputs = [self.status, self.identity, self.roster, self.chat_payload,
                                 self.start_button, self.pause_button, self.input, self.send_button]
            self.start_button.click(self.start, [self.key, *self.settings],
                                    [self.key, *self.live_outputs, self.feedback])
            self.pause_button.click(self.pause, self.key, self.live_outputs, queue=False)
            self.clear_button.click(self.clear, self.key,
                                    [self.key, *self.live_outputs, self.feedback], queue=False)
            self.send_button.click(self.send, [self.key, self.input],
                                   [self.input, self.feedback], queue=False)
            self.input.submit(self.send, [self.key, self.input],
                              [self.input, self.feedback], queue=False)
            moderator_enabled.change(lambda enabled: gr.update(visible=enabled),
                                     moderator_enabled, moderator_settings, queue=False)
            self.timer = gr.Timer(0.5)
            # Markdown also dims and grows while pending, even with progress
            # hidden. Poll into an invisible transport, then apply the result
            # locally so no visible component enters a network loading state.
            self.poll_payload = gr.JSON(visible=False)
            self.poll_payload.change(
                fn=None, inputs=self.poll_payload, outputs=self.live_outputs,
                js="(updates) => updates", queue=False, show_progress="hidden",
            )
            self.timer.tick(self.poll, self.key, self.poll_payload, queue=False,
                            show_progress="hidden")

    def unchanged(self):
        # Gradio 4 does not expand an empty dictionary into skipped outputs.
        return {comp: gr.skip() for comp in self.live_outputs}

    def poll(self, key):
        updates = self.refresh(key)
        return [updates.get(comp, gr.skip()) for comp in self.live_outputs]

    def render(self, session):
        with session.lock:
            snap = session.snapshot()
            env = session.arena.environment
            ctrl = session.controller
            state = snap["status"]
            human = snap["human"]
            own_alive = human in env.alive_players
            discussion = env.phase == "DAY_DISCUSSION"
            next_player = None if discussion or snap["terminal"] else env.get_next_player()
            can_send = bool(human and own_alive and not snap["terminal"] and
                            (discussion or next_player == human))
            phase = PHASE_NAMES.get(env.phase, env.phase)
            if human and env.phase.startswith("NIGHT_") and next_player != human:
                phase = "밤 · 비공개 행동 진행"
            labels = {"paused": "일시정지", "running": "진행 중", "waiting_human": "입력 대기",
                      "waiting_capacity": "모델 호출 대기", "error": "오류 · 일시정지", "terminal": "게임 종료"}
            detail = ""
            if snap["terminal"]:
                detail = {"citizens_win": "시민 팀 승리", "mafia_win": "마피아 팀 승리",
                          "max_days": "최대 일수 도달 · 무승부"}.get(env.termination_reason, "게임 종료")
            elif state == "error":
                detail = html.escape(snap["error"] or "모델 호출을 확인하고 다시 시작해주세요.")
            elif ctrl.pending and not ctrl.paused:
                done = sum(f.done() for _, f in ctrl.pending)
                kind = ctrl.pending[0][0].kind
                task = {"intent": "발언 의향 수집", "speech": "AI 발언 생성",
                        "moderation": "진행자 판단", "game_action": "밤 / 투표 행동 생성"}[kind]
                elapsed = max(0, int(ctrl.clock() - ctrl._batch_started))
                detail = f"{task} · {done}/{len(ctrl.pending)} 완료 · {elapsed}초 대기"
            elif next_player:
                # Only name the actor if this human is entitled to know it.
                detail = (f"{next_player}의 행동 차례" if not human or next_player == human
                          or env.phase == "DAY_VOTING" else "밤 행동을 처리하고 있습니다.")
            elif state == "waiting_human":
                detail = "추가 발언을 기다리고 있습니다."
            status = f"**{env.day}일차 · {phase}** — {labels.get(state, state)}\n\n{detail}"
            identity = (f"**내 이름: {human} · 역할: {ROLE_NAMES[env.player_roles[human]]}**"
                        if human else "**전체 관전 모드** · 비밀 역할과 야간 행동도 표시됩니다.")
            if human and not own_alive:
                identity += " · 탈락 (발언 불가)"
            def player_label(name):
                if not human and name in env.player_roles:
                    role = env.player_roles[name]
                    return f"{name} ({ROLE_NAMES.get(role, role)})"
                return name

            roster = " · ".join(f"{player_label(name)} {'●' if name in env.alive_players else '탈락'}"
                                for name in env.player_names)
            messages = []
            for msg in snap["messages"]:
                private = " · 비공개" if msg.visible_to != "all" else ""
                if private and not human:
                    recipients = msg.visible_to if isinstance(msg.visible_to, str) else ", ".join(msg.visible_to)
                    private += " → " + html.escape(recipients)
                speaker = "사회자" if msg.agent_name == "Moderator" else player_label(msg.agent_name)
                content = f"**{html.escape(speaker)}{private}**\n\n{msg.content}"
                messages.append((content, None) if human and msg.agent_name == human
                                else (None, content))
            placeholder = "토론 중 언제든 발언하세요." if discussion else "대상 이름을 포함해 행동을 입력하세요."
            if not can_send:
                placeholder = "관전 중입니다." if not human else "자신의 행동 차례를 기다려주세요."
            return {
                self.status: status, self.identity: identity, self.roster: roster, self.chat_payload: self.chat.postprocess(messages).model_dump(),
                self.start_button: gr.update(value="계속하기", interactive=state in ("paused", "error") and not snap["terminal"]),
                self.pause_button: gr.update(interactive=not ctrl.paused and not snap["terminal"]),
                self.input: gr.update(interactive=can_send, placeholder=placeholder),
                self.send_button: gr.update(interactive=can_send),
            }

    def refresh(self, key):
        session = self.registry.get(key)
        if session is None:
            return self.unchanged()
        with session.lock:
            rendered = self.render(session)
            # Each session has its own browser view. Polling the waiting timer
            # must not redraw unchanged chat bubbles or reset input properties.
            previous = getattr(session, "_mafia_ui_render", {})
            updates = {comp: gr.skip() if previous.get(comp) == value else value
                       for comp, value in rendered.items()}
            # Gradio mutates update dictionaries during postprocessing.
            session._mafia_ui_render = {comp: deepcopy(value) for comp, value in rendered.items()}
            return updates

    def start(self, key, *settings):
        session = self.registry.get(key)
        if session is None:
            try:
                config = make_config(*settings)
                key = self.registry.add(self.arena_factory(config))
                session = self.registry.get(key)
            except Exception as exc:
                return {self.key: key, **self.unchanged(),
                        self.feedback: "시작하지 못했습니다: " + html.escape(str(exc))}
        session.start()
        return {self.key: key, **self.render(session), self.feedback: ""}

    def pause(self, key):
        session = self.registry.get(key)
        if session:
            session.pause()
            return self.render(session)
        return self.unchanged()

    def send(self, key, text):
        session = self.registry.get(key)
        if session is None:
            return {self.input: gr.skip(), self.feedback: "게임을 먼저 시작해주세요."}
        try:
            session.submit(text)
        except ValueError as exc:
            return {self.input: gr.skip(), self.feedback: "전송하지 못했습니다: " + html.escape(str(exc))}
        return {self.input: "", self.feedback: ""}

    def clear(self, key):
        self.registry.remove(key)
        return {
            self.key: "", self.status: "**대기 중** · 새 게임을 시작할 수 있습니다.",
            self.identity: "참여하면 자신의 비밀 역할과 메시지만 표시됩니다.",
            self.roster: "", self.chat_payload: [], self.feedback: "",
            self.start_button: gr.update(value="게임 시작", interactive=True),
            self.pause_button: gr.update(interactive=False),
            self.input: gr.update(value="", interactive=False, placeholder="게임을 시작해주세요."),
            self.send_button: gr.update(interactive=False),
        }
