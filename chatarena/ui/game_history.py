"""Read saved pilot/UI transcripts, including each message's original audience."""
import html
import json
import re
from copy import deepcopy
from pathlib import Path

import gradio as gr


GAME_NAMES = {"liar": "라이어", "mafia": "마피아"}
GAME_DIRS = {"liar": "liar", "liar_game": "liar", "mafia": "mafia"}
HEADER = re.compile(r"^\[\d+ \| turn ([^\n]*?) \| ([^\n]*?) \| ([^\n]*?)\]\r?$", re.MULTILINE)


class HistoryBlocks(gr.Blocks):
    def get_api_info(self, all_endpoints=False):
        # Match the numeric function-ID lookup used by the existing Gradio 4 UIs.
        info = super().get_api_info(all_endpoints=all_endpoints)
        for fn_id, fn in self.fns.items():
            endpoint = info["named_endpoints"].get("/" + str(fn.api_name))
            if endpoint is not None:
                info["unnamed_endpoints"][str(fn_id)] = deepcopy(endpoint)
        return info


def scan_records(root, game):
    """Index only known transcript layouts; never accept a browser file path."""
    records = {}
    candidates = list(root.glob("*/job_*/*/seed_*"))
    candidates.extend(root.glob("mixed_games/job_*/*/*.json"))
    for path in sorted(candidates):
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if GAME_DIRS.get(path.parent.name) != game:
            continue
        pilot = path.is_dir()
        if pilot and not any((path / name).is_file() for name in ("messages.json", "conversation.txt")):
            continue
        if not pilot and not path.is_file():
            continue
        job = path.parent.parent.name[len("job_"):]
        source = path.parent.parent.parent.name
        label = f"{path.name if pilot else path.stem} · {source}"
        records.setdefault(job, {})[label] = path
    return records


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_transcript(path):
    """Read the existing full-text format when structured messages aren't saved yet."""
    text = path.read_text(encoding="utf-8")
    headers = list(HEADER.finditer(text))
    if not headers and text.strip():
        raise ValueError("대화 파일에서 발언 구분을 읽지 못했습니다.")
    messages = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        turn, audience, speaker = header.groups()
        messages.append({"turn": turn, "speaker": speaker,
                         "audience_label": audience,
                         "content": text[header.end():end].strip()})
    return messages


def role_label(name, roles):
    name = str(name)
    return f"{name} ({roles.get(name, '역할 미기록')})" if re.fullmatch(r"Player \d+", name) else name


def audience_label(message, public_only, roles):
    if "audience_label" in message:
        return re.sub(r"\bPlayer \d+\b", lambda match: role_label(match.group(0), roles),
                      message["audience_label"])
    audience = message.get("visible_to")
    if audience == "all" or (audience is None and public_only):
        return "공개 · 모든 플레이어"
    if isinstance(audience, list):
        return "비공개 → " + (", ".join(role_label(name, roles) for name in audience) or "수신자 없음")
    if audience is not None:
        return "비공개 → " + role_label(audience, roles)
    return "공개 범위 미기록"


def player_roles(messages, metadata):
    """Read official initial role cards, never players' claims during discussion."""
    roles = {}
    for message in messages:
        if (message.get("agent_name", message.get("speaker")) != "Moderator"
                or str(message.get("turn")) != "0"):
            continue
        recipients = message.get("visible_to")
        if recipients is None:
            audience = message.get("audience_label", "")
            recipients = (re.findall(r"Player \d+", audience)
                          if audience.startswith(("비밀 →", "비공개 →")) else [])
        elif isinstance(recipients, str):
            recipients = [] if recipients == "all" else [recipients]
        content = str(message.get("content", "")).strip()
        mafia_role = re.match(r"당신의 비밀 역할은 \*\*(마피아|의사|경찰|시민)\*\*입니다\.", content)
        if mafia_role:
            role = mafia_role.group(1)
        elif content.startswith("You are a non-liar."):
            role = "일반인"
        elif content.startswith("You are the Liar."):
            role = "라이어"
        else:
            continue
        for name in recipients:
            roles[name] = role
    liar = metadata.get("result", {}).get("liar_name")
    if liar:
        for name in metadata.get("rewards", {}):
            roles[name] = "라이어" if name == liar else "일반인"
        roles[liar] = "라이어"
    return roles


def settings_details(metadata, messages, roles):
    def value_html(value):
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))

    roster = metadata.get("players")
    shared = {key: metadata[key] for key in (
        "temperature", "max_tokens", "enable_thinking", "thinking_token_budget",
        "control_thinking_token_budget", "request_timeout") if key in metadata}
    shared.update(metadata.get("sampling_parameters") or {})
    if not isinstance(roster, list):
        names = set(metadata.get("rewards", {})) | set(roles)
        names.update(str(m.get("agent_name", m.get("speaker", ""))) for m in messages)
        names = {name for name in names if re.fullmatch(r"Player \d+", name)}
        roster = [{"name": name, "type": "ai", "model": metadata.get("model"),
                   "settings": shared}
                  for name in sorted(names, key=lambda name: int(name.split()[-1]))]
    rows = []
    for player in roster:
        name = str(player.get("name", "미기록"))
        human = player.get("type") == "human"
        model = "—" if human else player.get("model") or "미기록"
        settings = player.get("settings")
        cells = [html.escape(name), html.escape(roles.get(name, "역할 미기록")),
                 "사람" if human else "AI", html.escape(str(model)),
                 "—" if human else (f"<pre>{value_html(settings)}</pre>" if settings else "미기록")]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    identity = {key: metadata[key] for key in (
        "game", "job_id", "seed", "session_id", "game_number", "finished_at") if key in metadata}
    saved = metadata.get("settings") or {}
    environment = dict(saved.get("environment") or {})
    for key in ("clue_rounds", "reveal_topic", "max_days", "reveal_role_on_death"):
        if key in metadata and key not in environment:
            environment[key] = metadata[key]
    moderator = saved.get("moderator") or {
        key: metadata[key] for key in ("moderator_enabled", "moderator_model") if key in metadata}
    return ("<div class='history-settings'><h3>플레이어별 설정</h3>"
            "<p>저장된 설정만 표시합니다. 미기록은 저장 정보 없음, null은 명시값 없음입니다.</p>"
            "<table><thead><tr><th>플레이어</th><th>역할</th><th>유형</th>"
            "<th>모델</th><th>생성 설정</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>"
            + "".join(f"<h3>{title}</h3><pre>{value_html(values) if values else '미기록'}</pre>"
                      for title, values in (("기록 정보", identity), ("게임 설정", environment),
                                            ("사회자 설정", moderator))) + "</div>")


def load_record(path):
    note = ""
    public_only = False
    if path.is_dir():
        summary = path / "summary.json"
        metadata = read_json(summary) if summary.is_file() else {}
        structured = path / "messages.json"
        if structured.is_file():
            messages = read_json(structured)
        else:
            messages = read_transcript(path / "conversation.txt")
            note = "conversation.txt에 저장된 전체 대화입니다. 실행 중인 기록은 새로고침하면 다시 읽습니다."
    else:
        metadata = read_json(path)
        messages = metadata.get("messages", [])
        public_only = metadata.get("message_scope") != "all"
        if public_only:
            note = "이전 UI 기록은 공개 대화만 저장했습니다. 저장되지 않은 비공개 대화는 표시할 수 없습니다."
    roles = player_roles(messages, metadata)
    chat = []
    speakers = sorted({str(m.get("agent_name", m.get("speaker", "알 수 없음")))
                       for m in messages})
    for index, message in enumerate(messages, 1):
        speaker = str(message.get("agent_name", message.get("speaker", "알 수 없음")))
        player = re.fullmatch(r"Player (\d+)", speaker)
        color = (int(player.group(1)) - 1 if player else speakers.index(speaker)) % 10 + 1
        speaker_class = "history-moderator" if speaker == "Moderator" else f"history-speaker-{color}"
        kind = message.get("msg_type", message.get("type", "text"))
        label = audience_label(message, public_only, roles)
        speaker_label = role_label(speaker, roles)
        heading = html.escape(f"#{index} · {speaker_label} · turn {message.get('turn', '—')} · {label} · {kind}")
        # Render log text literally so saved HTML cannot impersonate UI controls.
        content = html.escape(str(message.get("content", "")))
        bubble = (f"<div class='history-bubble {speaker_class}'>"
                  f"<div class='history-label'>{heading}</div>"
                  f"<div class='history-text'>{content}</div></div>")
        chat.append((None, bubble))
    result = metadata.get("winner") or metadata.get("termination_reason")
    if not result:
        result = metadata.get("result", {}).get("winner")
    if result:
        status = str(result)
    elif metadata.get("terminal") or metadata.get("finished_at"):
        status = "게임 종료"
    else:
        status = "완료 여부 미확인"
    info = f"**저장된 메시지 {len(messages)}개 · {html.escape(status)}**"
    if metadata.get("error"):
        info += "\n\n오류: " + html.escape(str(metadata["error"]))
    if note:
        info += "\n\n" + note
    return info, settings_details(metadata, messages, roles), chat


def build_demo(output_root):
    root = Path(output_root)

    def show(game, job, record):
        path = scan_records(root, game).get(job, {}).get(record)
        if path is None:
            return "조회할 기록이 없습니다.", "", []
        try:
            return load_record(path)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
            return "기록을 읽지 못했습니다: " + html.escape(str(exc)), "", []

    def choose_job(game, job, previous=None):
        records = scan_records(root, game).get(job, {})
        selected = previous if previous in records else next(iter(records), None)
        info, details, chat = show(game, job, selected)
        return gr.update(choices=list(records), value=selected), info, details, chat

    def refresh(game, job=None, record=None):
        catalog = scan_records(root, game)
        jobs = sorted(catalog, key=lambda value: (value.isdigit(), int(value) if value.isdigit() else value), reverse=True)
        selected = job if job in catalog else next(iter(jobs), None)
        selection, info, details, chat = choose_job(game, selected, record)
        return gr.update(choices=jobs, value=selected), selection, info, details, chat

    css = """
    .gradio-container {max-width: 1250px !important; margin: auto;}
    .history-bubble {padding: 12px 16px; border-radius: 10px; border-left: 4px solid; color: #0f172a;}
    .history-label {font-size: 12px; font-weight: 600; color: #334155; margin-bottom: 8px;}
    .history-text {white-space: pre-wrap; overflow-wrap: anywhere;}
    .history-moderator {background: #f1f5f9; border-color: #64748b;}
    .history-speaker-1 {background: #eff6ff; border-color: #2563eb;}
    .history-speaker-2 {background: #fff7ed; border-color: #ea580c;}
    .history-speaker-3 {background: #f0fdf4; border-color: #16a34a;}
    .history-speaker-4 {background: #faf5ff; border-color: #9333ea;}
    .history-speaker-5 {background: #fff1f2; border-color: #e11d48;}
    .history-speaker-6 {background: #ecfeff; border-color: #0891b2;}
    .history-speaker-7 {background: #fefce8; border-color: #ca8a04;}
    .history-speaker-8 {background: #fdf2f8; border-color: #db2777;}
    .history-speaker-9 {background: #eef2ff; border-color: #4f46e5;}
    .history-speaker-10 {background: #f0fdfa; border-color: #0d9488;}
    .history-settings {overflow-x: auto;}
    .history-settings table {width: 100%; border-collapse: collapse;}
    .history-settings th, .history-settings td {padding: 10px; border-bottom: 1px solid #cbd5e1; text-align: left; vertical-align: top;}
    .history-settings pre {white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px;}
    """
    with HistoryBlocks(title="게임 기록 조회", css=css, analytics_enabled=False) as demo:
        gr.Markdown("# 게임 기록 조회\n저장된 전체 대화를 순서대로 표시합니다. 비공개 역할·행동도 원래 수신 대상과 함께 보입니다.")
        with gr.Row():
            game = gr.Dropdown(choices=[(label, key) for key, label in GAME_NAMES.items()],
                               value="liar", label="게임", interactive=True)
            job = gr.Dropdown(choices=[], label="Job ID", interactive=True)
            record = gr.Dropdown(choices=[], label="Seed / 세션 · 판", interactive=True, scale=2)
            reload_button = gr.Button("목록·대화 새로고침")
        info = gr.Markdown("게임 기록을 선택해주세요.")
        with gr.Accordion("게임 설정 상세 · 플레이어별 모델 / 생성 설정", open=False):
            details = gr.HTML("")
        chat = gr.Chatbot(type="tuples", label="전체 대화", height=680,
                          render_markdown=True, sanitize_html=True,
                          show_copy_button=True)
        selection_outputs = [job, record, info, details, chat]
        demo.load(refresh, [game, job, record], selection_outputs)
        game.input(refresh, [game], selection_outputs)
        job.input(choose_job, [game, job], [record, info, details, chat])
        record.input(show, [game, job, record], [info, details, chat])
        reload_button.click(refresh, [game, job, record], selection_outputs)
    return demo.queue()
