"""Offline audit of saved game I/O; no model, network, or scheduler calls."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def actor_of(call):
    return re.search(r"Your name is ([^.]+)\.", call["request"]["messages"][0]["content"])[1]


def rendered(message, actor):
    return ("assistant" if message["agent_name"] == actor else "user",
            f'[{message["agent_name"]}]: {message["content"]}<EOS>')


def visible(message, actor):
    return message["visible_to"] == "all" or (actor != "Moderator" and actor in message["visible_to"])


def signature(actor, phase, history, instruction):
    return json.dumps([actor, phase, history, instruction], ensure_ascii=False)


def audit(directory):
    messages = read(directory / "messages.json")
    decisions = read(directory / "decisions.json")
    calls = sorted([json.loads(line) for line in (directory / "api_calls.jsonl").read_text().splitlines()], key=lambda c: c["call"])
    errors = []
    expected_decisions = Counter()
    for decision in decisions:
        req = decision["request"]
        expected_decisions[signature(req["player_name"], req["phase"],
                                    [rendered(m, req["player_name"]) for m in req["observation"]],
                                    req["instruction"])] += 1
        if any(not visible(m, req["player_name"]) for m in req["observation"]):
            errors.append({"scope": "decision_visibility", "actor": req["player_name"]})
    actual_decisions = Counter()
    for call in calls:
        actor = actor_of(call)
        api_messages = call["request"]["messages"]
        history = [(m["role"], m["content"]) for m in api_messages[1:-1]]
        expected = [rendered(m, actor) for m in messages if visible(m, actor)]
        if history != expected[:len(history)]:
            errors.append({"call": call["call"], "scope": "visible_history_prefix", "actor": actor})
        if any(m["role"] == "system" for m in api_messages[1:]):
            errors.append({"call": call["call"], "scope": "multiple_system_messages"})
        if decisions:
            actual_decisions[signature(actor, call["phase"], history, api_messages[-1]["content"])] += 1
        else:
            # Ordinary Liar turns must include all visible messages before the action.
            raw = call["response"]["choices"][0]["message"]["content"].strip()
            raw = re.sub(r"^\s*\[.*]:", "", raw).strip()
            raw = re.sub(rf"^\s*{re.escape(actor)}\s*:", "", raw).strip()
            raw = re.sub(r"<EOS>$", "", raw).strip()
            candidates = [i for i, m in enumerate(messages)
                          if m["agent_name"] == actor and m["content"] == raw]
            if not any(history == [rendered(m, actor) for m in messages[:i] if visible(m, actor)] for i in candidates):
                errors.append({"call": call["call"], "scope": "liar_action_history"})
    if decisions and expected_decisions != actual_decisions:
        errors.append({"scope": "api_decision_multiset_mismatch",
                       "missing": sum((expected_decisions - actual_decisions).values()),
                       "extra": sum((actual_decisions - expected_decisions).values())})
    kinds = Counter(d["request"]["kind"] for d in decisions)
    phases = Counter(d["request"]["phase"] + "/" + d["request"]["kind"] for d in decisions)
    responses = [c.get("response", {}).get("choices", [{}])[0] for c in calls]
    speech = [d for d in decisions if d["request"]["kind"] == "speech"]
    moderation = [d for d in decisions if d["request"]["kind"] == "moderation"]
    roles = {}
    for m in messages:
        match = re.search(r"당신의 비밀 역할은 \*\*(.+?)\*\*", m["content"])
        if match:
            roles[m["visible_to"][0]] = match[1]
    actions = [d for d in decisions if d["request"]["kind"] == "game_action"]
    self_votes = [d["request"]["player_name"] for d in actions
                  if d["request"]["phase"] == "DAY_VOTING"
                  and re.search(r"Player\s*\d+", d["response"])[0] == d["request"]["player_name"]]
    result = {
        "summary": read(directory / "summary.json"),
        "api_calls": len(calls), "audit_errors": errors,
        "api_errors": sum("error" in c for c in calls),
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in responses)),
        "thinking_off_requests": sum(c["request"].get("extra_body", {}).get("chat_template_kwargs", {}).get("enable_thinking") is False for c in calls),
        "reasoning_or_think_outputs": sum(bool(r.get("message", {}).get("reasoning")) or "<think>" in (r.get("message", {}).get("content") or "") for r in responses),
        "prompt_tokens_min_max": [min(c["response"]["usage"]["prompt_tokens"] for c in calls), max(c["response"]["usage"]["prompt_tokens"] for c in calls)],
        "kind_counts": dict(kinds), "phase_kind_counts": dict(phases),
        "invalid_decisions": sum(not d["valid"] for d in decisions),
        "stale_decisions": sum(d["stale"] for d in decisions),
        "format_eligible_decisions": sum(d.get("format_eligible", d.get("trainable", False)) for d in decisions),
        "expert_approved_decisions": sum(d.get("expert_approved", False) for d in decisions),
        "intent_scores": dict(Counter(str(d["score"]) for d in decisions if d["request"]["kind"] == "intent")),
        "moderator_responses": dict(Counter(d["response"] for d in moderation)),
        "speech_responses": len(speech), "exact_pass_responses": sum(d["response"] == "PASS" for d in speech),
        "roles": roles, "self_votes": self_votes,
        "public_speech_by_player": dict(Counter(d["request"]["player_name"] for d in speech if d["request"]["phase"] == "DAY_DISCUSSION" and d["response"] != "PASS")),
        "speech_exact_duplicate_excess": sum(n - 1 for n in Counter(d["response"] for d in speech if d["response"] != "PASS").values()),
        "message_timestamp_unique": len({m["timestamp"] for m in messages}),
    }
    if directory.parent.name == "liar_game":
        secret = next(m for m in messages if m["content"].startswith("You are a non-liar."))
        word = re.search(r"The secret word is (.*?)\.", secret["content"])[1]
        liar = next(m["visible_to"][0] for m in messages if m["content"].startswith("You are the Liar."))
        votes = [m for m in messages if m["msg_type"] == "vote"]
        result["liar"] = {"word": word, "player": liar, "first_turn_position": int(liar.split()[-1]),
                          "strict_vote_format": sum(bool(re.fullmatch(r"I vote for Player [1-5]", m["content"])) and m["agent_name"] != m["content"][len("I vote for "):] for m in votes),
                          "non_liar_correct_votes": sum(m["agent_name"] != liar and m["content"] == "I vote for " + liar for m in votes),
                          "explicit_word_clues": [{"turn": m["turn"], "actor": m["agent_name"], "content": m["content"]} for m in messages if m["agent_name"].startswith("Player") and m["turn"] <= 10 and re.search(r"\b" + re.escape(word) + r"\b", m["content"], re.I)]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = {str(p.relative_to(args.input)): audit(p) for p in sorted(args.input.glob("*/seed_*")) if (p / "summary.json").exists()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"games": len(data), "calls": sum(v["api_calls"] for v in data.values()), "audit_errors": {k: v["audit_errors"] for k, v in data.items() if v["audit_errors"]}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
