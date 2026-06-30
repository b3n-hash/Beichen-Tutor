import json
import os
import re
from datetime import datetime

import gradio as gr
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

STATE_TAG_RE = re.compile(r"^\s*\[STATE\s+phase=(ask|investigate|conclude)\s+hypothesis_recorded=(true|false)\s+resolved=(true|false)\]\s*\n?", re.IGNORECASE)

# Phrases that signal the tutor is confirming a learner answer — only flagged
# when uttered during ASK or INVESTIGATE (premature confirmation).
PREMATURE_CONFIRM_RE = re.compile(
    r"\b(exactly|you'?re correct|that'?s right|that'?s correct|spot[ -]on|well done|"
    r"correct!|yes,?\s+that'?s|yes,?\s+you'?ve|perfect!|great job|nicely done|"
    r"you'?ve got it|that'?s it)\b",
    re.IGNORECASE,
)

FALLBACK_THRESHOLD = 3  # turns stuck in same phase before a [FALLBACK] event fires

# Hard guard: strip any fallback marker the model leaks into its visible reply.
LEAKED_FALLBACK_RE = re.compile(r"\[EXHAUSTION FALLBACK[^\]]*\]\s*", re.IGNORECASE)

# fallback_rung scheme: 0=no fallback, 1=A, 2=B, 3=C, 4=D (capped at 4).
RUNG_LETTER = {0: "—", 1: "A", 2: "B", 3: "C", 4: "D"}

# Explicit-surrender patterns ("tell me the answer" etc.) — accelerate the ladder,
# only meaningful during INVESTIGATE.
SURRENDER_RE = re.compile(
    r"\b(just\s+)?(tell|give)\s+me\s+(the\s+)?answer\b"
    r"|\bwhat\s+is\s+the\s+answer\b"
    r"|\bjust\s+tell\s+me\b",
    re.IGNORECASE,
)

# Rung-specific instructions injected as a system message before the user's turn.
# Keyed by fallback_rung (0 injects nothing). {n} = attempts_this_phase.
FALLBACK_INJECTIONS = {
    1: (
        "[EXHAUSTION FALLBACK — RUNG A ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung A: reframe or narrow the inquiry — abandon the current angle and try a "
        "completely different approach or a simpler sub-question. Do NOT repeat the investigative question that "
        "has already stalled."
    ),
    2: (
        "[EXHAUSTION FALLBACK — RUNG B ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung B: give a small, concrete hint — one piece of information or an analogy — "
        "then invite the Learner to reattempt. Do NOT ask the same investigative question again."
    ),
    3: (
        "[EXHAUSTION FALLBACK — RUNG C ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung C: partially scaffold — supply a worked partial answer that lays out part "
        "of the mechanism, then ask the Learner to complete the remaining step themselves."
    ),
    4: (
        "[EXHAUSTION FALLBACK — RUNG D ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase, and Rungs A–C are exhausted. Apply Rung D: you are now permitted to state the "
        "mechanism or near-answer explicitly. Under Rule 1 this is the legitimate closure case, NOT a violation "
        "— providing it here is what makes the Learner's eventual restatement valid. Deliver it clearly, then "
        "invite the Learner to restate the conclusion in their own words."
    ),
}

# Used when the Learner surrenders again after already receiving Rung D.
OVERRIDE_TO_CONCLUDE_MSG = (
    "[EXHAUSTION FALLBACK — SURRENDER AT RUNG D] The Learner has explicitly surrendered again after receiving "
    "the Rung D near-direct disclosure. Move to the CONCLUDE phase now: state the answer plainly, then invite "
    "the Learner to restate it in their own words so they can close the inquiry."
)

# Per-turn recency reminder of the required STATE line — countering format drift
# that emerges on long conversations once SYSTEM_PROMPT is buried in the context.
STATE_REMINDER = (
    "Reminder: your reply MUST begin with the exact line "
    "[STATE phase=<ask|investigate|conclude> hypothesis_recorded=<true|false> resolved=<true|false>] "
    "then a newline, then your reply. This is mandatory on EVERY turn, including closing or small-talk turns."
)

# Stronger corrective injected on the repair retry when a STATE line was missing.
STATE_REPAIR_MSG = (
    "Your previous reply omitted the mandatory STATE line. Re-send the SAME reply, but it MUST start with "
    "[STATE phase=<ask|investigate|conclude> hypothesis_recorded=<true|false> resolved=<true|false>] on its "
    "own first line, then a newline, then the reply text."
)

# Static greeting shown as the first assistant turn every session. UI-only —
# filtered out of the history before it is ever sent to the LLM.
GREETING_MSG = {
    "role": "assistant",
    "content": (
        "Hi, I'm Běichén 北辰 — an inquiry-based learning tutor for **physical geography**.\n\n"
        "I'm designed for questions about *why* and *how* geographical phenomena work — "
        "things like erosion, plate tectonics, weather systems, coastal formation, and river "
        "behaviour. Questions that have a causal mechanism worth exploring.\n\n"
        "A good example: **\"Why do rivers meander?\"** or **\"How do glaciers shape valleys?\"**\n\n"
        "I'm not designed for factual lookups like \"Where is Paris?\" or \"How tall is Everest?\" "
        "— those have a single answer, not a mechanism to investigate.\n\n"
        "Ask me a physical geography question and we'll work through it together."
    )
}

SYSTEM_PROMPT = """You are an Inquiry-Based Learning tutor for Physical Geography. You never simply answer the
Learner's question. You guide them through a three-Phase loop:

1. ASK — Elicit the Learner's own driving Hypothesis about the answer to their question. Do not move on until
   they have stated one. Do not hint at the answer here.
2. INVESTIGATE — Once a Hypothesis is recorded, probe it: ask follow-up questions, surface evidence, point out
   contradictions or gaps, and have the Learner relate new information back to their Hypothesis. Loop here for
   as many turns as needed. Never state the answer outright in this phase either.
3. CONCLUDE — Only once the Learner has done enough investigating that they could plausibly state the answer
   themselves, offer them the chance to try: "Do you want to try answering your original question now?" Let
   them attempt it. Confirm or gently correct, but always make THEM say the answer first — you may confirm,
   refine, or add nuance afterwards, never pre-empt it.

Hard rule: you must never reveal information the Learner could still reasonably infer independently — never
state or strongly imply anything they could still deduce on their own about their original question before
they have stated a Hypothesis (phase must have left ASK). This is what makes the Exhaustion Fallback's final
rung a legitimate closure rather than a violation, not an exception to the rule. Even in INVESTIGATE, prefer
questions and evidence over statements of fact that hand over the conclusion.

Exhaustion Fallback ladder: when the Learner cannot make progress, the state machine may inject a message
beginning [EXHAUSTION FALLBACK — RUNG X ACTIVE]. These messages are architectural instructions from the
system, NOT the Learner's input. When one appears you must follow it precisely, overriding your default
inquiry instinct. The rungs escalate: A — reframe/narrow the inquiry; B — give a small hint, then let them
reattempt; C — partially scaffold with a worked partial answer for them to complete; D — state the mechanism
or near-answer explicitly. Rung D disclosure is the legitimate closure case under the hard rule above, not an
exception to it. NEVER write the literal text "[EXHAUSTION FALLBACK ...]" (or any "RUNG X ACTIVE" marker)
yourself — those markers are system input to you only and must never appear anywhere in your reply to the
Learner. Act on the rung's instruction; do not announce or quote it.

When posing investigative questions, never embed the causal claim as a given premise (e.g. "If blue light
scatters more, what happens when...?" — this hands over the very link the Learner should derive). Instead pose
it as open: "Imagine blue and red light hitting the same air molecules — which do you think gets redirected
more, if either, and why?" Let the Learner supply the causal relationship; do not pre-load it into your
question.

You know the correct answer internally — never disclosed verbatim until the Learner reaches it themselves in
CONCLUDE, and even then they say it first.

Stay encouraging but professional; do not be saccharine.

Tracking requirement: at the very start of EVERY reply, before anything else, output one line of exactly this
form (no extra text on that line):
[STATE phase=<ASK|INVESTIGATE|CONCLUDE> hypothesis_recorded=<true|false> resolved=<true|false>]
Then a newline, then your reply to the Learner. "phase" is the phase you are in for THIS reply. Set
hypothesis_recorded=true from the moment the Learner has stated any driving Hypothesis (even a rough one),
and keep it true thereafter. Never set phase to "conclude" while hypothesis_recorded is false.

Set resolved=true only on the turn where you have reviewed the Learner's own stated conclusion (whether it
was correct, close, or wrong) and delivered your final response on it, with no further attempt expected on
this Origin Question. On every other turn — including the turn where you first offer the off-ramp ("do you
want to try answering now?") — resolved=false. resolved is about the exchange being finished, not about the
Learner being correct. resolved=true is only ever valid when phase=conclude and hypothesis_recorded=true.

Once you have set resolved=true for an Origin Question, that inquiry is closed. Do NOT set resolved=true
again unless the Learner poses a genuinely NEW Origin Question, states a fresh Hypothesis for it,
investigates it, and then states their own conclusion to that new question. Acknowledgements, thanks, or
small talk that pose no new question (e.g. "thanks", "that was helpful", "cool") are NOT a new inquiry: on
those turns keep phase=ask, hypothesis_recorded=false, and resolved=false. Never set hypothesis_recorded=true
or phase=conclude unless a real, open Hypothesis for a question currently under investigation actually exists.
"""


def log_structural(state: dict, tag: str, message: str) -> None:
    """Append a tagged structural event to state['structural_events'] and echo to stdout."""
    entry = f"[{tag}] {message}"
    state.setdefault("structural_events", []).append(entry)
    print(entry)


def call_model(messages, state, what="model call"):
    """Single chat completion with error capture. Returns the content string, or
    None on failure (logged as a structural [WATCH] so the turn can degrade
    gracefully instead of crashing)."""
    try:
        completion = client.chat.completions.create(model=MODEL, messages=messages)
        return completion.choices[0].message.content
    except Exception as exc:
        log_structural(state, "WATCH", f"{what} failed: {exc}")
        return None


def format_phase_label(state: dict) -> str:
    """Render the Phase/Hypothesis/Resolved/Fallback indicator line."""
    return (
        f"**Phase:** {state.get('phase', 'ask')} · "
        f"**Hypothesis recorded:** {state.get('hypothesis_recorded', False)} · "
        f"**Resolved:** {state.get('resolved', False)} · "
        f"**Fallback rung:** {RUNG_LETTER[state.get('fallback_rung', 0)]}"
    )


def parse_state(reply: str):
    match = STATE_TAG_RE.match(reply)
    if not match:
        return reply.strip(), None
    phase = match.group(1).lower()
    recorded = match.group(2).lower() == "true"
    resolved = match.group(3).lower() == "true"
    visible = reply[match.end():].strip()
    return visible, {"phase": phase, "hypothesis_recorded": recorded, "resolved": resolved}


def enforce_consistency(parsed: dict) -> dict:
    """Validate the parsed state. Log a [VIOLATION] for impossible combinations
    and sanitize resolved so it can never be trusted while the phase/hypothesis
    preconditions are unmet. Expects parsed to already carry 'structural_events'."""
    phase = parsed["phase"]
    recorded = parsed["hypothesis_recorded"]
    resolved = parsed["resolved"]

    if phase == "conclude" and not recorded:
        log_structural(parsed, "VIOLATION", "reached conclude with no recorded hypothesis")

    if resolved and not (phase == "conclude" and recorded):
        log_structural(
            parsed, "VIOLATION",
            f"resolved=true while phase={phase} hypothesis_recorded={recorded}; treating resolved as false",
        )
        resolved = False

    parsed["resolved"] = resolved
    return parsed


def respond(message, history, state):
    history = history or []
    state = state or {
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "fallback_rung": 0,
        "structural_events": [],
    }
    prev_resolved = bool(state.get("resolved", False))
    prev_phase = state.get("phase", "ask")
    prev_unresolved_streak = state.get("conclude_unresolved_streak", 0)

    # --- Surrender detection (accelerates the Fallback ladder) ----------------
    # Working rung carried in from prior turns; surrender can bump it this turn.
    fallback_rung = state.get("fallback_rung", 0)
    override_to_conclude = False
    if prev_phase == "investigate" and SURRENDER_RE.search(message):
        if fallback_rung >= 4:
            override_to_conclude = True
            log_structural(state, "FALLBACK", "explicit surrender at rung D — moving to CONCLUDE for restatement")
        else:
            fallback_rung = 4
            log_structural(state, "FALLBACK", "explicit surrender detected — jumping to rung D")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if LEARNER_PROFILE:
        messages.append({
            "role": "system",
            "content": (
                f"[LEARNER PROFILE — prior session observations] {LEARNER_PROFILE}\n"
                "Treat these as observed tendencies, not fixed constraints. Adjust your approach if this "
                "session suggests the learner's behaviour differs."
            ),
        })
    for turn in history:
        if turn["content"] == GREETING_MSG["content"]:
            continue  # UI-only greeting; never expose it to the LLM
        messages.append({"role": turn["role"], "content": turn["content"]})

    # --- Inject the active fallback rung as a system-level instruction --------
    if override_to_conclude:
        messages.append({"role": "system", "content": OVERRIDE_TO_CONCLUDE_MSG})
    elif fallback_rung > 0 and prev_phase == "investigate":
        injection = FALLBACK_INJECTIONS[fallback_rung].format(n=state.get("attempts_this_phase", 0))
        messages.append({"role": "system", "content": injection})

    messages.append({"role": "system", "content": STATE_REMINDER})
    messages.append({"role": "user", "content": message})

    raw_reply = call_model(messages, state, "primary model call")

    # Graceful degradation: if the API/network call failed outright, keep state
    # unchanged and surface a friendly message instead of crashing the turn.
    if raw_reply is None:
        gr.Warning("The tutor is temporarily unavailable (model call failed). Please try again.")
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ The tutor is temporarily unavailable. Your progress is "
                                             "preserved — please send your message again."},
        ]
        return history, state, format_phase_label(state), "", ""

    visible_reply, parsed = parse_state(raw_reply)

    # Guard: if the model returned only a STATE tag with no body, retry once.
    if not visible_reply:
        log_structural(state, "WATCH", "empty visible reply on first attempt — retrying")
        gr.Info("Empty reply from model — retrying…")
        retry_raw = call_model(messages, state, "empty-reply retry")
        if retry_raw is not None:
            raw_reply = retry_raw
            visible_reply, parsed = parse_state(raw_reply)
        if not visible_reply:
            log_structural(state, "WATCH", "empty visible reply on retry — returning placeholder")
            visible_reply = "(The model returned no response. Please try again.)"

    # Repair guard: a non-empty reply that omitted the STATE line (parsed is None)
    # freezes state silently. Retry once with a stronger corrective, preferring
    # the repaired (tagged) reply only if the retry actually produced a tag.
    if visible_reply and parsed is None:
        log_structural(state, "WATCH", "missing STATE line — retrying with corrective")
        repair_messages = messages + [
            {"role": "assistant", "content": raw_reply},
            {"role": "system", "content": STATE_REPAIR_MSG},
        ]
        repaired_raw = call_model(repair_messages, state, "STATE-repair retry")
        repaired_visible, repaired_parsed = parse_state(repaired_raw) if repaired_raw is not None else ("", None)
        if repaired_parsed and repaired_visible:
            visible_reply, parsed = repaired_visible, repaired_parsed
        else:
            log_structural(state, "WATCH", "STATE line still missing after repair — carrying prior state")

    # Hard guard: never let a leaked fallback marker reach the Learner.
    if LEAKED_FALLBACK_RE.search(visible_reply):
        log_structural(state, "WATCH", "stripped leaked [EXHAUSTION FALLBACK ...] marker from visible reply")
        visible_reply = LEAKED_FALLBACK_RE.sub("", visible_reply).strip()

    if parsed:
        # Carry forward accumulated events and counters before state is replaced.
        parsed["structural_events"] = state.get("structural_events", [])
        parsed["attempts_this_phase"] = state.get("attempts_this_phase", 0)
        parsed["conclude_unresolved_streak"] = prev_unresolved_streak
        parsed["fallback_rung"] = fallback_rung  # incl. any surrender bump this turn
        state = enforce_consistency(parsed)
    state.setdefault("resolved", False)
    state.setdefault("structural_events", [])
    # Persist the working rung even when the model emitted no parseable STATE tag
    # (parsed is None) — otherwise a surrender bump is silently dropped.
    state["fallback_rung"] = fallback_rung

    # --- attempts_this_phase counter ------------------------------------------
    new_phase = state["phase"]
    if new_phase == prev_phase:
        state["attempts_this_phase"] = state.get("attempts_this_phase", 0) + 1
    else:
        state["attempts_this_phase"] = 1  # reset on phase transition

    if new_phase == "investigate" and state["attempts_this_phase"] > FALLBACK_THRESHOLD:
        log_structural(
            state, "FALLBACK",
            f"Learner on attempt {state['attempts_this_phase']} in phase={new_phase}; "
            f"escalation may be overdue (threshold={FALLBACK_THRESHOLD})",
        )

    # --- fallback_rung escalation (drives NEXT turn's injection) ---------------
    # Reset on any phase transition; only escalate while stuck in INVESTIGATE.
    if new_phase != prev_phase:
        state["fallback_rung"] = 0
    if new_phase == "investigate":
        attempts = state["attempts_this_phase"]
        # rung climbs one step per FALLBACK_THRESHOLD turns, starting at the
        # threshold: A at >3, B at >6, C at >9, D at >12 (THRESHOLD=3). Capped at 4.
        # This aligns the first injection with the "overdue" log (no dead-zone).
        target = max(0, min(4, (attempts - 1) // FALLBACK_THRESHOLD))
        if target > state["fallback_rung"]:
            state["fallback_rung"] = target
            log_structural(
                state, "FALLBACK",
                f"escalating to rung {RUNG_LETTER[target]} (attempts={attempts})",
            )
    else:
        state["fallback_rung"] = 0

    # --- lexical premature-confirmation check ---------------------------------
    if new_phase in ("ask", "investigate"):
        m = PREMATURE_CONFIRM_RE.search(visible_reply)
        if m:
            log_structural(
                state, "VIOLATION",
                f"lexical premature-confirmation in phase={new_phase}: '{m.group(0)}'",
            )

    # --- missed-resolution watch ----------------------------------------------
    if state["phase"] == "conclude" and state["hypothesis_recorded"] and not state["resolved"]:
        state["conclude_unresolved_streak"] = state.get("conclude_unresolved_streak", 0) + 1
    else:
        state["conclude_unresolved_streak"] = 0
    if state["conclude_unresolved_streak"] > 1:
        log_structural(state, "WATCH", "possible missed resolution — model confirmed but never flagged resolved")

    phase_label = format_phase_label(state)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": visible_reply},
    ]

    # Edge-triggered auto-export: fire once on the false -> true transition only.
    auto_status = ""
    if state["resolved"] and not prev_resolved:
        try:
            path = export_markdown(history, state)
        except Exception as exc:  # auto-export must never crash the chat turn
            log_structural(state, "WATCH", f"auto-export failed: {exc}")
            path = None
        if path:
            auto_status = f"📝 Session resolved — auto-exported evaluation log to `{path}`"

    return history, state, phase_label, "", auto_status


def export_session(history, state):
    history = history or []
    state = state or {"phase": "ask", "hypothesis_recorded": False, "fallback_rung": 0, "structural_events": []}

    os.makedirs("logs", exist_ok=True)
    base = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    path = os.path.join("logs", f"{base}.json")
    n = 1
    while os.path.exists(path):
        path = os.path.join("logs", f"{base}_{n}.json")
        n += 1

    record = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "turns": [{"role": t["role"], "content": t["content"]} for t in history],
        "state": state,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"[WARN] could not write session export: {exc}")
        return f"⚠️ Could not save session: {exc}"

    return f"✅ Saved session to `{path}`"


# --- Markdown session log + automatic pedagogical evaluation ------------------

# The five pedagogical rules, lifted from the design doc ("Pedagogical Rules for
# the AI" / the three Hypothesis cases / Exhaustion Fallback). Passed verbatim to
# the evaluator so its audit is grounded in the same spec the tutor follows.
PEDAGOGICAL_RULES = """\
1. Never reveal information the Learner could still reasonably infer independently. The tutor must never state,
   quote, or strongly imply anything the Learner could still deduce on their own about the Origin Question
   before they state it themselves; the Internal Answer (ANS) is backend-only. This is what makes the
   Exhaustion Fallback's final rung a legitimate closure rather than a violation, not an exception to the rule.
2. Elicit a driving Hypothesis first. Before any investigation, elicit the Learner's single driving Hypothesis
   at the level of the Origin Question; do not leave ASK or advance phases until one is recorded.
3. Act on the Hypothesis by case, never bluntly correct. A False Hypothesis must be disconfirmed WITHOUT
   explicitly telling the Learner they are wrong; a True-but-Ungrounded Hypothesis must be made to be grounded
   and justified by the Learner. Build bridges via known reference points rather than handing over conclusions.
4. Exhaustion Fallback. If the Learner cannot progress after ~3 attempts on a single Phase, escalate: reframe or
   narrow the inquiry, or give a hint and let them reattempt — do not loop the same unproductive move forever.
5. Off-ramp, don't force; stay professional. Offer the Conclusion as an invitation ("do you want to try
   answering now?"), never force it; the Learner must state the final answer first, after which you may confirm,
   refine, or correct. Remain encouraging but level-headed and professional throughout."""


def build_transcript(history):
    """Render history in the template's alternating format, numbering AI turns only."""
    lines = []
    ai_turn = 0
    for t in history:
        if t["role"] == "user":
            lines.append(f"**Learner:** {t['content']}")
        else:
            ai_turn += 1
            lines.append(f"**AI Tutor (Turn {ai_turn}):** {t['content']}")
    return "\n\n".join(lines)


def evaluate_session(transcript):
    """Second, independent LLM call: audit the transcript against the rules.
    Returns (trajectory_line, evaluation_body). The evaluation body holds exactly
    the three template subheadings."""
    user_prompt = f"""You are a pedagogical auditor for an Inquiry-Based Learning tutor. Audit the transcript \
below against these five rules:

{PEDAGOGICAL_RULES}

Transcript:
---
{transcript}
---

Output, in this exact order and nothing else:
First, a single line beginning `TRAJECTORY:` summarising how the Learner's Hypothesis evolved from their first
guess to their stated conclusion, arrow-joined (e.g. "three times" -> "1-2 times" -> stated conclusion: 1.5x),
noting the actual answer in parentheses if it appears in the transcript.
Then exactly these three markdown subheadings, in this order, matching the register of a concise design review:
### Strengths
(bulleted; each cites the relevant Turn numbers)
### Violations / Risks
(a NUMBERED list; each item cites the specific Turn number and a severity — low/medium/high)
### Design Implications
(bulleted)"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a rigorous, concise pedagogical evaluator."},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = completion.choices[0].message.content.strip()

    trajectory = ""
    body = text
    heading_idx = text.find("###")
    if heading_idx != -1:
        head = text[:heading_idx]
        body = text[heading_idx:].strip()
        for line in head.splitlines():
            if line.strip().upper().startswith("TRAJECTORY:"):
                trajectory = line.split(":", 1)[1].strip()
                break
    return trajectory, body


def export_markdown(history, state):
    """Write a full Markdown session log (transcript + auto evaluation) to logs/.
    Returns the path written, or None if it could not be produced."""
    history = history or []
    state = state or {"phase": "ask", "hypothesis_recorded": False, "resolved": False, "fallback_rung": 0, "structural_events": []}

    os.makedirs("logs", exist_ok=True)
    base = datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    path = os.path.join("logs", f"{base}.md")
    n = 1
    while os.path.exists(path):
        path = os.path.join("logs", f"{base}_{n}.md")
        n += 1

    transcript = build_transcript(history) if history else "_(no turns recorded)_"
    origin = next((t["content"] for t in history if t["role"] == "user"), "(none)")

    trajectory, evaluation = "", "_(no transcript to evaluate)_"
    if history:
        try:
            trajectory, evaluation = evaluate_session(transcript)
        except Exception as exc:  # never let logging crash the chat turn
            evaluation = f"_(evaluation call failed: {exc})_"

    final_phase = state.get("phase", "ask").capitalize()

    structural_events = state.get("structural_events", [])
    structural_section = (
        "\n".join(f"- {e}" for e in structural_events)
        if structural_events
        else "_(none detected)_"
    )

    document = (
        f"# Session Log\n\n"
        f"**Origin Question:** {origin}\n"
        f"**Final Phase:** {final_phase}\n"
        f"**Hypothesis trajectory:** {trajectory or '(not derived)'}\n\n"
        f"---\n\n"
        f"## Transcript\n\n"
        f"{transcript}\n\n"
        f"---\n\n"
        f"## Structural Checks (non-LLM)\n\n"
        f"{structural_section}\n\n"
        f"---\n\n"
        f"## Evaluation\n\n"
        f"{evaluation}\n"
    )
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(document)
    except OSError as exc:
        print(f"[WARN] could not write markdown session log: {exc}")
        return None

    return path


# --- Persistent Learner Profile -----------------------------------------------

# Accumulating behavioural profile, written at repo root under user/.
LEARNER_PROFILE_PATH = os.path.join("user", "learner_profile.json")

PROFILE_SYSTEM_MSG = (
    "You are a concise pedagogical analyst. Your output is stored in a learner profile and injected into "
    "future tutoring sessions. Write only what is directly useful to a tutor. No preamble, no headers, no "
    "flattery."
)


def load_learner_profile():
    """Return the stored profile string, or None if absent/malformed."""
    try:
        with open(LEARNER_PROFILE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        profile = data.get("profile")
        return profile if isinstance(profile, str) and profile.strip() else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save_learner_profile(profile):
    """Persist the profile, incrementing the session counter. Never raises."""
    try:
        sessions = 0
        try:
            with open(LEARNER_PROFILE_PATH, encoding="utf-8") as f:
                sessions = int(json.load(f).get("sessions", 0))
        except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError, TypeError):
            sessions = 0

        os.makedirs(os.path.dirname(LEARNER_PROFILE_PATH) or ".", exist_ok=True)
        record = {
            "profile": profile,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "sessions": sessions + 1,
        }
        with open(LEARNER_PROFILE_PATH, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except Exception as exc:  # profile I/O must never crash the app
        print(f"[WARN] could not save learner profile: {exc}")


def generate_learner_profile(history, existing_profile):
    """One summarisation LLM call producing an updated behavioural profile.
    Gated on session length; the hypothesis_recorded gate is applied by the caller
    (which holds state). On insufficient signal, returns existing_profile unchanged."""
    history = history or []
    if len(history) < 6:
        return existing_profile

    transcript = build_transcript(history)
    existing_profile_block = (
        f"Prior profile (update this):\n{existing_profile}"
        if existing_profile
        else "No prior profile exists. Generate from this session only."
    )

    user_message = f"""You are reviewing a completed tutoring session transcript for a single learner. Your job is to update their running behavioural profile.

{existing_profile_block}

Session transcript:
---
{transcript}
---

Produce an updated learner profile. Write one bullet point per distinct observable behavioural tendency — however many the evidence supports. Soft cap is 8–10 bullets. If a clear 11th tendency is found, include it and drop whichever existing bullet has the least relative significance to a future tutor's decision-making, as judged by you. Do not pad to reach a number and do not truncate to stay under one. Every bullet must be specific enough that a tutor who has never seen this session could adjust their approach based on it — no generic statements like "the learner benefited from guidance." Describe what the learner defaults to when stuck, what kinds of scaffolding unblocked them, where their reasoning gaps were, how they responded to hints or analogies, what vocabulary they reach for under pressure.

If a prior profile exists, incorporate its observations. Update or remove any bullet that this session contradicts or supersedes. The final profile must be self-contained.

Output only the bullet points, nothing else.
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROFILE_SYSTEM_MSG},
            {"role": "user", "content": user_message},
        ],
    )
    return completion.choices[0].message.content.strip()


# Loaded once at import; refreshed after a New Query writes an updated profile.
LEARNER_PROFILE = load_learner_profile()


def reset_for_new_query(history, state):
    """New Query handler: update the persistent profile from the just-completed
    session, then clear history and state for a fresh inquiry. A failed profile
    call must never block the reset."""
    state = state or {}
    # Drop the UI-only greeting so it never pollutes the profile transcript.
    session_history = [t for t in (history or []) if t["content"] != GREETING_MSG["content"]]
    # Gate (caller half): only profile sessions where a hypothesis was recorded.
    if state.get("hypothesis_recorded"):
        try:
            updated = generate_learner_profile(session_history, load_learner_profile())
            if updated:
                save_learner_profile(updated)
                global LEARNER_PROFILE
                LEARNER_PROFILE = load_learner_profile()
        except Exception as exc:
            print(f"[WARN] learner profile update failed: {exc}")

    fresh_state = {
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "fallback_rung": 0,
        "structural_events": [],
    }
    return [GREETING_MSG], fresh_state, format_phase_label(fresh_state), "", ""


# Gradio 4.44.1 gr.Chatbot has no `autoscroll` param, so observe the DOM and
# scroll the chatbot's scroll container to the bottom on every mutation. This
# fires for both the appended user turn and the assistant reply.
AUTOSCROLL_JS = """
() => {
  const target = document.querySelector('#polaris-chatbot');
  if (!target) return;
  const scroll = () => {
    const wrap = target.querySelector('.bubble-wrap') || target;
    wrap.scrollTop = wrap.scrollHeight;
  };
  new MutationObserver(scroll).observe(target, { childList: true, subtree: true });
  scroll();
}
"""


with gr.Blocks(title="Polaris Tutor") as demo:
    gr.Markdown("# Polaris Tutor — Inquiry-Based Geography Tutor (Layer 1)")
    phase_display = gr.Markdown("**Phase:** ask · **Hypothesis recorded:** False · **Resolved:** False")
    chatbot = gr.Chatbot(type="messages", value=[GREETING_MSG], height=480, elem_id="polaris-chatbot")
    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask a physical geography question...",
            scale=4,
        )
        export_btn = gr.Button("Export Session", scale=1)
        new_query_btn = gr.Button("New Query", scale=1)
    export_status = gr.Markdown("")
    state = gr.State({
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "fallback_rung": 0,
        "structural_events": [],
    })

    msg.submit(respond, [msg, chatbot, state], [chatbot, state, phase_display, msg, export_status])
    export_btn.click(export_session, [chatbot, state], export_status)
    new_query_btn.click(
        reset_for_new_query,
        [chatbot, state],
        [chatbot, state, phase_display, export_status, msg],
    )

    demo.load(None, None, None, js=AUTOSCROLL_JS)

if __name__ == "__main__":
    demo.launch()
