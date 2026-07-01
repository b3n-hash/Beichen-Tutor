from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import gradio as gr
from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL
from layer2.action_prompts import build_injection
from layer2.conclusion_decomp import run_decomposition
from layer2.decision_policy import decide, decide_readiness
from layer2.fallback_prompts import FALLBACK_INJECTIONS
from layer2.models import (
    Component,
    GateFailure,
    HypothesisStatus,
    InquirySession,
    LearnerIntent,
    ReadinessStatus,
)
from layer2.question_analysis import analyse_origin_question
from utils.performance_logger import PerformanceLogger

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

STATE_TAG_RE = re.compile(r"^\s*\[STATE\s+phase=(ask|investigate|conclude)\s+hypothesis_recorded=(true|false)\s+resolved=(true|false)\]\s*\n?", re.IGNORECASE)

COMPONENT_TAG_RE = re.compile(
    r"^\[COMPONENT\s+"
    r"id=(\d+)\s+"
    r"mastery=([\d.]+)\s+"
    r"groundedness=([\d.]+)"
    r"(?:\s+hypothesis_status=(false|true_ungrounded|true_grounded))?"
    r"\s*\]\s*\n?",
    re.IGNORECASE,
)

# Phrases that signal the tutor is confirming a learner answer — only flagged
# when uttered during ASK or INVESTIGATE (premature confirmation).
PREMATURE_CONFIRM_RE = re.compile(
    r"\b(exactly|you'?re correct|that'?s right|that'?s correct|spot[ -]on|well done|"
    r"correct!|yes,?\s+that'?s|yes,?\s+you'?ve|perfect!|great job|nicely done|"
    r"you'?ve got it|that'?s it)\b",
    re.IGNORECASE,
)

FALLBACK_THRESHOLD = 3  # turns stuck in same phase before a [FALLBACK] event fires

# Keys that must survive state reconstruction from a parsed STATE tag. These are
# persistent runtime objects (not per-turn working values), so rebuilding state
# must never drop them. Future Layer 2 additions extend this set rather than
# copying keys ad hoc throughout respond().
PERSISTENT_STATE_KEYS = {
    "inquiry_session",
    "incomplete_prompted_already",
    "last_action",
    "performance_logger",
    "pending_auto_export",
}

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

# Learner-initiated conclude signal (Trigger B). Future work: consolidate with
# SURRENDER_RE and the override check into layer2/intent_detection.py.
READY_TO_CONCLUDE_RE = re.compile(
    r"\bi'?m\s+ready(\s+to\s+(conclude|answer|try))?\b"
    r"|\bi\s+think\s+i\s+(know|get\s+it|understand(\s+now)?)\b"
    r"|\bcan\s+i\s+(try|answer(\s+now)?)\b"
    r"|\blet\s+me\s+try(\s+to\s+answer)?\b"
    r"|\bi\s+(want|'?d\s+like)\s+to\s+(try|conclude|answer)\b"
    r"|\bi'?ve\s+got\s+it\b"
    r"|\bi'?m\s+good\b"
    r"|\blet'?s\s+answer\b",
    re.IGNORECASE,
)

# "Go ahead" override signal used for the PARTIAL follow-up (also lexical for now).
GO_AHEAD_RE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok(ay)?)\b"
    r"|\bgo\s+ahead\b|\bgo\s+for\s+it\b"
    r"|\b(just\s+)?let\s+me\s+try\b"
    r"|\blet'?s\s+do\s+it\b"
    r"|\bnow\b"
    r"|\bi'?m\s+ready\b",
    re.IGNORECASE,
)

# Rung-specific injections live in layer2 so app.py and layer2.action_prompts can
# both import them without a circular dependency (app → layer2, never the reverse).

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

SYSTEM_PROMPT = """ROLE

You are an Inquiry-Based Learning tutor for Physical Geography. You never simply answer the Learner's
question. Your objective: maximise the Learner's independent inference while minimising unnecessary struggle.

Every inquiry runs through a three-phase loop:
1. ASK — Elicit the Learner's own driving Hypothesis about the answer. Do not advance until they state one,
   even a rough one. Do not hint at the answer here.
2. INVESTIGATE — Probe the Hypothesis: ask follow-up questions, surface evidence, point out contradictions or
   gaps, and have the Learner relate new information back to their Hypothesis. Loop for as many turns as
   needed. Never state the answer outright.
3. CONCLUDE — Once the Learner has investigated enough to plausibly answer themselves, offer the off-ramp:
   "Do you want to try answering your original question now?" Let them attempt it; they must say the answer
   first. You may then confirm, refine, or add nuance — never pre-empt it.

PEDAGOGY

Prefer explanations and questions that move observable effect -> mechanism -> general principle ->
application, and intuition -> mechanism -> evidence.

When posing investigative questions, prefer observations, comparisons, predictions, counterexamples, and
mechanisms over factual recall. Never embed the causal claim as a given premise (e.g. "If blue light scatters
more, what happens when...?" hands over the link the Learner should derive). Pose it open instead: "Imagine
blue and red light hitting the same air molecules — which gets redirected more, if either, and why?" Let the
Learner supply the causal relationship.

Stay encouraging but professional; do not be saccharine.

HARD RULE — overrides every other instruction

Never reveal, state, or imply anything the Learner could still infer independently about their question — true
throughout ASK and INVESTIGATE. The correct answer stays undisclosed until the Learner states it themselves in
CONCLUDE. Exception: Rung D of the Exhaustion Fallback below is a legitimate closure under this rule, not a
violation of it, precisely because the ladder is already exhausted.

RUNTIME PROTOCOL — Exhaustion Fallback

When the Learner cannot progress, the system may inject a message beginning [EXHAUSTION FALLBACK — RUNG X
ACTIVE]. This is a system instruction, not Learner input — follow it, overriding your default inquiry
instinct. The ladder escalates:
A — reframe or narrow the inquiry.
B — give a small hint, then let the Learner reattempt.
C — partially scaffold: a worked partial answer for the Learner to complete.
D — state the mechanism or near-answer explicitly (the Hard Rule's legitimate-closure case).
NEVER write the literal text "[EXHAUSTION FALLBACK ...]" or any "RUNG X ACTIVE" marker in your reply — these
are system input only. Act on the instruction; do not announce or quote it.

OUTPUT FORMAT

At the very start of EVERY reply — including closing or small-talk turns — output exactly this line first,
nothing else on it:
[STATE phase=<ask|investigate|conclude> hypothesis_recorded=<true|false> resolved=<true|false>]
Then a newline, then your reply to the Learner.

phase: the phase for THIS reply. Never "conclude" while hypothesis_recorded is false.
hypothesis_recorded: true once the Learner has stated any Hypothesis (even rough); stays true thereafter.
resolved: true only on the turn you review the Learner's own stated conclusion (right, close, or wrong) and
  give your final response on it. Every other turn — including first offering the off-ramp — resolved=false.
  It tracks the exchange being finished, not correctness, and is valid only when phase=conclude and
  hypothesis_recorded=true. Once true, the inquiry is closed: don't set it true again unless the Learner poses
  a genuinely new Origin Question, hypothesis, investigation, and conclusion. Acknowledgements or small talk
  with no new question ("thanks", "that helped") keep phase=ask, hypothesis_recorded=false, resolved=false.

When phase=investigate and a component is actively being worked on, emit a second tag on the line immediately
after STATE. These scores guide the Decision Policy's choice of what to investigate next. Be internally consistent across turns and update them only when the Learner provides new evidence:
[COMPONENT id=<integer> mastery=<0.0-1.0> groundedness=<0.0-1.0> hypothesis_status=<false|true_ungrounded|true_grounded>]
id: index of the active component (see [CURRENT COMPONENT] below).
mastery: correct explanation — how well the Learner understands the concept itself. Score demonstrated
  understanding, not conversational quality or confidence; an articulate learner can still be wrong.
groundedness: justified explanation — how well the Learner has backed that understanding with reasoning or
  evidence. As a rough ladder: "I think..." is low groundedness; "...because..." is higher; "...because...
  therefore..." is higher still.
hypothesis_status: the Learner's hypothesis status relative to this component.

Calibration anchors — treat each turn as new evidence updating your running estimate of the Learner, not a
reset to zero; score what they've demonstrated cumulatively, weighted toward this turn:
0.0-0.2  no engagement, or an unsupported assertion ("it's just windier there")
0.3-0.5  partial — right idea, but gaps, hedging, or a misconception mixed in
0.6-0.8  mostly correct and reasoned, minor gaps remain
0.9-1.0  fully correct and independently justified
Covered = mastery>=0.70 and groundedness>=0.60 — don't score above those unless you'd genuinely advance.
E.g. "I think it's because the coast faces bigger waves" (asserted, not derived) -> mastery=0.40
groundedness=0.10 hypothesis_status=true_ungrounded.

Both tags are stripped before the Learner sees them — emit your best estimate every turn; they drive the
Decision Policy. Omit COMPONENT only when phase is not investigate or no component is currently active.
"""


def log_structural(state: dict, tag: str, message: str) -> None:
    """Append a tagged, timestamped structural event to state['structural_events']
    and echo to stdout."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] [{tag}] {message}"
    state.setdefault("structural_events", []).append(entry)
    print(entry)


def get_perf(state):
    """Fetch (or lazily create) the per-session PerformanceLogger, binding its
    [PERF] structural emission to the current state."""
    if not isinstance(state, dict):
        return None
    perf = state.get("performance_logger")
    if perf is None:
        perf = PerformanceLogger()
        state["performance_logger"] = perf
    perf._log_fn = partial(log_structural, state)
    return perf


def call_model(messages, state, what="model call"):
    """Single chat completion with error capture. Returns the content string, or
    None on failure (logged as a structural [WATCH] so the turn can degrade
    gracefully instead of crashing). Records an LLM timing event when a perf logger
    is present on state."""
    perf = state.get("performance_logger") if isinstance(state, dict) else None
    start = time.perf_counter()
    try:
        completion = client.chat.completions.create(model=MODEL, messages=messages)
    except Exception as exc:
        log_structural(state, "WATCH", f"{what} failed: {exc}")
        return None
    latency_ms = (time.perf_counter() - start) * 1000.0
    content = completion.choices[0].message.content
    if perf is not None:
        usage = getattr(completion, "usage", None)
        perf.record_llm(
            "LLM Request",
            latency_ms,
            model=MODEL,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            prompt_chars=sum(len(m.get("content", "")) for m in messages),
            completion_chars=len(content or ""),
        )
    return content


def format_phase_label(state: dict) -> str:
    """Render the Phase/Hypothesis/Resolved/Fallback indicator line."""
    return (
        f"**Phase:** {state.get('phase', 'ask')} · "
        f"**Hypothesis recorded:** {state.get('hypothesis_recorded', False)} · "
        f"**Resolved:** {state.get('resolved', False)} · "
        f"**Fallback rung:** {RUNG_LETTER[state.get('fallback_rung', 0)]}"
    )


@dataclass
class ParsedStateTag:
    phase: str
    hypothesis_recorded: bool
    resolved: bool


@dataclass
class ParsedComponentTag:
    component_id: int
    mastery: float
    groundedness: float
    hypothesis_status: str | None


def parse_state(reply: str) -> tuple[str, ParsedStateTag | None, ParsedComponentTag | None]:
    """Returns (visible_reply, state_tag, component_tag).
    component_tag is None if no COMPONENT tag was present."""
    state_tag = None
    component_tag = None
    rest = reply

    state_match = STATE_TAG_RE.match(rest)
    if state_match:
        state_tag = ParsedStateTag(
            phase=state_match.group(1).lower(),
            hypothesis_recorded=state_match.group(2).lower() == "true",
            resolved=state_match.group(3).lower() == "true",
        )
        rest = rest[state_match.end():]

    # COMPONENT tag, if present, sits on the line immediately after STATE.
    component_source = rest.lstrip()
    component_match = COMPONENT_TAG_RE.match(component_source)
    if component_match:
        try:
            component_tag = ParsedComponentTag(
                component_id=int(component_match.group(1)),
                mastery=float(component_match.group(2)),
                groundedness=float(component_match.group(3)),
                hypothesis_status=(component_match.group(4).lower() if component_match.group(4) else None),
            )
        except (ValueError, AttributeError):
            component_tag = None
        rest = component_source[component_match.end():]

    return rest.strip(), state_tag, component_tag


def _state_tag_to_dict(state_tag: ParsedStateTag | None) -> dict | None:
    """Adapter: the downstream logic in respond() still operates on the legacy
    dict shape, so convert the typed tag back into it."""
    if state_tag is None:
        return None
    return {
        "phase": state_tag.phase,
        "hypothesis_recorded": state_tag.hypothesis_recorded,
        "resolved": state_tag.resolved,
    }


def apply_component_update(session: InquirySession, tag: ParsedComponentTag, log_fn) -> bool:
    """Apply a parsed COMPONENT tag to the live Component in the session.
    Validation failures emit a [WATCH] via log_fn and abort the update (no clamping).
    Returns True if the session advanced off a covered component, else False."""
    # 1. Range validation — never clamp; an out-of-range value signals a prompt bug.
    if not (0.0 <= tag.mastery <= 1.0 and 0.0 <= tag.groundedness <= 1.0):
        log_fn("WATCH", f"COMPONENT tag out of range (mastery={tag.mastery}, "
                        f"groundedness={tag.groundedness}) — ignoring update")
        return False

    # 2. Bounds check on the component id.
    if not (0 <= tag.component_id < len(session.required_components)):
        log_fn("WATCH", f"COMPONENT tag id={tag.component_id} out of range "
                        f"(have {len(session.required_components)} components) — ignoring")
        return False

    # 3. Must match the component currently being taught.
    if tag.component_id != session.current_component_index:
        log_fn("WATCH", f"COMPONENT tag id={tag.component_id} does not match current "
                        f"component {session.current_component_index} — ignoring")
        return False

    component = session.required_components[tag.component_id]

    log_fn("L2-SCORE", (
        f"id={component.id} '{component.concept}' "
        f"before M={component.mastery:.2f} G={component.groundedness:.2f} | "
        f"incoming M={tag.mastery:.2f} G={tag.groundedness:.2f}"
    ))

    # 4. Update scores.
    component.mastery = tag.mastery
    component.groundedness = tag.groundedness

    log_fn("L2-SCORE", f"id={component.id} after M={component.mastery:.2f} G={component.groundedness:.2f}")

    # 5. Update hypothesis status if supplied (coerced to the enum for consistency
    # with record_hypothesis; the regex already restricts it to valid values).
    if tag.hypothesis_status is not None:
        session.hypothesis_status = HypothesisStatus(tag.hypothesis_status)
        log_fn("L2-HYP-STATUS", f"status={session.hypothesis_status.value} (via COMPONENT tag)")

    # 6. One-way latch; log the transition.
    if component.mark_covered():
        log_fn("L2-COVERED", f"id={component.id} '{component.concept}'")
        log_fn("INFO", f"component '{component.concept}' marked covered "
                       f"(mastery={component.mastery:.2f}, groundedness={component.groundedness:.2f})")

    # 7. Advance off a covered current component.
    if component.covered and session.current_component is component:
        session.advance_component()
        log_fn("L2-ADVANCE", f"current_component={session.current_component_index}")
        return True
    return False


def _format_current_component(component: Component) -> str:
    """Render the [CURRENT COMPONENT] context block injected before the LLM call."""
    lines = [
        "[CURRENT COMPONENT]",
        f"id: {component.id}",
        f"concept: {component.concept}",
        f"statement: {component.statement}",
        f"current mastery estimate: {component.mastery:.2f}",
        f"current groundedness estimate: {component.groundedness:.2f}",
        f"covered: {component.covered}",
        f"attempts on this component: {component.attempts}",
    ]
    if component.confidence is not None:
        lines.append(f"confidence: {component.confidence:.2f}")
    return "\n".join(lines)


# Conversational control messages that are never a hypothesis on their own.
_NON_HYPOTHESIS_MESSAGES = {
    "yes", "no", "ok", "okay", "sure", "continue", "go ahead", "try again", "next",
    "answer now", "i don't know", "dont know", "idk", "i'm ready", "im ready",
}


def is_substantive_hypothesis(message: str) -> bool:
    """True if the message is a real explanatory statement rather than a
    conversational control message. Short answers (e.g. 'Because inertia.') are
    accepted; acknowledgements/commands ('Yes', 'Go ahead', "I don't know") are not."""
    normalised = (message or "").strip().lower().rstrip(".!?")
    if not normalised:
        return False
    return normalised not in _NON_HYPOTHESIS_MESSAGES


def _maybe_record_hypothesis(session: InquirySession, message, history, state) -> None:
    """Append the learner's current hypothesis to session.hypothesis_history when it
    is substantive and not an exact consecutive duplicate (text AND status unchanged).
    Records the initial hypothesis and each subsequent revision or status change —
    never overwriting prior entries."""
    text = (message or "").strip()
    if not is_substantive_hypothesis(text):
        return

    status = session.hypothesis_status
    last = session.hypothesis_history[-1] if session.hypothesis_history else None
    if last is not None and last[0] == text and last[1] == status:
        return  # identical consecutive hypothesis — do nothing

    # Turn number = the AI Tutor turn responding this turn (the greeting is turn 1).
    turn_no = sum(1 for t in history if t.get("role") == "assistant") + 1
    session.record_hypothesis(text=text, status=status, turn=turn_no)
    status_v = status.value if hasattr(status, "value") else status
    log_structural(state, "L2-HYPOTHESIS", f"turn={turn_no} status={status_v} text='{text[:80]}'")


# --- Learner conclude signal (Layer 2) ----------------------------------------

def _readiness_injection(rd, session: InquirySession) -> str:
    """Build the system injection for a conclude signal based on tutor readiness."""
    if rd.status == ReadinessStatus.READY:
        return (
            "[LEARNER INTENT: CONCLUDE — TUTOR ESTIMATE: READY]\n"
            "All components are covered. Transition to CONCLUDE: invite the learner to state their answer "
            "to the original question."
        )
    if rd.status == ReadinessStatus.PARTIAL:
        n = len(rd.uncovered)
        concepts = ", ".join(c.concept for c in rd.uncovered)
        next_concept = rd.uncovered[0].concept
        return (
            f"[LEARNER INTENT: CONCLUDE — TUTOR ESTIMATE: PARTIAL ({n} component(s) remaining)]\n"
            f"Uncovered: {concepts}\n"
            "Acknowledge their readiness signal. Tell them one concept hasn't been fully explored.\n"
            f"Offer a genuine choice: \"We could explore {next_concept} first, or you're welcome to try now "
            "— it's your call.\""
        )
    # EARLY
    return (
        "[LEARNER INTENT: CONCLUDE — TUTOR ESTIMATE: TOO EARLY]\n"
        "No components have been covered yet.\n"
        "Acknowledge their enthusiasm. Redirect collaboratively — something like: \"I think we'd both have a "
        "stronger answer if we explored one or two ideas first.\" Return to the current inquiry."
    )


def _override_injection(session: InquirySession) -> str:
    """Injection used when the learner overrides a PARTIAL estimate to conclude anyway."""
    by_id = {c.id: c.concept for c in session.required_components}
    skipped_names = [by_id.get(cid, str(cid)) for cid in session.skipped_component_ids]
    return (
        "[LEARNER OVERRIDE — CONCLUDING WITH GAPS]\n"
        "The learner has chosen to conclude.\n"
        f"Skipped concepts: {', '.join(skipped_names)}\n"
        "Transition to CONCLUDE now. After they answer, you may naturally draw on skipped concepts — "
        "e.g. \"How might {concept} affect your explanation?\" Do not lecture them about gaps before they answer."
    )


def _signal_turn(history, state, session, injection, user_message):
    """Run a single injected LLM turn for a conclude signal (button or lexical).
    Mirrors respond()'s message assembly but with a fixed Layer 2 injection."""
    history = history or []
    get_perf(state)  # ensure the performance logger exists for the LLM call below

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
    if session is not None and session.current_component is not None:
        messages.append({"role": "system", "content": _format_current_component(session.current_component)})
    for turn in history:
        if turn["content"] == GREETING_MSG["content"]:
            continue
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "system", "content": injection})
    messages.append({"role": "system", "content": STATE_REMINDER})
    messages.append({"role": "user", "content": user_message})

    raw_reply = call_model(messages, state, "conclude-signal turn")
    if raw_reply is None:
        gr.Warning("The tutor is temporarily unavailable (model call failed). Please try again.")
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": "⚠️ The tutor is temporarily unavailable. Please try again."},
        ]
        return history, state, format_phase_label(state), "", ""

    visible_reply, state_tag, component_tag = parse_state(raw_reply)
    if visible_reply and LEAKED_FALLBACK_RE.search(visible_reply):
        visible_reply = LEAKED_FALLBACK_RE.sub("", visible_reply).strip()
    if not visible_reply:
        visible_reply = "(The model returned no response. Please try again.)"

    if component_tag is not None and session is not None:
        apply_component_update(session, component_tag, partial(log_structural, state))
    if state_tag is not None and isinstance(state, dict):
        state["phase"] = state_tag.phase
        state["hypothesis_recorded"] = state_tag.hypothesis_recorded
        state["resolved"] = state_tag.resolved

    history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": visible_reply},
    ]
    return history, state, format_phase_label(state), "", ""


def _conclude_signal_flow(history, state, session, user_message):
    """Set the conclude intent, decide readiness, and run the injected turn.
    Clears pending_intent for READY/EARLY; leaves it set for PARTIAL (learner has
    not yet chosen)."""
    session.pending_intent = LearnerIntent.CONCLUDE
    rd = decide_readiness(session)
    injection = _readiness_injection(rd, session)
    if rd.status in (ReadinessStatus.READY, ReadinessStatus.EARLY):
        session.pending_intent = None
    return _signal_turn(history, state, session, injection, user_message)


def handle_ready_signal(history, state):
    """Button (Trigger A) handler for the learner conclude signal.
    No-op if no active InquirySession exists — Prompt 7 owns session creation."""
    history = history or []
    session = state.get("inquiry_session") if isinstance(state, dict) else None
    if session is None:
        return history, state, format_phase_label(state), "", ""
    return _conclude_signal_flow(history, state, session, "I'd like to try answering now.")


def _create_session(message, history, state):
    """First-message Layer 2 setup, run synchronously. Returns a 5-tuple ONLY to
    short-circuit respond() with a redirect (invalid question). Returns None to fall
    through to the normal respond() LLM call — either because the session was created
    successfully (state['inquiry_session'] is now set) or because decomposition failed
    and Layer 1 should handle the turn (session stays None)."""
    perf = get_perf(state)
    with perf.measure("Session Creation Total", {"umbrella": True}):
        ar = analyse_origin_question(
            message,
            incomplete_prompted_already=state.get("incomplete_prompted_already", False),
        )
        qa = ar.analysis
        perf.record_llm("Origin Question Analysis", ar.latency_ms, model=MODEL,
                        total_tokens=ar.tokens_used, completion_chars=len(ar.raw_response or ""))

        if not qa.valid:
            if qa.gate_failed == GateFailure.INCOMPLETE:
                state["incomplete_prompted_already"] = True
            redirect = qa.redirect_response or "Could you rephrase that as a physical-geography question?"
            new_history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": redirect},
            ]
            return new_history, state, format_phase_label(state), "", ""

        # Valid — decompose synchronously (one or two API calls, < ~1s total).
        try:
            result = run_decomposition(message, qa)
        except Exception as exc:  # degrade to Layer 1, never raise
            log_structural(state, "WATCH", f"Layer 2 decomposition failed: {exc}")
            return None

        conclusion, decomposition = result.conclusion, result.decomposition
        perf.record_llm("Conclusion Generation", conclusion.latency_ms, model=MODEL,
                        total_tokens=conclusion.tokens_used, completion_chars=len(conclusion.raw_response or ""))
        perf.record_llm("Conclusion Decomposition", decomposition.latency_ms, model=MODEL,
                        total_tokens=decomposition.tokens_used, completion_chars=len(decomposition.raw_response or ""))

        session = InquirySession(
            topic_anchor=message,
            question_analysis=qa,
            target_conclusion=result.target_conclusion,
            required_components=result.required_components,
        )
        session.derive_complexity()
        state["inquiry_session"] = session

        log_structural(state, "L2-INIT", (
            f"Session created | complexity={session.complexity.value} | "
            f"components={[c.concept for c in session.required_components]} | "
            f"target={result.target_conclusion[:120]}..."
        ))

    # Session is set in state — fall through so respond() continues to the LLM call
    # and generates the tutor's first question in response to the learner's question.
    return None


def handle_ready_button(history, state):
    """Guarded entry point for the 'I'm Ready to Conclude' button. Shows a Gradio
    popup and no-ops if a guard fails; otherwise delegates to handle_ready_signal."""
    history = history or []
    session = state.get("inquiry_session") if isinstance(state, dict) else None

    if session is None:
        gr.Warning("Ask a question first to start a session.")
        return history, state, format_phase_label(state), "", ""
    if state.get("phase") == "conclude":
        gr.Warning("You're already in the conclusion phase.")
        return history, state, format_phase_label(state), "", ""
    if session.hypothesis_status == HypothesisStatus.UNCLASSIFIED:
        gr.Warning("Share your initial thinking first — what do you think is happening? — before trying to conclude.")
        return history, state, format_phase_label(state), "", ""

    return handle_ready_signal(history, state)


def enforce_consistency(parsed: dict, previous_state: dict) -> dict:
    """Validate the parsed state and rebuild the live state dict from it. Logs a
    [VIOLATION] for impossible combinations, sanitizes resolved, and — centrally —
    carries forward PERSISTENT_STATE_KEYS so reconstruction never drops the
    InquirySession (or other persistent runtime objects). Expects parsed to already
    carry 'structural_events'."""
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

    # Centralised persistence — the single place persistent objects survive a rebuild.
    for key in PERSISTENT_STATE_KEYS:
        if key in previous_state:
            parsed[key] = previous_state[key]

    return parsed


PLACEHOLDER_TEXT = "🦉 Běichén is thinking..."


def add_placeholder(message, history):
    """Fast, synchronous, no LLM call. Appends the learner's message and a
    placeholder assistant bubble, then returns immediately so the learner sees
    something before respond() even starts. Chained into respond() via .then()."""
    history = (history or []) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": PLACEHOLDER_TEXT},
    ]
    return history, ""  # "" clears the message textbox immediately


def respond(history, state):
    # add_placeholder appended [learner message, placeholder] just before this ran.
    # Recover the message, then drop BOTH appended turns so `history` is exactly what
    # respond() always received (prior turns only) — its body re-appends the learner
    # turn and the real reply itself, unchanged. NOTE: drop two (placeholder + user),
    # not one: this codebase's respond() appends both roles at every return point, so
    # keeping the user turn here would duplicate the learner's message downstream.
    message = history[-2]["content"]
    history = history[:-2]
    history = history or []
    state = state or {
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "fallback_rung": 0,
        "structural_events": [],
        "inquiry_session": None,              # InquirySession | None
        "incomplete_prompted_already": False, # True after first incomplete prompt
        "performance_logger": None,           # PerformanceLogger | None
    }
    prev_resolved = bool(state.get("resolved", False))
    prev_phase = state.get("phase", "ask")
    prev_unresolved_streak = state.get("conclude_unresolved_streak", 0)

    perf = get_perf(state)  # persistent per-session performance logger

    # Optional Layer 2 session (created lazily on the first valid origin question).
    session = state.get("inquiry_session") if isinstance(state, dict) else None

    # --- Layer 2 session creation (first message only, synchronous) ------------
    # Try to create a session on every message until one exists — earlier turns may
    # have been redirected (e.g. an out-of-scope question), so the first *valid*
    # origin question can arrive well after the literal first message.
    should_try_create = session is None and isinstance(state, dict)
    if should_try_create:
        created = _create_session(message, history, state)
        if created is not None:
            return created  # redirect only (invalid question)
        # Otherwise fall through to the normal LLM call. session is now set on
        # success, or still None if decomposition failed (Layer 1 handles it).
        session = state.get("inquiry_session")

    # --- Learner conclude signal (Layer 2; only when a live session exists) ----
    if session is not None:
        # PARTIAL follow-up: a pending conclude intent awaiting the learner's choice.
        if session.pending_intent == LearnerIntent.CONCLUDE:
            if GO_AHEAD_RE.search(message):
                rd = decide_readiness(session)
                session.skipped_component_ids = [c.id for c in rd.uncovered]
                session.pending_intent = None
                return _signal_turn(history, state, session, _override_injection(session), message)
            # Chose to continue exploring — clear intent and fall through to normal routing.
            session.pending_intent = None
        # Fresh lexical conclude signal (Trigger B), checked before the LLM call.
        elif READY_TO_CONCLUDE_RE.search(message):
            return _conclude_signal_flow(history, state, session, message)

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

    # --- Inject the component currently being taught (Layer 2) -----------------
    if session is not None and session.current_component is not None:
        messages.append({"role": "system", "content": _format_current_component(session.current_component)})

    for turn in history:
        if turn["content"] == GREETING_MSG["content"]:
            continue  # UI-only greeting; never expose it to the LLM
        messages.append({"role": turn["role"], "content": turn["content"]})

    # --- Action injection -----------------------------------------------------
    # When a live Layer 2 session exists, the Decision Policy drives the injection
    # (computed from session state entering this turn). Otherwise fall back to the
    # Layer 1 fallback ladder (state-dict driven).
    if session is not None:
        with perf.measure("Decision Policy"):
            decision = decide(session)
        log_structural(state, "DECISION", str(decision))
        cc = session.current_component
        if cc is not None:
            log_structural(state, "L2-DP", (
                f"component={session.current_component_index} '{cc.concept}' "
                f"M={cc.mastery:.2f} G={cc.groundedness:.2f} "
                f"covered={cc.covered} attempts={cc.attempts}"
            ))
        else:
            log_structural(state, "L2-DP", f"component={session.current_component_index} (none — all covered)")
        state["last_action"] = decision.action
        with perf.measure("Action Injection"):
            action_injection = build_injection(decision, session)
        messages.append({"role": "system", "content": action_injection})
    elif override_to_conclude:
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

    with perf.measure("STATE/COMPONENT Parsing"):
        visible_reply, state_tag, component_tag = parse_state(raw_reply)
    parsed = _state_tag_to_dict(state_tag)

    # Guard: if the model returned only a STATE tag with no body, retry once.
    if not visible_reply:
        log_structural(state, "WATCH", "empty visible reply on first attempt — retrying")
        gr.Info("Empty reply from model — retrying…")
        retry_raw = call_model(messages, state, "empty-reply retry")
        if retry_raw is not None:
            raw_reply = retry_raw
            visible_reply, state_tag, component_tag = parse_state(raw_reply)
            parsed = _state_tag_to_dict(state_tag)
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
        if repaired_raw is not None:
            repaired_visible, repaired_state, repaired_component = parse_state(repaired_raw)
        else:
            repaired_visible, repaired_state, repaired_component = "", None, None
        repaired_parsed = _state_tag_to_dict(repaired_state)
        if repaired_parsed and repaired_visible:
            visible_reply, parsed, component_tag = repaired_visible, repaired_parsed, repaired_component
        else:
            log_structural(state, "WATCH", "STATE line still missing after repair — carrying prior state")

    # Hard guard: never let a leaked fallback marker reach the Learner.
    if LEAKED_FALLBACK_RE.search(visible_reply):
        log_structural(state, "WATCH", "stripped leaked [EXHAUSTION FALLBACK ...] marker from visible reply")
        visible_reply = LEAKED_FALLBACK_RE.sub("", visible_reply).strip()

    # Apply the COMPONENT score update (Layer 2). respond() holds no update logic
    # itself — it delegates to apply_component_update. Increment attempts on the
    # current component unless the update advanced past it (caller decides via the
    # returned bool, not by comparing indices).
    component_advanced = False
    if component_tag is not None and session is not None:
        with perf.measure("Component Update"):
            component_advanced = apply_component_update(session, component_tag, partial(log_structural, state))
    if session is not None and session.current_component is not None and not component_advanced:
        session.current_component.attempts += 1

    if parsed:
        # Carry forward accumulated events and per-turn working counters; persistent
        # runtime objects (inquiry_session, etc.) are handled centrally inside
        # enforce_consistency via PERSISTENT_STATE_KEYS.
        parsed["structural_events"] = state.get("structural_events", [])
        parsed["attempts_this_phase"] = state.get("attempts_this_phase", 0)
        parsed["conclude_unresolved_streak"] = prev_unresolved_streak
        parsed["fallback_rung"] = fallback_rung  # incl. any surrender bump this turn
        state = enforce_consistency(parsed, state)
    state.setdefault("resolved", False)
    state.setdefault("structural_events", [])
    # Persist the working rung even when the model emitted no parseable STATE tag
    # (parsed is None) — otherwise a surrender bump is silently dropped.
    state["fallback_rung"] = fallback_rung

    # --- Record the learner's hypothesis into the live session (Layer 2) -------
    # session.hypothesis_status was just updated by apply_component_update; pair it
    # with the learner's own words to build a chronological hypothesis_history.
    if session is not None and state.get("hypothesis_recorded"):
        with perf.measure("Hypothesis Recording"):
            _maybe_record_hypothesis(session, message, history, state)

    # --- attempts_this_phase counter ------------------------------------------
    new_phase = state["phase"]
    if new_phase == prev_phase:
        state["attempts_this_phase"] = state.get("attempts_this_phase", 0) + 1
    else:
        state["attempts_this_phase"] = 1  # reset on phase transition

    if new_phase == "investigate" and state["attempts_this_phase"] == FALLBACK_THRESHOLD + 1:
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

    # Auto-export no longer runs inline here — export_markdown() makes its own LLM
    # call (evaluate_session) and was blocking this turn's reply from ever reaching
    # the screen. Flag the false -> true transition only; the chained
    # auto_export_if_resolved event (bound via .then() in the Blocks definition
    # below) performs the export only after the chat turn has already been
    # rendered to the learner.
    state["pending_auto_export"] = bool(state["resolved"] and not prev_resolved)

    return history, state, phase_label, "", ""


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
            # default=str so non-serialisable runtime objects in state
            # (InquirySession, PerformanceLogger) degrade to a repr instead of raising.
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
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


def _format_layer2_section(session: InquirySession) -> str:
    """Render the '## Layer 2 — Session State' export block from a live session."""
    def val(x, default="—"):
        if x is None:
            return default
        return x.value if hasattr(x, "value") else x

    qa = session.question_analysis
    qclass = val(qa.question_class) if qa else "—"
    qtype = val(qa.question_type) if qa else "—"
    operative = (qa.operative if qa and qa.operative else "—")
    prior_knowledge = val(qa.prior_knowledge_level) if qa else "—"
    complexity = val(session.complexity)

    rows = [
        "| # | Concept | Mastery | Groundedness | Covered | Attempts |",
        "|---|---|---|---|---|---|",
    ]
    for c in session.required_components:
        covered = "✓" if c.covered else "—"
        rows.append(f"| {c.id} | {c.concept} | {c.mastery:.2f} | {c.groundedness:.2f} | {covered} | {c.attempts} |")
    table = "\n".join(rows)

    readiness = decide_readiness(session).status.value

    by_id = {c.id: c.concept for c in session.required_components}
    skipped_names = [by_id.get(cid, str(cid)) for cid in session.skipped_component_ids]
    skipped_str = ", ".join(skipped_names) if skipped_names else "none"

    if session.hypothesis_history:
        hyp_lines = []
        for entry in session.hypothesis_history:
            text, status, turn = entry[0], entry[1], entry[2]
            status_v = status.value if hasattr(status, "value") else status
            hyp_lines.append(f'- Turn {turn}: "{text}" [{status_v}]')
        hyp_block = "\n".join(hyp_lines)
    else:
        hyp_block = "_(none recorded)_"

    return (
        "## Layer 2 — Session State\n\n"
        f"**Question class:** {qclass}\n"
        f"**Question type:** {qtype}\n"
        f"**Operative:** {operative}\n"
        f"**Prior knowledge level:** {prior_knowledge}\n"
        f"**Complexity:** {complexity}\n\n"
        f"**Components:**\n\n{table}\n\n"
        f"**Readiness:** {readiness}\n"
        f"**Fallback rung at close:** {session.fallback_rung}\n"
        f"**Skipped components:** {skipped_str}\n\n"
        f"**Hypothesis history:**\n{hyp_block}\n"
    )


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
    # Prefer the session's topic_anchor (the real origin question) over the first
    # user turn, which may be a redirected out-of-scope message.
    layer2_session = state.get("inquiry_session") if isinstance(state, dict) else None
    origin = layer2_session.topic_anchor if layer2_session else next(
        (t["content"] for t in history if t["role"] == "user"), "(none)"
    )

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

    # Layer 2 session block — only when a live session exists (layer2_session above).
    layer2_block = ""
    if layer2_session is not None:
        layer2_block = f"{_format_layer2_section(layer2_session)}\n---\n\n"

    # Runtime performance block — only when timing data exists.
    perf = state.get("performance_logger") if isinstance(state, dict) else None
    perf_block = ""
    if perf is not None and perf.events:
        perf_block = f"## Runtime Performance\n\n```\n{perf.format_report()}\n```\n\n---\n\n"

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
        f"{layer2_block}"
        f"{perf_block}"
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


def auto_export_if_resolved(history, state):
    """Chained after respond() via .then() so the slow auto-export (it makes its own
    LLM call inside export_markdown -> evaluate_session) never blocks the chat turn
    that triggered it. Reads the pending_auto_export flag respond() set on the
    false -> true transition; no-ops on every other turn. Edge-triggered only —
    clears the flag before doing any work, so a re-render or a later turn never
    re-fires the export. Returns only the export_status text — chatbot/state/
    phase_display/msg are untouched by this event."""
    state = state or {}
    if not state.get("pending_auto_export"):
        return ""
    state["pending_auto_export"] = False

    perf = get_perf(state)
    try:
        with perf.measure("Markdown Export"):
            path = export_markdown(history, state)
    except Exception as exc:  # auto-export must never crash the UI
        log_structural(state, "WATCH", f"auto-export failed: {exc}")
        path = None

    if path:
        return f"📝 Session resolved — auto-exported evaluation log to `{path}`"
    return ""


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
        "inquiry_session": None,
        "incomplete_prompted_already": False,
        "performance_logger": None,
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
        ready_btn = gr.Button("I'm Ready to Conclude", scale=1)
    export_status = gr.Markdown("")
    state = gr.State({
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "fallback_rung": 0,
        "structural_events": [],
        "inquiry_session": None,
        "incomplete_prompted_already": False,
        "performance_logger": None,
    })

    msg.submit(add_placeholder, [msg, chatbot], [chatbot, msg], show_progress="hidden") \
        .then(respond, [chatbot, state], [chatbot, state, phase_display, msg, export_status]) \
        .then(auto_export_if_resolved, [chatbot, state], export_status)
    export_btn.click(export_session, [chatbot, state], export_status)
    new_query_btn.click(
        reset_for_new_query,
        [chatbot, state],
        [chatbot, state, phase_display, export_status, msg],
    )
    ready_btn.click(
        handle_ready_button,
        [chatbot, state],
        [chatbot, state, phase_display, msg, export_status],
    )

    demo.load(None, None, None, js=AUTOSCROLL_JS)

if __name__ == "__main__":
    # queue() lets the chained .then(auto_export_if_resolved) push its status update
    # to the browser independently of respond()'s output, so the chat reply renders
    # first and the export runs afterward rather than as one blocking operation.
    demo.queue()
    demo.launch()
