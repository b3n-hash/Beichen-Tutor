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
        "structural_events": [],
    }
    prev_resolved = bool(state.get("resolved", False))
    prev_phase = state.get("phase", "ask")
    prev_unresolved_streak = state.get("conclude_unresolved_streak", 0)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})

    completion = client.chat.completions.create(model=MODEL, messages=messages)
    raw_reply = completion.choices[0].message.content
    visible_reply, parsed = parse_state(raw_reply)

    # Guard: if the model returned only a STATE tag with no body, retry once.
    if not visible_reply:
        log_structural(state, "WATCH", "empty visible reply on first attempt — retrying")
        gr.Info("Empty reply from model — retrying…")
        completion = client.chat.completions.create(model=MODEL, messages=messages)
        raw_reply = completion.choices[0].message.content
        visible_reply, parsed = parse_state(raw_reply)
        if not visible_reply:
            log_structural(state, "WATCH", "empty visible reply on retry — returning placeholder")
            visible_reply = "(The model returned no response. Please try again.)"

    if parsed:
        # Carry forward accumulated events and counters before state is replaced.
        parsed["structural_events"] = state.get("structural_events", [])
        parsed["attempts_this_phase"] = state.get("attempts_this_phase", 0)
        parsed["conclude_unresolved_streak"] = prev_unresolved_streak
        state = enforce_consistency(parsed)
    state.setdefault("resolved", False)
    state.setdefault("structural_events", [])

    # --- attempts_this_phase counter ------------------------------------------
    new_phase = state["phase"]
    if new_phase == prev_phase:
        state["attempts_this_phase"] = state.get("attempts_this_phase", 0) + 1
    else:
        state["attempts_this_phase"] = 1  # reset on phase transition

    if state["attempts_this_phase"] > FALLBACK_THRESHOLD:
        log_structural(
            state, "FALLBACK",
            f"Learner on attempt {state['attempts_this_phase']} in phase={new_phase}; "
            f"escalation may be overdue (threshold={FALLBACK_THRESHOLD})",
        )

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

    phase_label = (
        f"**Phase:** {state['phase']} · **Hypothesis recorded:** {state['hypothesis_recorded']}"
        f" · **Resolved:** {state['resolved']}"
    )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": visible_reply},
    ]

    # Edge-triggered auto-export: fire once on the false -> true transition only.
    auto_status = ""
    if state["resolved"] and not prev_resolved:
        path = export_markdown(history, state)
        if path:
            auto_status = f"📝 Session resolved — auto-exported evaluation log to `{path}`"

    return history, state, phase_label, "", auto_status


def export_session(history, state):
    history = history or []
    state = state or {"phase": "ask", "hypothesis_recorded": False, "structural_events": []}

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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

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
    state = state or {"phase": "ask", "hypothesis_recorded": False, "resolved": False, "structural_events": []}

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
    with open(path, "w", encoding="utf-8") as f:
        f.write(document)

    return path


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
    chatbot = gr.Chatbot(type="messages", height=480, elem_id="polaris-chatbot")
    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Ask a physical geography question...",
            scale=4,
        )
        export_btn = gr.Button("Export Session", scale=1)
    export_status = gr.Markdown("")
    state = gr.State({
        "phase": "ask",
        "hypothesis_recorded": False,
        "resolved": False,
        "attempts_this_phase": 0,
        "structural_events": [],
    })

    msg.submit(respond, [msg, chatbot, state], [chatbot, state, phase_display, msg, export_status])
    export_btn.click(export_session, [chatbot, state], export_status)

    demo.load(None, None, None, js=AUTOSCROLL_JS)

if __name__ == "__main__":
    demo.launch()
