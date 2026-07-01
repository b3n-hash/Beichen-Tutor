"""Verification for the learner conclude signal (Prompt 6). Run from repo root:

    python3 tests/test_readiness.py

decide_readiness is pure (no LLM). The PARTIAL state sequence mocks the LLM call
so the app handlers run offline. Prints ReadinessDecision fields + session state.
"""

import os
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app
from layer2.decision_policy import decide_readiness
from layer2.models import Component, InquirySession, LearnerIntent, ReadinessStatus

results = []


def check(label, got, expected):
    ok = got == expected
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")


def make_session(covered_flags):
    comps = []
    for i, cov in enumerate(covered_flags):
        c = Component(id=i, concept=f"concept_{i}", statement=f"statement {i}")
        if cov:
            c.mastery, c.groundedness = 0.9, 0.9
            c.mark_covered()
        comps.append(c)
    return InquirySession(topic_anchor="why do spits form", required_components=comps)


def dump_rd(rd):
    print(f"  status    = {rd.status}")
    print(f"  uncovered = {[c.concept for c in rd.uncovered]} (ids={[c.id for c in rd.uncovered]})")


# --- 1. READY: all 3 covered --------------------------------------------------
print("== Case 1: all 3 covered -> READY ==")
s = make_session([True, True, True])
before = deepcopy(s)
rd = decide_readiness(s)
dump_rd(rd)
check("status", rd.status, ReadinessStatus.READY)
check("uncovered empty", rd.uncovered, [])
check("no mutation", s == before, True)

# --- 2. PARTIAL: 2 of 3 covered ----------------------------------------------
print("\n== Case 2: 2 of 3 covered -> PARTIAL ==")
s = make_session([True, False, True])
before = deepcopy(s)
rd = decide_readiness(s)
dump_rd(rd)
check("status", rd.status, ReadinessStatus.PARTIAL)
check("uncovered is component 1", [c.id for c in rd.uncovered], [1])
check("uncovered[0].id is int", isinstance(rd.uncovered[0].id, int), True)
check("no mutation", s == before, True)

# --- 3. EARLY: 0 of 3 covered ------------------------------------------------
print("\n== Case 3: 0 of 3 covered -> EARLY ==")
s = make_session([False, False, False])
before = deepcopy(s)
rd = decide_readiness(s)
dump_rd(rd)
check("status", rd.status, ReadinessStatus.EARLY)
check("3 uncovered", len(rd.uncovered), 3)
check("no mutation", s == before, True)

# --- 4. PARTIAL state sequence (handlers, mocked LLM) ------------------------
print("\n== Case 4: PARTIAL -> signal -> 'go ahead' override ==")


class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})})]


# Mock: model always returns a valid STATE tag, no COMPONENT tag (coverage unchanged).
app.client.chat.completions.create = lambda **k: _FakeResp(
    "[STATE phase=conclude hypothesis_recorded=true resolved=false]\nOkay, let's hear your answer."
)

session = make_session([True, False, True])  # PARTIAL; component 1 uncovered
uncovered_id = [c.id for c in session.required_components if not c.covered][0]
state = {
    "phase": "investigate", "hypothesis_recorded": True, "resolved": False,
    "attempts_this_phase": 1, "fallback_rung": 0, "structural_events": [],
    "inquiry_session": session,
}

# Step A: fire the conclude signal (button path).
history, state, _, _, _ = app.handle_ready_signal([], state)
print(f"  after signal: pending_intent={session.pending_intent}, "
      f"skipped_component_ids={session.skipped_component_ids}")
check("pending_intent still CONCLUDE (PARTIAL)", session.pending_intent, LearnerIntent.CONCLUDE)

# Step B: learner replies "yes, go ahead" -> override path.
# respond() now takes (history, state); the UI's add_placeholder step appends the
# learner message + placeholder bubble first, so simulate that here.
history = history + [
    {"role": "user", "content": "yes, go ahead"},
    {"role": "assistant", "content": app.PLACEHOLDER_TEXT},
]
history, state, _, _, _ = app.respond(history, state)
print(f"  after override: pending_intent={session.pending_intent}, "
      f"skipped_component_ids={session.skipped_component_ids}")
check("skipped_component_ids == [uncovered.id]", session.skipped_component_ids, [uncovered_id])
check("pending_intent cleared", session.pending_intent, None)

print(f"\nSUMMARY: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
