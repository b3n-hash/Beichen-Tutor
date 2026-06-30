"""Verification for layer2/decision_policy. Run from the repo root:

    python3 tests/test_decision_policy.py

Pure policy logic — no LLM calls. Prints all PolicyDecision fields per case.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.decision_policy import (
    ACTION_ADVANCE,
    ACTION_BRIDGE,
    ACTION_DISCONFIRM,
    ACTION_FALLBACK,
    decide,
)
from layer2.models import Component, HypothesisStatus, InquirySession

results = []


def check(label, got, expected):
    ok = got == expected
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")


def dump(decision):
    print(f"  action     = {decision.action!r}")
    print(f"  rung       = {decision.rung}")
    print(f"  reason     = {decision.reason!r}")
    print(f"  context    = {decision.context}")
    print(f"  confidence = {decision.confidence}")


def make_session(m, g, attempts=0, fallback_rung=0, hypothesis_status=None):
    c0 = Component(id=0, concept="longshore drift", statement="sediment moves along the coast",
                   mastery=m, groundedness=g, attempts=attempts)
    c1 = Component(id=1, concept="deposition", statement="sediment settles where flow slows")
    s = InquirySession(topic_anchor="why do spits form", required_components=[c0, c1],
                       fallback_rung=fallback_rung)
    if hypothesis_status is not None:
        s.hypothesis_status = hypothesis_status
    return s, c0


# --- Case 1: BRIDGE (m<0.4, g>=0.4) ------------------------------------------
print("== Case 1: m=0.3 g=0.55 attempts=2 fallback_rung=0 -> BRIDGE ==")
s, c0 = make_session(0.3, 0.55, attempts=2, fallback_rung=0)
d = decide(s)
dump(d)
check("action", d.action, ACTION_BRIDGE)

# --- Case 2: FALLBACK rung 2 ((8-1)//3 = 2) ----------------------------------
print("\n== Case 2: m=0.3 g=0.3 attempts=8 fallback_rung=0 -> FALLBACK rung=2 ==")
s, c0 = make_session(0.3, 0.3, attempts=8, fallback_rung=0)
d = decide(s)
dump(d)
check("action", d.action, ACTION_FALLBACK)
check("rung", d.rung, 2)

# --- Case 3: ADVANCE + mark_covered ------------------------------------------
print("\n== Case 3: m=0.75 g=0.65 -> ADVANCE, component covered ==")
s, c0 = make_session(0.75, 0.65)
d = decide(s)
dump(d)
print(f"  component0.covered = {c0.covered}")
print(f"  context.next_component = {d.context.get('next_component')}")
check("action", d.action, ACTION_ADVANCE)
check("component covered", c0.covered, True)

# --- Case 4: DISCONFIRM fires before matrix ----------------------------------
print("\n== Case 4: hypothesis_status=FALSE, m=0.3 g=0.3 -> DISCONFIRM ==")
s, c0 = make_session(0.3, 0.3, hypothesis_status=HypothesisStatus.FALSE)
d = decide(s)
dump(d)
check("action", d.action, ACTION_DISCONFIRM)

print(f"\nSUMMARY: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
