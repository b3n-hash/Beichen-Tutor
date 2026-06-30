"""Verification for the COMPONENT score-extraction system (Prompt 4).

Run from the repo root:

    python3 tests/test_score_extraction.py

Pure parsing + state-update logic; no LLM calls. Prints all values verbatim.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import ParsedComponentTag, apply_component_update, parse_state
from layer2.models import Component, InquirySession

results = []


def check(label, got, expected):
    ok = got == expected
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got={got!r} expected={expected!r}")


# --- 1. parse_state on a full STATE + COMPONENT reply -------------------------
print("== parse_state: STATE + COMPONENT + body ==")
reply = (
    "[STATE phase=investigate hypothesis_recorded=true resolved=false]\n"
    "[COMPONENT id=0 mastery=0.72 groundedness=0.61 hypothesis_status=true_ungrounded]\n"
    "The waves approach the shore at an angle..."
)
visible, state_tag, component_tag = parse_state(reply)
print(f"  visible_reply  = {visible!r}")
print(f"  state_tag      = {state_tag}")
print(f"  component_tag  = {component_tag}")
check("visible_reply", visible, "The waves approach the shore at an angle...")
check("state.phase", state_tag.phase, "investigate")
check("state.hypothesis_recorded", state_tag.hypothesis_recorded, True)
check("state.resolved", state_tag.resolved, False)
check("component.component_id", component_tag.component_id, 0)
check("component.mastery", component_tag.mastery, 0.72)
check("component.groundedness", component_tag.groundedness, 0.61)
check("component.hypothesis_status", component_tag.hypothesis_status, "true_ungrounded")


# --- 2. apply_component_update — happy path, crosses both thresholds ----------
print("\n== apply_component_update: thresholds crossed -> covered + advance ==")
# Direct confirmation of mark_covered() per the spec.
fresh = Component(id=0, concept="longshore drift", statement="...", mastery=0.72, groundedness=0.61)
check("fresh.mark_covered() returns True", fresh.mark_covered(), True)

comp0 = Component(id=0, concept="longshore drift", statement="sediment moves along the coast")
comp1 = Component(id=1, concept="deposition", statement="sediment settles where flow slows")
session = InquirySession(topic_anchor="why do spits form", required_components=[comp0, comp1])
events = []
log_fn = lambda tag, msg: events.append((tag, msg))  # noqa: E731
apply_component_update(session, component_tag, log_fn)
print(f"  comp0.mastery={comp0.mastery} comp0.groundedness={comp0.groundedness} covered={comp0.covered}")
print(f"  session.current_component_index={session.current_component_index}")
print(f"  session.hypothesis_status={session.hypothesis_status}")
print(f"  events={events}")
check("comp0.mastery updated", comp0.mastery, 0.72)
check("comp0.covered", comp0.covered, True)
check("advanced off covered component", session.current_component_index, 1)
check("hypothesis_status propagated", session.hypothesis_status, "true_ungrounded")
check("INFO covered event emitted", any(t == "INFO" and "marked covered" in m for t, m in events), True)


# --- 3. Out-of-range mastery (1.4) -> ignored, [WATCH] -----------------------
print("\n== apply_component_update: out-of-range mastery=1.4 -> ignored ==")
c = Component(id=0, concept="x", statement="y", mastery=0.1, groundedness=0.1)
sess = InquirySession(topic_anchor="t", required_components=[c])
ev = []
bad_tag = ParsedComponentTag(component_id=0, mastery=1.4, groundedness=0.5, hypothesis_status=None)
apply_component_update(sess, bad_tag, lambda tag, msg: ev.append((tag, msg)))
print(f"  c.mastery={c.mastery} (unchanged) covered={c.covered}")
print(f"  events={ev}")
check("mastery unchanged (not clamped)", c.mastery, 0.1)
check("update ignored — not covered", c.covered, False)
check("[WATCH] emitted", any(t == "WATCH" for t, m in ev), True)


# --- 4. Mismatched id (tag id=1, current index=0) -> ignored, [WATCH] --------
print("\n== apply_component_update: id mismatch -> ignored ==")
a0 = Component(id=0, concept="a", statement="...")
a1 = Component(id=1, concept="b", statement="...")
sess2 = InquirySession(topic_anchor="t", required_components=[a0, a1])  # current index 0
ev2 = []
mismatch_tag = ParsedComponentTag(component_id=1, mastery=0.9, groundedness=0.9, hypothesis_status=None)
apply_component_update(sess2, mismatch_tag, lambda tag, msg: ev2.append((tag, msg)))
print(f"  a1.mastery={a1.mastery} (unchanged) covered={a1.covered}")
print(f"  current_component_index={sess2.current_component_index} (unchanged)")
print(f"  events={ev2}")
check("a1 untouched", a1.mastery, 0.0)
check("no advance", sess2.current_component_index, 0)
check("[WATCH] mismatch emitted", any(t == "WATCH" and "does not match current" in m for t, m in ev2), True)


print(f"\nSUMMARY: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
