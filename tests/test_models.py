"""Verification for the Layer 2 data model. Run from the repo root:

    python3 tests/test_models.py

Prints every result verbatim, then a final PASS/FAIL summary.
"""

import os
import sys

# No tests/__init__.py exists, so put the repo root on the path explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.models import Component, InquirySession

results = []


def check(label, got, expected):
    ok = got == expected
    results.append(ok)
    print(f"{label}: got={got!r} expected={expected!r} -> {'PASS' if ok else 'FAIL'}")


# 1. Component crosses both thresholds -> mark_covered() transitions to True.
comp = Component(concept="mantle convection", statement="heat drives circulation",
                 mastery=0.8, groundedness=0.65)
print("== Test 1: mark_covered transition ==")
check("  mark_covered() return", comp.mark_covered(), True)
check("  covered after call", comp.covered, True)

# 2. Latch holds: drop mastery to 0.0, re-call -> returns False, stays covered.
print("== Test 2: one-way latch holds ==")
comp.mastery = 0.0
check("  mark_covered() return (already covered)", comp.mark_covered(), False)
check("  covered still True", comp.covered, True)

# 3. Session with 3 components; advance twice -> current_component is the third.
print("== Test 3: advance_component navigation ==")
c1 = Component(concept="c1", statement="s1")
c2 = Component(concept="c2", statement="s2")
c3 = Component(concept="c3", statement="s3")
session = InquirySession(topic_anchor="why do rivers meander",
                         required_components=[c1, c2, c3])
session.advance_component()
session.advance_component()
check("  current_component_index", session.current_component_index, 2)
check("  current_component is third", session.current_component is c3, True)

# 4. all_covered is False until every component is covered.
print("== Test 4: all_covered gating ==")
check("  all_covered (none covered)", session.all_covered, False)
c1.mastery, c1.groundedness = 0.9, 0.9
c2.mastery, c2.groundedness = 0.9, 0.9
c3.mastery, c3.groundedness = 0.9, 0.9
c1.mark_covered()
c2.mark_covered()
check("  all_covered (2 of 3 covered)", session.all_covered, False)
c3.mark_covered()
check("  all_covered (all 3 covered)", session.all_covered, True)

print()
print(f"SUMMARY: {sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
