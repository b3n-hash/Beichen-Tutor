"""Live verification for layer2/question_analysis. Run from the repo root:

    python3 tests/test_question_analysis.py

Hard assertions are made ONLY on deterministic gate outputs (valid, gate_failed).
Model judgement fields (question_class, prior_knowledge_level, assumed_known_concepts)
get weaker membership/substring checks. Each full AnalysisResult is printed verbatim
for human inspection, with PASS/FAIL per assertion underneath.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layer2.question_analysis import analyse_origin_question


def lower_set(values):
    return [str(v).lower() for v in (values or [])]


def has_term(term, concepts):
    """Case-insensitive substring match of `term` anywhere in the concept list."""
    return any(term.lower() in c for c in lower_set(concepts))


# Each case: the question, the deterministic gate expectations (hard), and a list
# of (label, predicate) soft checks on model-judgement fields. gate_failed may be a
# tuple, meaning "must be one of these" (used where the spec itself allows either).
CASES = [
    {
        "q": "Why do rivers meander?",
        "valid": True,
        "gate_failed": None,
        "soft": [
            ("prior_knowledge_level in (novice, intermediate)",
             lambda a: a.prior_knowledge_level in ("novice", "intermediate")),
        ],
    },
    {
        "q": "Where is Paris relative to Madrid?",
        "valid": False,
        "gate_failed": ("out_of_scope", "not_decomposable"),
        "soft": [],
    },
    {
        "q": "Why does adiabatic cooling cause orographic precipitation?",
        "valid": True,
        "gate_failed": None,
        "soft": [
            ("'adiabatic' present in assumed_known_concepts",
             lambda a: has_term("adiabatic", a.assumed_known_concepts)),
            ("'orographic' present in assumed_known_concepts",
             lambda a: has_term("orographic", a.assumed_known_concepts)),
        ],
    },
    {
        "q": "Why is the sky green?",
        "valid": True,
        "gate_failed": None,
        "soft": [
            ("question_class in (false_premise, misconception)",
             lambda a: a.question_class in ("false_premise", "misconception")),
        ],
    },
    {
        "q": "Why is the sky blue because of the ocean?",
        "valid": True,
        "gate_failed": None,
        "soft": [
            ("question_class in (misconception, false_premise)",
             lambda a: a.question_class in ("misconception", "false_premise")),
        ],
    },
    {
        "q": "Explain river meanders.",
        "valid": True,
        "gate_failed": None,
        "soft": [],
    },
    {
        "q": "",
        "valid": False,
        "gate_failed": "incomplete",
        "soft": [],
    },
]


def dump(label, result):
    a = result.analysis
    print(f"\n===== {label} =====")
    print(f"  valid                 : {a.valid}")
    print(f"  gate_failed           : {a.gate_failed}")
    print(f"  question_class        : {a.question_class}")
    print(f"  question_type         : {a.question_type}")
    print(f"  operative             : {a.operative}")
    print(f"  action                : {a.action}")
    print(f"  content_keywords      : {a.content_keywords}")
    print(f"  grammar_keywords      : {a.grammar_keywords}")
    print(f"  prior_knowledge_level : {a.prior_knowledge_level}")
    print(f"  assumed_known_concepts: {a.assumed_known_concepts}")
    print(f"  confidence            : {a.confidence}")
    print(f"  redirect_response     : {a.redirect_response}")
    print(f"  latency_ms            : {result.latency_ms:.1f}")
    print(f"  tokens_used           : {result.tokens_used}")


def gate_failed_check(actual, expected):
    if isinstance(expected, tuple):
        return actual in expected, f"gate_failed in {expected}"
    if expected is None:
        return actual is None, "gate_failed is None"
    return actual == expected, f"gate_failed == {expected!r}"


hard_total = 0
hard_pass = 0
heuristic_pass = 0
heuristic_warn = 0

for i, case in enumerate(CASES, 1):
    result = analyse_origin_question(case["q"])
    a = result.analysis
    dump(f"{i}. {case['q']!r}", result)

    print("  --- assertions ---")
    # Hard: valid (exact)
    ok = a.valid == case["valid"]
    hard_total += 1
    hard_pass += ok
    print(f"    [{'PASS' if ok else 'FAIL'}] (hard) valid == {case['valid']}")

    # Hard: gate_failed (exact or membership)
    ok, desc = gate_failed_check(a.gate_failed, case["gate_failed"])
    hard_total += 1
    hard_pass += ok
    print(f"    [{'PASS' if ok else 'FAIL'}] (hard) {desc}")

    # Heuristic: model-judgement fields
    for label, pred in case["soft"]:
        ok = pred(a)
        heuristic_pass += ok
        heuristic_warn += not ok
        print(f"    [{'PASS' if ok else 'WARN'}] Heuristic check: {label}")

print()
print("===========================")
print(f"Hard assertions: {hard_pass}/{hard_total} PASS")
print(f"Heuristic checks: {heuristic_pass} PASS | {heuristic_warn} WARN")
print("===========================")
sys.exit(1 if hard_pass != hard_total else 0)
