"""Tests for layer2/conclusion_decomp.py

Four cases:
  1. "Why do rivers meander?"           — novice, known_phenomenon (discoverability ordering)
  2. "Why does adiabatic cooling cause orographic precipitation?"
                                         — advanced, mastery preseeding
  3. "Why is the sky blue because of the ocean?"
                                         — misconception, ensure pipeline succeeds
  4. "Why is the sky blue?"             — novice, known_phenomenon (discoverability ordering)

Hard assertions: component count bounds, non-empty fields, no duplicates,
                 mastery preseeding correctness.
Heuristic checks (WARN, not FAIL): causal ordering looks plausible.
"""

from __future__ import annotations

import sys
from dataclasses import fields

from layer2.conclusion_decomp import run_decomposition
from layer2.models import (
    PriorKnowledge,
    QuestionAnalysis,
    QuestionClass,
    QuestionType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(
    question_type: QuestionType,
    question_class: QuestionClass,
    operative: str,
    prior_knowledge: PriorKnowledge = PriorKnowledge.NOVICE,
    assumed_known_concepts: list[str] | None = None,
) -> QuestionAnalysis:
    return QuestionAnalysis(
        valid=True,
        question_type=question_type,
        question_class=question_class,
        operative=operative,
        prior_knowledge_level=prior_knowledge,
        assumed_known_concepts=assumed_known_concepts or [],
        confidence=0.9,
    )


def _assert_components_valid(components, label: str) -> list[str]:
    """Run all hard assertions; return list of WARN strings for heuristic checks."""
    hard_pass = 0
    hard_total = 0
    warns: list[str] = []

    def hard(condition: bool, msg: str):
        nonlocal hard_pass, hard_total
        hard_total += 1
        if condition:
            hard_pass += 1
        else:
            raise AssertionError(f"[{label}] HARD FAIL: {msg}")

    # Count bounds
    hard(2 <= len(components) <= 6,
         f"Component count {len(components)} not in [2, 6]")

    concepts_seen: set[str] = set()
    for i, c in enumerate(components):
        hard(bool(c.concept.strip()),
             f"Component {i}: empty concept")
        hard(bool(c.statement.strip()),
             f"Component {i}: empty statement")
        key = c.concept.strip().lower()
        hard(key not in concepts_seen,
             f"Duplicate concept at index {i}: '{c.concept}'")
        concepts_seen.add(key)
        hard(c.mastery in (0.0, 0.5),
             f"Component {i}: mastery {c.mastery} not in {{0.0, 0.5}}")
        hard(c.groundedness == 0.0,
             f"Component {i}: groundedness should be 0.0, got {c.groundedness}")
        hard(not c.covered,
             f"Component {i}: covered should be False")
        hard(c.attempts == 0,
             f"Component {i}: attempts should be 0")

    # Heuristic: first concept looks like a prerequisite (very loose — just warn)
    if components:
        first = components[0].concept.lower()
        last = components[-1].concept.lower()
        causal_words = {"cause", "mechanism", "process", "effect", "result", "consequence"}
        if any(w in last for w in causal_words) and any(w in first for w in causal_words):
            warns.append("Heuristic: first and last component both look like consequences — ordering may be reversed")

    return warns, hard_pass, hard_total


def _print_components(components, prefix=""):
    for i, c in enumerate(components):
        print(f"  [{i}] concept      : {c.concept}")
        print(f"       statement   : {c.statement}")
        print(f"       mastery     : {c.mastery}")
        print(f"       groundedness: {c.groundedness}")
        print(f"       covered     : {c.covered}")
        print(f"       attempts    : {c.attempts}")
        print()


def _warn_if_premature_jargon(components, assumed_known_concepts, jargon_terms, label) -> list[str]:
    """Heuristic only, never a hard fail: Component 0 should not be built around jargon the
    learner hasn't already demonstrated (per assumed_known_concepts)."""
    if not components:
        return []
    known_lower = {k.lower() for k in assumed_known_concepts}
    first_text = f"{components[0].concept} {components[0].statement}".lower()
    warns = []
    for term in jargon_terms:
        if term in first_text and not any(term in k or k in term for k in known_lower):
            warns.append(
                f"Heuristic [{label}]: Component 0 leads with specialist term '{term}' "
                f"('{components[0].concept}') with no prior exposure in assumed_known_concepts"
            )
    return warns


# ---------------------------------------------------------------------------
# Case 1: "Why do rivers meander?" — novice, known_phenomenon
# ---------------------------------------------------------------------------

def case_1() -> tuple[int, int, list[str]]:
    question = "Why do rivers meander?"
    analysis = _make_analysis(
        question_type=QuestionType.WHY,
        question_class=QuestionClass.KNOWN_PHENOMENON,
        operative="Why do",
        prior_knowledge=PriorKnowledge.NOVICE,
    )

    result = run_decomposition(question, analysis)

    print("=" * 60)
    print("CASE 1 — Why do rivers meander?")
    print("=" * 60)
    print(f"\nTarget conclusion:\n  {result.target_conclusion}\n")
    print(f"  Conclusion latency : {result.conclusion.latency_ms:.0f} ms")
    print(f"  Conclusion tokens  : {result.conclusion.tokens_used}")
    print(f"  Decomp latency     : {result.decomposition.latency_ms:.0f} ms")
    print(f"  Decomp tokens      : {result.decomposition.tokens_used}")
    print(f"\nComponents ({len(result.required_components)}):")
    _print_components(result.required_components)

    warns, hp, ht = _assert_components_valid(result.required_components, "Case 1")
    warns += _warn_if_premature_jargon(
        result.required_components, [],
        ["helicoidal flow", "helicoidal", "secondary circulation", "differential erosion"],
        "Case 1",
    )
    return hp, ht, warns


# ---------------------------------------------------------------------------
# Case 2: Adiabatic cooling / orographic precipitation — advanced, preseeding
# ---------------------------------------------------------------------------

def case_2() -> tuple[int, int, list[str]]:
    question = "Why does adiabatic cooling cause orographic precipitation?"
    assumed_known = ["adiabatic cooling", "orographic precipitation"]
    analysis = _make_analysis(
        question_type=QuestionType.WHY,
        question_class=QuestionClass.KNOWN_PHENOMENON,
        operative="Why does",
        prior_knowledge=PriorKnowledge.ADVANCED,
        assumed_known_concepts=assumed_known,
    )

    result = run_decomposition(question, analysis)

    print("=" * 60)
    print("CASE 2 — Adiabatic cooling / orographic precipitation")
    print("=" * 60)
    print(f"\nTarget conclusion:\n  {result.target_conclusion}\n")
    print(f"  Conclusion latency : {result.conclusion.latency_ms:.0f} ms")
    print(f"  Conclusion tokens  : {result.conclusion.tokens_used}")
    print(f"  Decomp latency     : {result.decomposition.latency_ms:.0f} ms")
    print(f"  Decomp tokens      : {result.decomposition.tokens_used}")
    print(f"\nComponents ({len(result.required_components)}):")
    _print_components(result.required_components)

    warns, hp, ht = _assert_components_valid(result.required_components, "Case 2")

    # Mastery preseeding hard assertions
    for c in result.required_components:
        cl = c.concept.lower()
        should_be_preseeded = any(
            cl in known.lower() or known.lower() in cl
            for known in assumed_known
        )
        ht += 1
        if should_be_preseeded:
            assert c.mastery == 0.5, (
                f"[Case 2] HARD FAIL: '{c.concept}' should have mastery=0.5 (in assumed_known), got {c.mastery}"
            )
            hp += 1
        else:
            assert c.mastery == 0.0, (
                f"[Case 2] HARD FAIL: '{c.concept}' should have mastery=0.0, got {c.mastery}"
            )
            hp += 1

    warns += _warn_if_premature_jargon(
        result.required_components, assumed_known,
        ["adiabatic cooling", "orographic precipitation", "adiabatic lapse rate"],
        "Case 2",
    )
    return hp, ht, warns


# ---------------------------------------------------------------------------
# Case 3: "Why is the sky blue because of the ocean?" — misconception
# ---------------------------------------------------------------------------

def case_3() -> tuple[int, int, list[str]]:
    question = "Why is the sky blue because of the ocean?"
    analysis = _make_analysis(
        question_type=QuestionType.WHY,
        question_class=QuestionClass.MISCONCEPTION,
        operative="Why is",
        prior_knowledge=PriorKnowledge.NOVICE,
    )

    result = run_decomposition(question, analysis)

    print("=" * 60)
    print("CASE 3 — Misconception: sky blue because of the ocean?")
    print("=" * 60)
    print(f"\nTarget conclusion:\n  {result.target_conclusion}\n")
    print(f"  Conclusion latency : {result.conclusion.latency_ms:.0f} ms")
    print(f"  Conclusion tokens  : {result.conclusion.tokens_used}")
    print(f"  Decomp latency     : {result.decomposition.latency_ms:.0f} ms")
    print(f"  Decomp tokens      : {result.decomposition.tokens_used}")
    print(f"\nComponents ({len(result.required_components)}):")
    _print_components(result.required_components)

    # Hard: pipeline must return a valid result
    ht = 1
    hp = 0
    assert isinstance(result.required_components, list), "[Case 3] HARD FAIL: required_components not a list"
    hp += 1

    warns, sub_hp, sub_ht = _assert_components_valid(result.required_components, "Case 3")
    hp += sub_hp
    ht += sub_ht

    # Heuristic: ocean should NOT appear as the causal mechanism
    ocean_in_statements = any(
        "ocean" in c.statement.lower() and "not" not in c.statement.lower()
        for c in result.required_components
    )
    if ocean_in_statements:
        warns.append("Heuristic: ocean appears as positive causal claim — model may not have corrected the misconception")

    return hp, ht, warns


# ---------------------------------------------------------------------------
# Case 4: "Why is the sky blue?" — novice, discoverability ordering
# ---------------------------------------------------------------------------

def case_4() -> tuple[int, int, list[str]]:
    question = "Why is the sky blue?"
    analysis = _make_analysis(
        question_type=QuestionType.WHY,
        question_class=QuestionClass.KNOWN_PHENOMENON,
        operative="Why is",
        prior_knowledge=PriorKnowledge.NOVICE,
    )

    result = run_decomposition(question, analysis)

    print("=" * 60)
    print("CASE 4 — Why is the sky blue?")
    print("=" * 60)
    print(f"\nTarget conclusion:\n  {result.target_conclusion}\n")
    print(f"  Conclusion latency : {result.conclusion.latency_ms:.0f} ms")
    print(f"  Conclusion tokens  : {result.conclusion.tokens_used}")
    print(f"  Decomp latency     : {result.decomposition.latency_ms:.0f} ms")
    print(f"  Decomp tokens      : {result.decomposition.tokens_used}")
    print(f"\nComponents ({len(result.required_components)}):")
    _print_components(result.required_components)

    warns, hp, ht = _assert_components_valid(result.required_components, "Case 4")
    warns += _warn_if_premature_jargon(
        result.required_components, [],
        ["rayleigh scattering", "rayleigh", "electric dipole", "dipole oscillation"],
        "Case 4",
    )
    return hp, ht, warns


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    all_hard_pass = 0
    all_hard_total = 0
    all_warns: list[str] = []

    for case_fn in (case_1, case_2, case_3, case_4):
        hp, ht, warns = case_fn()
        all_hard_pass += hp
        all_hard_total += ht
        all_warns.extend(warns)

    print()
    print("===========================")
    print(f"Hard assertions: {all_hard_pass}/{all_hard_total} PASS")
    warn_pass = sum(1 for w in all_warns if "WARN" not in w.upper() or True)
    print(f"Heuristic checks: {len(all_warns) - len([w for w in all_warns if 'WARN' in w.upper()])} PASS | {len(all_warns)} WARN")
    for w in all_warns:
        print(f"  WARN: {w}")
    print("===========================")

    if all_hard_pass < all_hard_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
