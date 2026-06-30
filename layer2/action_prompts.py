"""Layer 2 — Action injection strings.

Turns a PolicyDecision into the system message injected before the LLM turn.
This is Layer 2 behaviour, kept out of app.py (which owns only UI concerns).
"""

from __future__ import annotations

from layer2.decision_policy import (
    ACTION_ADVANCE,
    ACTION_BRIDGE,
    ACTION_DISCONFIRM,
    ACTION_EVIDENCE,
    ACTION_FALLBACK,
    ACTION_INQUIRY,
    PolicyDecision,
)
from layer2.models import InquirySession


def build_injection(decision: PolicyDecision, session: InquirySession) -> str:
    """Build the system message injected before the LLM turn."""
    cc = session.current_component

    if decision.action == ACTION_INQUIRY:
        if cc is None:
            return "[DECISION: INQUIRY] No specific component is active — elicit the Learner's reasoning openly."
        return (
            f"[DECISION: INQUIRY] Keep eliciting the Learner's own reasoning about '{cc.concept}'. "
            f"Ask an open question that surfaces their understanding of: {cc.statement}. Do not state the answer."
        )

    if decision.action == ACTION_EVIDENCE:
        preferred = decision.context.get("preferred_type", "empirical")
        return (
            f"[DECISION: EVIDENCE] The Learner grasps '{cc.concept}' but has not grounded it. Introduce "
            f"{preferred} evidence that helps them justify: {cc.statement}. Then ask them to connect it back "
            f"to their reasoning."
        )

    if decision.action == ACTION_BRIDGE:
        return (
            f"[DECISION: BRIDGE] The Learner can justify reasoning but lacks comprehension of '{cc.concept}'. "
            f"Build a bridge from what they have already grounded toward the concept itself: {cc.statement}."
        )

    if decision.action == ACTION_ADVANCE:
        nxt = decision.context.get("next_component")
        if nxt is None:
            return (
                "[DECISION: ADVANCE] All components are covered. Offer the Learner the chance to state their "
                "overall conclusion (the conclude off-ramp). Do not state it for them."
            )
        covered_concept = cc.concept if cc is not None else "the current component"
        return (
            f"[DECISION: ADVANCE] '{covered_concept}' is covered. Transition the Learner to the next component: "
            f"'{nxt.concept}' — {nxt.statement}."
        )

    if decision.action == ACTION_DISCONFIRM:
        focus = f" Focus on '{cc.concept}'." if cc is not None else ""
        return (
            "[DECISION: DISCONFIRM] The Learner's hypothesis is false. Guide them to disconfirm it themselves "
            "through questions and evidence — never tell them outright that they are wrong." + focus
        )

    if decision.action == ACTION_FALLBACK:
        # Delegate to the canonical rung wording defined in app.py — do not duplicate.
        from app import FALLBACK_INJECTIONS  # lazy import avoids a circular dependency

        rung = decision.rung or decision.context.get("rung", 0)
        attempts = cc.attempts if cc is not None else 0
        template = FALLBACK_INJECTIONS.get(rung)
        if template is None:
            return ""
        return template.format(n=attempts)

    return ""  # unknown action — inject nothing
