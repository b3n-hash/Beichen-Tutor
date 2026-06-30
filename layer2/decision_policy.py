"""Layer 2 — Decision Policy.

Pure Python, no LLM calls. Reads the session state and returns a typed
PolicyDecision. The policy must not mutate session state except by calling
component.mark_covered() (the one-way coverage latch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from layer2.models import HypothesisStatus, InquirySession

# --- Action constants ---------------------------------------------------------
ACTION_INQUIRY = "inquiry"
ACTION_EVIDENCE = "evidence"
ACTION_BRIDGE = "bridge"
ACTION_ADVANCE = "advance"
ACTION_FALLBACK = "fallback"
ACTION_DISCONFIRM = "disconfirm"

FALLBACK_THRESHOLD = 3

# Default evidence-type priority when none have been ruled out.
_EVIDENCE_PRIORITY = ["empirical", "analogy", "contradiction", "example"]


@dataclass
class PolicyDecision:
    action: str
    rung: int = 0                        # 0 unless ACTION_FALLBACK; 1–4 for fallback rung
    reason: str = ""                     # brief rationale, used in injection and debug logs
    context: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0              # policy's confidence in this decision (not LLM confidence)

    def __str__(self) -> str:
        rung = f" rung={self.rung}" if self.action == ACTION_FALLBACK else ""
        return f"PolicyDecision({self.action}{rung}, confidence={self.confidence:.2f}, reason={self.reason!r})"


def decide(session: InquirySession) -> PolicyDecision:
    """Read session state and return the next tutoring action.

    Does not modify session state except via component.mark_covered()."""
    # --- Step 1: prerequisites ------------------------------------------------
    if not session.required_components:
        return PolicyDecision(
            action=ACTION_INQUIRY,
            reason="components not yet available — use Layer 1 heuristic",
            confidence=0.5,
        )

    component = session.current_component
    if component is None:  # all components covered
        return PolicyDecision(
            action=ACTION_ADVANCE,
            reason="all components covered — offer conclude off-ramp",
            context={"next_component": None},
        )

    # --- Step 2: fallback override -------------------------------------------
    attempts = component.attempts
    target_rung = max(0, min(4, (attempts - 1) // FALLBACK_THRESHOLD))
    if target_rung > session.fallback_rung:
        return PolicyDecision(
            action=ACTION_FALLBACK,
            rung=target_rung,
            reason=f"attempts={attempts} crossed rung threshold",
            context={"rung": target_rung},
        )
    if session.fallback_rung > 0 and target_rung > 0:
        return PolicyDecision(
            action=ACTION_FALLBACK,
            rung=session.fallback_rung,
            reason="fallback already active",
            context={"rung": session.fallback_rung},
        )

    # --- Step 3: hypothesis disconfirmation ----------------------------------
    if session.hypothesis_status == HypothesisStatus.FALSE:
        return PolicyDecision(
            action=ACTION_DISCONFIRM,
            reason="hypothesis is false — disconfirmation required before component work",
            confidence=0.95,
        )

    # --- Step 4: mastery/groundedness matrix (6 cells, every combination) -----
    # Bridge = connect what they already have; Evidence = supply missing
    # justification; Inquiry = still exploring; Advance = endpoint.
    m, g = component.mastery, component.groundedness

    def evidence_decision(reason: str) -> PolicyDecision:
        return PolicyDecision(
            action=ACTION_EVIDENCE,
            reason=reason,
            context={"preferred_type": _preferred_evidence_type(session)},
        )

    if m >= 0.7 and g >= 0.6:
        component.mark_covered()  # the only permitted session mutation
        idx = session.current_component_index
        nxt = session.required_components[idx + 1] if idx + 1 < len(session.required_components) else None
        return PolicyDecision(
            action=ACTION_ADVANCE,
            reason="mastery and groundedness thresholds met",
            context={"next_component": nxt},
        )
    if m >= 0.7:  # g < 0.6
        return evidence_decision("mastery high but groundedness short of threshold — introduce evidence")
    if m < 0.4 and g < 0.4:
        return PolicyDecision(action=ACTION_INQUIRY, reason="low mastery and groundedness — open inquiry")
    if m < 0.4:  # g >= 0.4
        return PolicyDecision(action=ACTION_BRIDGE, reason="grounded reasoning but low comprehension — bridge")
    # 0.4 <= m < 0.7
    if g < 0.4:
        return evidence_decision("comprehension present but ungrounded — introduce evidence")
    return PolicyDecision(action=ACTION_BRIDGE, reason="partial mastery with some grounding — bridge")


def _preferred_evidence_type(session: InquirySession) -> str:
    """Pick the next evidence type for the current component: the first in the
    default priority order that has not already been tried ineffectively."""
    concept = session.current_component.concept

    def type_value(t):
        return t.value if hasattr(t, "value") else t

    # Types already tried for this component that did NOT improve understanding.
    excluded = {
        type_value(ev.type)
        for ev in session.evidence_repository
        if ev.component == concept and ev.effective is False
    }
    for evidence_type in _EVIDENCE_PRIORITY:
        if evidence_type not in excluded:
            return evidence_type
    return _EVIDENCE_PRIORITY[0]  # everything excluded — fall back to first
