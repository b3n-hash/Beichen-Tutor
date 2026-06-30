"""Exhaustion Fallback rung injections.

Lives in layer2 (not app.py) so both app.py and layer2.action_prompts can import
it without a circular dependency. Keyed by fallback_rung (0 injects nothing);
{n} = the attempt count.
"""

FALLBACK_INJECTIONS = {
    1: (
        "[EXHAUSTION FALLBACK — RUNG A ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung A: reframe or narrow the inquiry — abandon the current angle and try a "
        "completely different approach or a simpler sub-question. Do NOT repeat the investigative question that "
        "has already stalled."
    ),
    2: (
        "[EXHAUSTION FALLBACK — RUNG B ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung B: give a small, concrete hint — one piece of information or an analogy — "
        "then invite the Learner to reattempt. Do NOT ask the same investigative question again."
    ),
    3: (
        "[EXHAUSTION FALLBACK — RUNG C ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase. Apply Rung C: partially scaffold — supply a worked partial answer that lays out part "
        "of the mechanism, then ask the Learner to complete the remaining step themselves."
    ),
    4: (
        "[EXHAUSTION FALLBACK — RUNG D ACTIVE] The Learner has not progressed after {n} attempts in the "
        "INVESTIGATE phase, and Rungs A–C are exhausted. Apply Rung D: you are now permitted to state the "
        "mechanism or near-answer explicitly. Under Rule 1 this is the legitimate closure case, NOT a violation "
        "— providing it here is what makes the Learner's eventual restatement valid. Deliver it clearly, then "
        "invite the Learner to restate the conclusion in their own words."
    ),
}
