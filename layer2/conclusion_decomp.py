"""Layer 2 — Conclusion Decomposition.

Two-step pipeline:
  1. generate_target_conclusion — single LLM call producing an internal grounded answer.
  2. decompose_conclusion        — single LLM call breaking that answer into ordered
                                   Component objects for the Decision Policy to traverse.

All domain dataclasses/enums are imported from layer2.models.
Result wrappers (ConclusionResult, DecompositionResult, DecompositionPipelineResult)
are output types local to this module.

Public API:
  run_decomposition(question, question_analysis) -> DecompositionPipelineResult

Internal helpers (_generate_target, _decompose, _validate_items, _build_components)
are prefixed with underscore and not intended for external use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL
from layer2.models import (
    Component,
    QuestionAnalysis,
    QuestionClass,
    QuestionType,
)
from layer2.question_analysis import DOMAIN

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ---------------------------------------------------------------------------
# Result wrappers
# ---------------------------------------------------------------------------

@dataclass
class ConclusionResult:
    target_conclusion: str
    raw_response: str
    latency_ms: float
    tokens_used: int


@dataclass
class DecompositionResult:
    components: list[Component]
    raw_response: str
    latency_ms: float
    tokens_used: int


@dataclass
class DecompositionPipelineResult:
    conclusion: ConclusionResult
    decomposition: DecompositionResult

    @property
    def target_conclusion(self) -> str:
        return self.conclusion.target_conclusion

    @property
    def required_components(self) -> list[Component]:
        return self.decomposition.components


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_items(items: list[dict]) -> str | None:
    """Return an error string if items fail validation, else None."""
    if not (2 <= len(items) <= 6):
        return f"Expected 2–6 items, got {len(items)}"
    seen: set[str] = set()
    for i, item in enumerate(items):
        if not isinstance(item.get("concept"), str) or not item["concept"].strip():
            return f"Item {i}: missing or empty 'concept'"
        if not isinstance(item.get("statement"), str) or not item["statement"].strip():
            return f"Item {i}: missing or empty 'statement'"
        key = item["concept"].strip().lower()
        if key in seen:
            return f"Duplicate concept: '{item['concept']}'"
        seen.add(key)
    return None


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------

def _build_components(
    items: list[dict],
    assumed_known_concepts: list[str],
) -> list[Component]:
    components: list[Component] = []
    for index, item in enumerate(items):
        concept = item["concept"].strip()
        mastery = 0.5 if any(
            concept.lower() in known.lower() or known.lower() in concept.lower()
            for known in assumed_known_concepts
        ) else 0.0
        components.append(Component(
            id=index,
            concept=concept,
            statement=item["statement"].strip(),
            mastery=mastery,
            groundedness=0.0,
            covered=False,
            attempts=0,
            evidence_used=[],
        ))
    return components


# ---------------------------------------------------------------------------
# Internal LLM helpers
# ---------------------------------------------------------------------------

def _generate_target(question: str, question_analysis: QuestionAnalysis) -> ConclusionResult:
    """Single LLM call. Returns ConclusionResult."""
    system = (
        f"You are a {DOMAIN} tutor preparing an internal answer that will later be broken into a sequence of "
        "discoveries for a novice learner doing guided inquiry — it is never shown to the learner directly.\n"
        "Produce the simplest scientifically sufficient explanation of the phenomenon: preserve full causal "
        "correctness, but do not optimise for encyclopaedic completeness or expert depth.\n"
        "Introduce technical terminology only where it is necessary to name the mechanism, or where it follows "
        "naturally from a concept you've just explained in plain language — never as the entry point.\n"
        "Prefer observable effects and everyday causal reasoning over abstract or derived concepts; lead with "
        "what a careful observer would notice, not with the deepest underlying theory.\n"
        "Be concise: 2–4 sentences maximum, including the key mechanism(s), cause(s), and consequence(s)."
    )
    user = (
        f'Question: "{question}"\n'
        f'Operative: "{question_analysis.operative}"\n'
        f'Question type: "{question_analysis.question_type}"\n\n'
        "Produce the target conclusion."
    )

    t0 = time.perf_counter()
    completion = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    raw = completion.choices[0].message.content.strip()
    tokens = completion.usage.total_tokens if completion.usage else 0

    return ConclusionResult(
        target_conclusion=raw,
        raw_response=raw,
        latency_ms=latency_ms,
        tokens_used=tokens,
    )


_ORDERING_TEMPLATE = """\
- "why"  → discovery chain: observation/pattern → mechanism → abstract principle → (terminology, if it earns its place) → broader consequence
- "how"  → process / sequence a learner could trace step by step: step 1 → step 2 → step 3
- "when" / "where" → conditional a learner could notice and test: condition → trigger → outcome"""

_JSON_SCHEMA = """\
[
  {
    "concept": "...",
    "statement": "..."
  }
]"""


def _decompose(
    question: str,
    question_type: QuestionType | None,
    target_conclusion: str,
    assumed_known_concepts: list[str],
) -> DecompositionResult:
    """Single LLM call with one automatic retry on validation failure.
    Raises ValueError if both attempts fail validation.
    """
    system = (
        f"You are a pedagogical analyst sequencing a {DOMAIN} explanation into a guided-discovery inquiry for "
        "a novice learner — not into an expert's logical dependency graph.\n"
        "Optimise for discoverability, inferability, and cognitive continuity: the sequence a good human tutor "
        "would lead a curious learner through, not the deepest or most theoretically complete ordering.\n"
        "Respond only with a valid JSON array. No explanation, no preamble, no markdown fences."
    )
    known_str = ", ".join(assumed_known_concepts) if assumed_known_concepts else "none"
    user = (
        "Sequence this conclusion into the smallest smoothly teachable sequence of conceptual discoveries that "
        "would let a novice learner independently reconstruct it through guided inquiry — not the minimal set "
        "of logical premises an expert would list, and not compressed further than the inquiry can stay smooth "
        "and natural.\n\n"
        f'Question: "{question}"\n'
        f'Question type: "{question_type}"\n'
        f'Target conclusion: "{target_conclusion}"\n'
        f"Concepts the learner already knows (may be used freely, including early): {known_str}\n\n"
        f"Ordering template:\n{_ORDERING_TEMPLATE}\n\n"
        "For every candidate component, ask internally:\n"
        "- Could a curious novice plausibly infer or notice this next, given what comes before it?\n"
        "- Is there an observable phenomenon that naturally precedes this concept?\n"
        "- Does this concept require specialist vocabulary the learner hasn't been given yet?\n"
        "- Would delaying this concept reduce cognitive load without breaking causal correctness?\n"
        "- Would successfully understanding this component make the next one feel like a natural question "
        "rather than a new topic? Prefer sequences where each answer generates the next question, over a list "
        "of isolated facts.\n"
        "- Could a single clear Socratic question reasonably lead the learner to discover this component? If "
        "you cannot imagine one, the component is probably too large or too abstract and should be split.\n"
        "- Could this be reached by extending or reusing an idea already introduced, rather than starting a "
        "fresh concept? Prefer extending and reusing ideas already introduced over introducing entirely new "
        "ones — a learner should feel each component is a continuation of the conversation so far, not a new "
        "topic.\n\n"
        "Between adjacent components, there should be no hidden inference that would reasonably require its own "
        "tutor question. If a learner would naturally ask \"why?\" or \"how?\" between two neighbouring "
        "components, insert another component instead of expecting the learner to bridge the gap unaided.\n\n"
        "A typical novice should usually be able to discover one component within roughly one to three tutor "
        "exchanges. If a component would normally require substantially longer, split it into smaller "
        "conceptual discoveries rather than expecting the learner to make several major inferences at once.\n\n"
        "The first one or two components should usually be concepts the learner could plausibly observe or "
        "infer without specialist vocabulary, not the deepest scientific explanation — sometimes the first "
        "observation is too trivial to stand alone, and the first pair forms the natural entry point. A learner "
        "should almost never need to memorise a name before understanding the underlying phenomenon — prefer "
        "teaching the idea first and introducing its technical name afterwards. Treat technical terminology as "
        "labels attached to ideas the learner has already constructed, not as the ideas themselves: it should "
        "usually appear once the learner has already built the underlying intuition, not in those opening "
        "components — unless the term is already in the learner's known concepts above.\n\n"
        "Example — prefer a chain where each component naturally raises the next question, e.g.:\n"
        "  \"the outer bend moves faster\" (→ why does that matter?) → \"the outer bank erodes\" (→ what does "
        "that cause?) → \"the bend grows\" (→ why does the flow organise itself that way?) → \"this reinforcing "
        "pattern is called helicoidal flow\"\n"
        "over a list of isolated facts:\n"
        "  \"helicoidal flow\" → \"differential erosion\" → \"meander growth\"\n\n"
        "Do not sacrifice scientific correctness for simplicity — every statement must remain fully accurate. "
        "The optimisation target is pedagogical sequencing, not dumbing down the science. When in doubt, "
        "optimise for the conversation the tutor will have, not the ontology of the subject. The output should "
        "resemble the path an excellent human teacher would guide a learner through, rather than the structure "
        "of an expert textbook or scientific paper.\n\n"
        f"Return a JSON array where each object has exactly two fields:\n{_JSON_SCHEMA}\n\n"
        "Rules:\n"
        "- Minimum 2 items, maximum 6\n"
        "- Order by the sequence a tutor would naturally help a learner discover, not by theoretical "
        "dependency depth\n"
        "- Every component should correspond to a single discoverable idea that can naturally become the focus "
        "of one investigative question.\n"
        "- Do not include concepts inferable from earlier items without new information\n"
        "- No hidden leaps: if a learner would need to ask \"why?\" or \"how?\" to get from one component to "
        "the next, insert the missing component rather than leaving the gap\n"
        "- Reuse and extend ideas already introduced before reaching for an entirely new one\n"
        "- Each statement must be a complete, standalone causal claim"
    )

    last_error: str = ""
    last_raw: str = ""
    last_latency: float = 0.0
    last_tokens: int = 0

    for attempt in range(2):
        t0 = time.perf_counter()
        completion = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        raw = completion.choices[0].message.content.strip()
        tokens = completion.usage.total_tokens if completion.usage else 0

        last_raw = raw
        last_latency = latency_ms
        last_tokens = tokens

        # Strip accidental markdown fences
        cleaned = raw
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0].strip()

        try:
            items = json.loads(cleaned)
            if not isinstance(items, list):
                raise ValueError("Response is not a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"JSON parse error: {exc}"
            if attempt == 0:
                continue
            raise ValueError(f"Decomposition failed after 2 attempts: {last_error}") from exc

        error = _validate_items(items)
        if error:
            last_error = error
            if attempt == 0:
                continue
            raise ValueError(f"Decomposition failed after 2 attempts: {last_error}")

        components = _build_components(items, assumed_known_concepts)
        return DecompositionResult(
            components=components,
            raw_response=last_raw,
            latency_ms=last_latency,
            tokens_used=last_tokens,
        )

    raise ValueError(f"Decomposition failed after 2 attempts: {last_error}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_decomposition(
    question: str,
    question_analysis: QuestionAnalysis,
) -> DecompositionPipelineResult:
    """Orchestrate conclusion generation and decomposition.

    Compound questions: decompose only the primary inquiry.
    sub_inquiries handling is a separate task — this module does not set them.

    The caller (app integration, Prompt 7) is responsible for attaching
    components to InquirySession and calling session.derive_complexity().
    """
    conclusion = _generate_target(question, question_analysis)

    decomposition = _decompose(
        question=question,
        question_type=question_analysis.question_type,
        target_conclusion=conclusion.target_conclusion,
        assumed_known_concepts=question_analysis.assumed_known_concepts,
    )

    return DecompositionPipelineResult(conclusion=conclusion, decomposition=decomposition)
