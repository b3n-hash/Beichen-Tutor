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
        f"You are an expert in {DOMAIN} producing an internal answer for a tutoring system.\n"
        "This answer will be decomposed into teaching components — it is never shown to the learner directly.\n"
        "Be accurate and concise. Include the key mechanism(s), cause(s), and consequence(s). 2–4 sentences maximum."
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
- "why"  → causal chain: prerequisite → mechanism → consequence
- "how"  → process / sequence: step 1 → step 2 → step 3
- "when" / "where" → conditional: condition → trigger → outcome"""

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
        f"You are a pedagogical analyst decomposing a {DOMAIN} explanation into teachable components.\n"
        "Respond only with a valid JSON array. No explanation, no preamble, no markdown fences."
    )
    user = (
        "Decompose this conclusion into the minimal ordered set of conceptual premises "
        "a learner must understand.\n\n"
        f'Question: "{question}"\n'
        f'Question type: "{question_type}"\n'
        f'Target conclusion: "{target_conclusion}"\n\n'
        f"Ordering template:\n{_ORDERING_TEMPLATE}\n\n"
        f"Return a JSON array where each object has exactly two fields:\n{_JSON_SCHEMA}\n\n"
        "Rules:\n"
        "- Minimum 2 items, maximum 6\n"
        "- Order by prerequisite dependency — earlier items must be understood before later ones\n"
        "- Do not include concepts derivable from earlier items without new information\n"
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
