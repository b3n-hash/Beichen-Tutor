"""Layer 2 — Origin Question Analysis.

Runs the learner's *origin* question through a gate sequence and returns an
AnalysisResult wrapping a QuestionAnalysis. The "origin" qualifier is load-bearing:
investigative questions, follow-ups, and sub-inquiries must NOT go through this
pipeline.

All dataclasses/enums are imported from layer2.models — none are redefined here.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL
from layer2.models import (
    GateFailure,
    GrammarKeywords,
    Interrogative,
    PriorKnowledge,
    QuestionAnalysis,
    QuestionClass,
    QuestionType,
)

# --- Domain configuration -----------------------------------------------------
# The prompt/templates reference these, never hardcoded strings, so the module
# is reusable for other domains without touching prompt logic.
DOMAIN = "physical geography"
DOMAIN_EXAMPLES = "erosion, plate tectonics, weather systems, coastal features, rivers, glaciers"
EXAMPLE_QUESTION = "Why do rivers meander?"

_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# Low confidence assigned when Gate 2 fails but we've already prompted the
# learner once — we let the LLM/conversation handle narrowing instead.
_PROMPTED_AGAIN_CONFIDENCE = 0.2


# --- Redirect templates -------------------------------------------------------
# redirect_response is ALWAYS built from these — never LLM wording — so the
# template voice and the model voice do not compete.
REDIRECT_TEMPLATES = {
    "out_of_scope": (
        "This tutor focuses on {DOMAIN} — things like {DOMAIN_EXAMPLES}. "
        "A good example would be: \"{EXAMPLE_QUESTION}\""
    ),
    "not_decomposable": (
        "That has a direct answer rather than a mechanism to investigate. {redirect_reason} "
        "Try something more open, like \"{EXAMPLE_QUESTION}\""
    ),
    "incomplete": (
        "Could you give me a bit more detail? Something with a clear subject and what you want "
        "to understand works best — for example, \"{EXAMPLE_QUESTION}\""
    ),
    "subjective": (
        "That's a matter of perspective rather than a physical mechanism. {redirect_reason} "
        "I can explore related physical processes if you have a follow-up question."
    ),
    "unanswerable": (
        "{redirect_reason} I work best with physical geography questions that have an investigable mechanism."
    ),
    "ambiguous": (
        "I want to make sure I understand what you're asking — {redirect_reason} "
        "Could you rephrase or give a bit more context?"
    ),
}


@dataclass
class AnalysisResult:
    analysis: QuestionAnalysis
    raw_response: str           # raw LLM response for debugging (format may change)
    latency_ms: float           # round-trip time for the LLM call
    tokens_used: int            # total tokens (prompt + completion)


# --- Orchestrator -------------------------------------------------------------

def analyse_origin_question(question: str, incomplete_prompted_already: bool = False) -> AnalysisResult:
    """Short orchestrator: runs Gate 2 (heuristic) then Gates 1/3/4 (one LLM call),
    assembling the AnalysisResult. All logic lives in the private helpers below."""
    # Gate 2 — heuristic, no LLM, fires first.
    if not _gate2_check(question):
        if incomplete_prompted_already:
            # Already nudged once — accept and let the conversation narrow it.
            qa = QuestionAnalysis(valid=True, confidence=_PROMPTED_AGAIN_CONFIDENCE)
            return AnalysisResult(qa, raw_response="", latency_ms=0.0, tokens_used=0)
        qa = QuestionAnalysis(valid=False, gate_failed=GateFailure.INCOMPLETE)
        qa.redirect_response = _build_redirect("incomplete", None)
        return AnalysisResult(qa, raw_response="", latency_ms=0.0, tokens_used=0)

    # Gates 1, 3, 4 — single LLM classification call.
    system, user = _build_prompt(question)
    parsed, raw, latency_ms, tokens = _call_classifier(system, user)
    qa = _parse_response(parsed)

    if not parsed.get("is_domain", False):
        _fail(qa, "out_of_scope", parsed.get("redirect_reason"))
    elif not parsed.get("is_decomposable", False):
        _fail(qa, "not_decomposable", parsed.get("redirect_reason"))
    elif qa.question_class in (QuestionClass.SUBJECTIVE, QuestionClass.UNANSWERABLE, QuestionClass.AMBIGUOUS):
        _fail(qa, qa.question_class.value, parsed.get("redirect_reason"))
    else:
        qa.valid = True

    return AnalysisResult(qa, raw_response=raw, latency_ms=latency_ms, tokens_used=tokens)


# --- Private helpers ----------------------------------------------------------

_INTERROGATIVES = ("why", "how", "what", "when", "where")
_IMPERATIVES = ("explain", "describe", "tell", "discuss", "outline", "compare", "analyse", "analyze", "summarise")


def _gate2_check(question: str) -> bool:
    """Heuristic minimal-structure check (no LLM). Rejects only inputs with no
    subject, no verb and no interrogative — a single word, a name, or punctuation."""
    q = (question or "").strip()
    if not q:
        return False
    words = re.findall(r"[A-Za-z']+", q)
    if len(words) < 2:
        return False
    lower = q.lower()
    if any(re.search(rf"\b{w}\b", lower) for w in _INTERROGATIVES):
        return True
    if words[0].lower() in _IMPERATIVES:      # imperative form
        return True
    if q.endswith("?"):                        # noun-phrase question
        return True
    return len(words) >= 3                      # has enough for a subject + verb


def _build_prompt(question: str) -> tuple[str, str]:
    """Return (system_msg, user_msg) for the single classification call."""
    system = (
        f"You are a question classifier for an inquiry-based {DOMAIN} tutor.\n"
        "Respond only with valid JSON matching the schema below. No explanation, no preamble."
    )
    user = f"""Classify this question for a {DOMAIN} tutor.

Question: "{question}"

Return JSON with exactly these fields:

{{
  "is_domain": true | false,
  "is_decomposable": true | false,
  "question_class": "known_phenomenon | misconception | speculation | compound | false_premise | subjective | unanswerable | ambiguous | out_of_scope",
  "question_type": "why | how | when | where | null",
  "operative": "the main subject noun or phrase | null",
  "action": "the governing verb or action | null",
  "content_keywords": ["list of domain terms"],
  "interrogative": "why | how | what | when | where | null",
  "auxiliary": "do | does | can | will | would | null",
  "prior_knowledge_level": "novice | intermediate | advanced",
  "assumed_known_concepts": ["concepts the question vocabulary implies the learner already knows"],
  "redirect_reason": "one sentence if the question should be redirected, else null",
  "confidence": 0.0
}}

Classification rules:
- is_domain: true only for questions about {DOMAIN} processes ({DOMAIN_EXAMPLES})
- is_decomposable: true for any "why" or "how" question about a process with an underlying causal mechanism —
  EVEN IF the premise is false or the framing contains a misconception (the mechanism is still investigable).
  false ONLY for single-answer factual lookups with no mechanism (e.g. "Where is Paris?", "How tall is Everest?",
  "When did X happen?"). For reference, "{EXAMPLE_QUESTION}" IS decomposable.
- question_class definitions:
    known_phenomenon: real process, correct framing, investigable mechanism
    misconception: incorrect *explanation* of a real process ("why is the sky blue because of the ocean?")
    false_premise: assumes an event or phenomenon that does not exist ("why does the sun orbit Earth?", "why is the sky green?")
    speculation: real phenomenon but answer is uncertain or contested
    compound: contains more than one distinct investigable question
    subjective: a matter of opinion with no physical mechanism
    unanswerable: no tractable mechanism exists
    ambiguous: referent is unclear and cannot be resolved without clarification
    out_of_scope: not about {DOMAIN}
- prior_knowledge_level:
    novice: everyday vocabulary only, no domain terms
    intermediate: some domain vocabulary
    advanced: technical terminology in the question itself (e.g. "adiabatic", "orographic")
- assumed_known_concepts: concepts *named using their technical terms* in the question — the learner demonstrated awareness of them
- confidence: your confidence in the classification (0.0–1.0); lower if the question is borderline
"""
    return system, user


def _call_classifier(system: str, user: str) -> tuple[dict, str, float, int]:
    """Issue the classification call. Returns (parsed_json, raw_response, latency_ms, tokens)."""
    start = time.perf_counter()
    completion = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    raw_response = completion.choices[0].message.content or ""
    tokens = getattr(getattr(completion, "usage", None), "total_tokens", 0) or 0

    parsed = _extract_json(raw_response)
    return parsed, raw_response, latency_ms, tokens


def _extract_json(raw: str) -> dict:
    """Pull the JSON object out of a raw model response (tolerant of code fences)."""
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _parse_response(raw: dict) -> QuestionAnalysis:
    """Build the descriptive QuestionAnalysis from the parsed JSON. valid/gate_failed/
    redirect_response are decided by the orchestrator's gate logic afterwards."""
    interrogative = _to_enum(Interrogative, raw.get("interrogative"))
    grammar = None
    if interrogative is not None:
        grammar = GrammarKeywords(interrogative=interrogative, auxiliary=_clean(raw.get("auxiliary")))

    return QuestionAnalysis(
        valid=True,
        question_type=_to_enum(QuestionType, raw.get("question_type")),
        question_class=_to_enum(QuestionClass, raw.get("question_class")),
        operative=_clean(raw.get("operative")),
        action=_clean(raw.get("action")),
        content_keywords=list(raw.get("content_keywords") or []),
        grammar_keywords=grammar,
        prior_knowledge_level=_to_enum(PriorKnowledge, raw.get("prior_knowledge_level"), PriorKnowledge.NOVICE),
        assumed_known_concepts=list(raw.get("assumed_known_concepts") or []),
        confidence=_to_float(raw.get("confidence")),
    )


def _build_redirect(gate_failed: str, redirect_reason: str | None) -> str:
    """Render the learner-facing redirect from REDIRECT_TEMPLATES only."""
    key = gate_failed.value if isinstance(gate_failed, Enum) else gate_failed
    template = REDIRECT_TEMPLATES.get(key, REDIRECT_TEMPLATES["incomplete"])
    return template.format(
        DOMAIN=DOMAIN,
        DOMAIN_EXAMPLES=DOMAIN_EXAMPLES,
        EXAMPLE_QUESTION=EXAMPLE_QUESTION,
        redirect_reason=(redirect_reason or "").strip(),
    ).strip()


def _fail(qa: QuestionAnalysis, gate_key: str, redirect_reason: str | None) -> None:
    """Mark a QuestionAnalysis as a gate failure with the matching redirect."""
    qa.valid = False
    qa.gate_failed = GateFailure(gate_key)
    qa.redirect_response = _build_redirect(gate_key, redirect_reason)


# --- Small parsing utilities --------------------------------------------------

def _clean(value):
    """Normalise the JSON's null-ish strings to None."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none")):
        return None
    return value


def _to_enum(enum_cls, value, default=None):
    value = _clean(value)
    if value is None:
        return default
    try:
        return enum_cls(value)
    except ValueError:
        return default


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
