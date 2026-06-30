"""Layer 2 data model — pure data structures only.

No pipeline logic, parsing, Decision Policy, or external integration lives here;
this module defines the dataclasses and the enums their fields are typed against.

Per the implementation notes:
  * fields with a fixed set of valid values use Enum classes (not string literals);
  * created_at is stored as a datetime object internally and only serialised to
    ISO format on export/logging (see InquirySession.created_at_iso).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# --- Enumerations -------------------------------------------------------------

class Interrogative(str, Enum):
    WHY = "why"
    HOW = "how"
    WHAT = "what"
    WHEN = "when"
    WHERE = "where"


class QuestionType(str, Enum): #no WHAT as what queries arent decomposable under our framework
    WHY = "why"
    HOW = "how"
    WHEN = "when"
    WHERE = "where"


class QuestionClass(str, Enum):
    KNOWN_PHENOMENON = "known_phenomenon"
    MISCONCEPTION = "misconception"
    SPECULATION = "speculation"
    COMPOUND = "compound"
    FALSE_PREMISE = "false_premise"
    SUBJECTIVE = "subjective"
    UNANSWERABLE = "unanswerable"
    AMBIGUOUS = "ambiguous"
    OUT_OF_SCOPE = "out_of_scope"


class GateFailure(str, Enum):
    INCOMPLETE = "incomplete"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_DECOMPOSABLE = "not_decomposable"


class PriorKnowledge(str, Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class EvidenceType(str, Enum):
    ANALOGY = "analogy"
    EMPIRICAL = "empirical"
    CONTRADICTION = "contradiction"
    EXAMPLE = "example"


class HypothesisStatus(str, Enum):
    FALSE = "false"
    TRUE_UNGROUNDED = "true_ungrounded"
    TRUE_GROUNDED = "true_grounded"
    UNCLASSIFIED = "unclassified"


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --- Question analysis --------------------------------------------------------

@dataclass
class GrammarKeywords:
    interrogative: Interrogative
    auxiliary: Optional[str] = None     # e.g. "do", "does", "can"


@dataclass
class QuestionAnalysis:
    valid: bool
    gate_failed: Optional[GateFailure] = None
    redirect_response: Optional[str] = None
    question_type: Optional[QuestionType] = None
    question_class: Optional[QuestionClass] = None
    operative: Optional[str] = None
    action: Optional[str] = None
    content_keywords: list[str] = field(default_factory=list)
    grammar_keywords: Optional[GrammarKeywords] = None
    prior_knowledge_level: PriorKnowledge = PriorKnowledge.NOVICE
    assumed_known_concepts: list[str] = field(default_factory=list)


# --- Evidence -----------------------------------------------------------------

@dataclass
class Evidence:
    content: str
    source: str                         # "wikipedia:Article_Name" | "tutor_generated"
    type: EvidenceType
    component: str                      # concept name this evidence supports
    turn: int
    covered: bool = False              # learner has engaged with it
    effective: Optional[bool] = None   # did mastery/groundedness improve; set post-hoc


# --- Component ----------------------------------------------------------------

@dataclass
class Component:
    concept: str                        # node term, e.g. "mantle convection"
    statement: str                      # causal claim
    mastery: float = 0.0               # 0–1: estimated learner comprehension
    groundedness: float = 0.0          # 0–1: degree understanding is justified
    covered: bool = False              # one-way latch — never set directly; use mark_covered()
    attempts: int = 0
    evidence_used: list[str] = field(default_factory=list)   # content strings of Evidence shown
    confidence: Optional[float] = None  # tutor's confidence in its mastery estimate

    # Thresholds are passed in rather than hardcoded so they can be tuned without
    # touching the dataclass. Starting values: mastery 0.7, groundedness 0.6.
    def mark_covered(self, mastery_threshold: float = 0.7, groundedness_threshold: float = 0.6) -> bool:
        """One-way latch. Sets covered=True if both thresholds are met; never reverts.
        Returns True if covered was just set (transition), False otherwise."""
        if not self.covered and self.mastery >= mastery_threshold and self.groundedness >= groundedness_threshold:
            self.covered = True
            return True
        return False


# --- Inquiry session ----------------------------------------------------------

@dataclass
class InquirySession:
    topic_anchor: str                           # original question verbatim
    created_at: datetime = field(default_factory=datetime.now)
    question_analysis: Optional[QuestionAnalysis] = None
    target_conclusion: Optional[str] = None
    required_components: list[Component] = field(default_factory=list)
    current_component_index: int = 0
    hypothesis: Optional[str] = None
    hypothesis_status: HypothesisStatus = HypothesisStatus.UNCLASSIFIED
    hypothesis_history: list[tuple[str, HypothesisStatus, int, datetime]] = field(default_factory=list)
    # each entry: (hypothesis_text, status_at_recording, turn_number, recorded_at)
    evidence_repository: list[Evidence] = field(default_factory=list)
    sub_inquiries: list[str] = field(default_factory=list)
    complexity: Optional[Complexity] = None

    @property
    def current_component(self) -> Optional[Component]:
        if 0 <= self.current_component_index < len(self.required_components):
            return self.required_components[self.current_component_index]
        return None

    @property
    def all_covered(self) -> bool:
        return bool(self.required_components) and all(c.covered for c in self.required_components)

    @property
    def created_at_iso(self) -> str:
        """ISO serialisation of created_at, for export/logging only."""
        return self.created_at.isoformat(timespec="seconds")

    def advance_component(self) -> bool:
        """Move to the next component in sequence. Returns True if advanced, False if at end."""
        self.current_component_index += 1
        return self.current_component_index < len(self.required_components)

    def record_hypothesis(self, text: str, status: HypothesisStatus, turn: int) -> None:
        """Overwrite current hypothesis and append to history as
        (hypothesis_text, status_at_recording, turn_number, recorded_at)."""
        self.hypothesis = text
        self.hypothesis_status = status
        self.hypothesis_history.append((text, status, turn, datetime.now()))

    def derive_complexity(self) -> Complexity:
        n = len(self.required_components)
        if n <= 2:
            c = Complexity.LOW
        elif n <= 4:
            c = Complexity.MEDIUM
        else:
            c = Complexity.HIGH
        self.complexity = c
        return c
