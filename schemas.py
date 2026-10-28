"""Serializable data structures shared by the workflow, retrieval and risk layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}


@dataclass
class ManualDocument:
    source_id: str
    model: str
    code: str | None
    phenomenon: str
    causes: list[str]
    steps: list[str]
    warning: str
    manual_risk: str
    section_index: int
    start_line: int
    end_line: int
    source_path: str
    categories: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        code = self.code or "无代码"
        return f"{code}｜{self.phenomenon}"

    @property
    def citation(self) -> str:
        return (
            f"{self.source_path} · 第{self.start_line}-{self.end_line}行"
            f" · {self.code or '无代码'} · {self.phenomenon}"
        )

    def to_brief(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "code": self.code,
            "title": self.phenomenon,
            "manual_risk": self.manual_risk,
            "citation": self.citation,
        }


@dataclass
class RiskSignal:
    category: str
    label: str
    evidence: str
    source: str = "keyword"
    negated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    level: str
    signals: list[RiskSignal] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    llm_used: bool = False
    llm_notes: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)

    @property
    def level_label(self) -> str:
        return RISK_LABELS.get(self.level, self.level)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "level_label": self.level_label,
            "signals": [signal.to_dict() for signal in self.signals],
            "reasons": self.reasons,
            "rule_ids": self.rule_ids,
            "llm_used": self.llm_used,
            "llm_notes": self.llm_notes,
        }


@dataclass
class RetrievedCandidate:
    document: ManualDocument
    raw_score: float = 0.0
    bm25: float = 0.0
    cosine: float = 0.0
    coverage: float = 0.0
    category_match: float = 0.0
    code_match: bool = False
    model_match: bool = False
    margin: float = 0.0
    confidence: float = 0.0
    applicable: bool = False
    validation: list[str] = field(default_factory=list)
    failure_code: str | None = None
    failure_reason: str | None = None

    def to_brief(self) -> dict[str, Any]:
        return {
            **self.document.to_brief(),
            "confidence": round(self.confidence, 3),
            "applicable": self.applicable,
            "category_match": round(self.category_match, 3),
            "code_match": self.code_match,
            "validation": self.validation,
            "failure_code": self.failure_code,
            "retrieval_score": round(self.raw_score, 4),
        }


@dataclass
class SearchResult:
    query: str
    candidates: list[RetrievedCandidate] = field(default_factory=list)
    winner: RetrievedCandidate | None = None
    confidence: float = 0.0
    failure_code: str | None = None
    failure_reason: str | None = None
    model_filter: str | None = None

    @property
    def ok(self) -> bool:
        return self.winner is not None and self.failure_code is None


@dataclass
class TurnResult:
    action: str  # ask | diagnose | guide | handoff | resolved
    message: str
    session_id: str
    stage_trace: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    risk: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    ticket: dict[str, Any] | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
