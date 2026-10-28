"""Risk-signal extraction and deterministic risk classification.

The LLM (when configured) is deliberately limited to *extracting* categories
and verbatim evidence.  The final level is assigned here by a small, testable
rule engine.  Keyword extraction always runs as a second, independent guardrail.
"""

from __future__ import annotations

import re
from typing import Protocol

from schemas import RiskAssessment, RiskSignal
from taxonomy import compact_text, is_negated, normalize_text


RISK_CATEGORY_LABELS = {
    "smoke": "冒烟/烟雾",
    "fire": "明火/着火",
    "burning_smell": "焦味/烧焦气味",
    "spark": "火花/电弧",
    "electric_shock": "漏电/触电/麻电",
    "power_cord_damage": "电源线或插头破损",
    "liquid": "液体进入设备",
    "short_circuit": "短路",
    "electrical_pop": "爆响/电弧声",
    "abnormal_heat": "异常发热",
    "abnormal_noise": "异响",
    "vibration": "震动",
    "physical_damage": "明显机械/外壳损伤",
}

HIGH_RISK_CATEGORIES = frozenset(
    {
        "smoke",
        "fire",
        "burning_smell",
        "spark",
        "electric_shock",
        "power_cord_damage",
        "liquid",
        "short_circuit",
        # Confirmed product decision: abnormal heat is safety-escalated.
        "abnormal_heat",
    }
)
MEDIUM_RISK_CATEGORIES = frozenset({"electrical_pop", "physical_damage"})

ALL_RISK_CATEGORIES = tuple(RISK_CATEGORY_LABELS.keys())

_KEYWORD_SPECS: dict[str, list[str]] = {
    "smoke": [r"冒烟", r"冒出.{0,4}烟", r"白烟", r"黑烟", r"烟雾", r"起烟"],
    "fire": [r"着火", r"起火", r"明火", r"火焰", r"烧起来", r"着火了"],
    "burning_smell": [
        r"烧焦",
        r"焦味",
        r"烧糊",
        r"糊味",
        r"烧焦味",
        r"电线.{0,4}烧",
        r"有东西.{0,4}烧",
    ],
    "spark": [r"火花", r"电弧", r"打火", r"闪火花"],
    "electric_shock": [
        r"漏电",
        r"触电",
        r"电击",
        r"手麻",
        r"麻了一下",
        r"触电感",
        r"外壳.{0,4}有电",
        r"机壳.{0,4}有电",
        r"摸.{0,6}麻",
    ],
    "power_cord_damage": [
        r"电源线.{0,4}破损",
        r"电线.{0,4}破损",
        r"线皮.{0,4}破损",
        r"线缆.{0,4}破损",
        r"电线.{0,4}裸露",
        r"插头.{0,4}(?:烧|熔化|变形)",
        r"插座.{0,4}(?:烧|冒烟|熔化)",
    ],
    "liquid": [r"进水", r"洒了水", r"水洒", r"泡水", r"漏水", r"液体.{0,4}进入"],
    "short_circuit": [r"短路"],
    "electrical_pop": [r"啪.{0,2}响", r"砰.{0,2}响", r"爆响", r"爆裂声", r"电弧声"],
    "abnormal_heat": [r"很烫", r"发烫", r"烫手", r"过热", r"异常高温", r"温度很高"],
    "abnormal_noise": [
        r"异响",
        r"咔咔",
        r"噪音",
        r"嗡嗡",
        r"响个不停",
        r"一直响",
        r"响了一声",
    ],
    "vibration": [r"震动", r"振动", r"抖动", r"晃个不停", r"抖得"],
    "physical_damage": [r"外壳.{0,4}破", r"裂开", r"摔坏", r"严重变形", r"烧黑"],
}

_COMPILED_SPECS = {
    category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for category, patterns in _KEYWORD_SPECS.items()
}

_DEAD_OR_ELECTRICAL_CONTEXT = re.compile(
    r"(?:没反应|不亮|停机|黑屏|无法启动|停止运行|跳闸|坏了|不能用了|没动静)",
    re.IGNORECASE,
)


def _evidence_around(text: str, start: int, end: int) -> str:
    return text[max(0, start - 14) : min(len(text), end + 14)].strip()


def _llm_evidence_is_negated(
    normalized_source: str, normalized_evidence: str, category: str
) -> bool:
    """Detect a verbatim quote that is positive in form but negated in context.

    LLM output normally marks negation itself, but a model can quote "冒烟"
    from "没有冒烟" and report ``negated=false``.  We re-check the quote and
    every occurrence of it in the source before trusting the signal.
    """

    patterns = _COMPILED_SPECS.get(category, [])
    if not patterns or not normalized_evidence:
        return False

    positions: list[int] = []
    cursor = 0
    while True:
        found = normalized_source.find(normalized_evidence, cursor)
        if found < 0:
            break
        positions.append(found)
        cursor = found + 1

    saw_category_term = False
    for pattern in patterns:
        for match in pattern.finditer(normalized_evidence):
            saw_category_term = True
            negated_in_evidence = is_negated(normalized_evidence, match.start())
            negated_in_source = bool(positions) and all(
                is_negated(normalized_source, position + match.start())
                for position in positions
            )
            if not negated_in_evidence and not negated_in_source:
                return False
    return saw_category_term


class RiskSignalExtractor(Protocol):
    name: str

    def extract(self, text: str) -> tuple[list[RiskSignal], list[str]]:
        """Return signals plus diagnostic notes. Must not assign a final level."""


class KeywordRiskExtractor:
    """High-recall deterministic extractor.  Negated mentions are discarded."""

    name = "keyword"

    def extract(self, text: str) -> list[RiskSignal]:
        normalized = normalize_text(text)
        signals: list[RiskSignal] = []
        seen: set[tuple[str, str]] = set()

        for category, patterns in _COMPILED_SPECS.items():
            for pattern in patterns:
                for match in pattern.finditer(normalized):
                    if is_negated(normalized, match.start()):
                        continue
                    evidence = _evidence_around(normalized, match.start(), match.end())
                    key = (category, compact_text(evidence))
                    if key in seen:
                        continue
                    seen.add(key)
                    signals.append(
                        RiskSignal(
                            category=category,
                            label=RISK_CATEGORY_LABELS[category],
                            evidence=evidence,
                            source="keyword",
                        )
                    )
        return signals


class RiskRuleEngine:
    """Deterministic policy.  LLM output never bypasses these rules."""

    def evaluate(
        self,
        text: str,
        signals: list[RiskSignal],
        *,
        llm_used: bool = False,
        llm_notes: list[str] | None = None,
    ) -> RiskAssessment:
        normalized = normalize_text(text)
        positive = [signal for signal in signals if not signal.negated]
        categories = {signal.category for signal in positive}
        reasons: list[str] = []
        rule_ids: list[str] = []

        high_hits = sorted(categories & HIGH_RISK_CATEGORIES)
        if high_hits:
            for category in high_hits:
                reasons.append(f"检测到{RISK_CATEGORY_LABELS[category]}")
            rule_ids.append("RISK-HIGH-DIRECT")
            level = "high"
        elif {"abnormal_noise", "vibration"} <= categories:
            reasons.append("检测到异响并伴随震动，可能涉及内部机械或运动部件异常")
            rule_ids.append("RISK-HIGH-NOISE-VIBRATION")
            level = "high"
        elif "electrical_pop" in categories and _DEAD_OR_ELECTRICAL_CONTEXT.search(normalized):
            reasons.append("爆响/电弧声后设备停止响应，可能涉及内部电气故障")
            rule_ids.append("RISK-HIGH-POP-AND-DEAD")
            level = "high"
        elif categories & MEDIUM_RISK_CATEGORIES:
            for category in sorted(categories & MEDIUM_RISK_CATEGORIES):
                reasons.append(f"检测到{RISK_CATEGORY_LABELS[category]}")
            rule_ids.append("RISK-MEDIUM-SIGNAL")
            level = "medium"
        else:
            level = "low"
            rule_ids.append("RISK-LOW-NO-SIGNAL")
            reasons.append("未发现明确的高风险安全信号")

        return RiskAssessment(
            level=level,
            signals=positive,
            reasons=reasons,
            llm_used=llm_used,
            llm_notes=llm_notes or [],
            rule_ids=rule_ids,
        )


class RiskEngine:
    def __init__(self, llm_extractor: RiskSignalExtractor | None = None):
        self.keyword_extractor = KeywordRiskExtractor()
        self.rule_engine = RiskRuleEngine()
        self.llm_extractor = llm_extractor

    def assess_keywords(self, text: str) -> RiskAssessment:
        signals = self.keyword_extractor.extract(text)
        return self.rule_engine.evaluate(text, signals, llm_used=False)

    def assess(self, text: str, *, use_llm: bool = True) -> RiskAssessment:
        keyword_signals = self.keyword_extractor.extract(text)
        llm_signals: list[RiskSignal] = []
        notes: list[str] = []
        llm_used = False

        if use_llm and self.llm_extractor is not None:
            try:
                extracted, notes = self.llm_extractor.extract(text)
                llm_used = True
            except Exception as exc:  # optional enhancement must never block triage
                notes = [f"LLM 风险信号提取失败，已使用关键词兜底：{type(exc).__name__}"]
                extracted = []
            llm_signals, validation_notes = self._validate_llm_signals(text, extracted)
            notes.extend(validation_notes)
        elif use_llm:
            notes.append("未配置 LLM，风险信号仅由关键词通道提取")

        merged = self._merge_signals(keyword_signals, llm_signals)
        return self.rule_engine.evaluate(
            text, merged, llm_used=llm_used, llm_notes=notes
        )

    @staticmethod
    def _validate_llm_signals(
        text: str, signals: list[RiskSignal]
    ) -> tuple[list[RiskSignal], list[str]]:
        """Drop hallucinated or unsupported LLM signals.

        The model must quote the user verbatim.  Evidence validation makes the
        extraction auditable and prevents a fabricated signal from escalating
        a case.
        """

        normalized_source = normalize_text(text)
        compact_source = compact_text(text)
        valid: list[RiskSignal] = []
        notes: list[str] = []
        for signal in signals:
            if signal.category not in ALL_RISK_CATEGORIES:
                notes.append(f"忽略未知风险类别：{signal.category}")
                continue
            normalized_evidence = normalize_text(signal.evidence)
            evidence = compact_text(signal.evidence)
            if not evidence or evidence not in compact_source:
                notes.append(f"忽略无原文依据的 LLM 信号：{signal.category}")
                continue
            if signal.negated:
                continue
            if _llm_evidence_is_negated(
                normalized_source, normalized_evidence, signal.category
            ):
                notes.append(f"忽略否定语境的 LLM 信号：{signal.category}")
                continue
            signal.label = RISK_CATEGORY_LABELS.get(signal.category, signal.label)
            signal.source = "llm"
            valid.append(signal)
        return valid, notes

    @staticmethod
    def _merge_signals(
        keyword_signals: list[RiskSignal], llm_signals: list[RiskSignal]
    ) -> list[RiskSignal]:
        merged: list[RiskSignal] = []
        seen: set[tuple[str, str]] = set()
        for signal in [*keyword_signals, *llm_signals]:
            key = (signal.category, compact_text(signal.evidence))
            if key in seen:
                continue
            seen.add(key)
            merged.append(signal)
        return merged
