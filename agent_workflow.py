"""State-machine orchestration for LX-2000 after-sales triage.

The workflow is intentionally a code-controlled machine, not a free-running
agent loop:

A input -> B sufficiency -> (C follow-up / D risk) -> F retrieval -> G gate
-> H grounded answer or I human handoff.

A grounded diagnosis is not terminal: it opens a bounded feedback loop
(H -> A).  Negative feedback ("没提示纸张错", "还是不行") re-enters check_info,
excludes the rejected chapter and re-retrieves an alternative.  Only an
explicit resolution, a handoff, or feedback-budget exhaustion ends the session.

Safety-critical keyword matches can make the case "sufficient for escalation"
so the assistant never delays a smoke/shock case just to collect a model number.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rag_engine import (
    FEEDBACK_MIN_CONFIDENCE,
    FEEDBACK_MIN_MARGIN,
    ManualIndex,
)
from risk_engine import RiskEngine, RiskSignalExtractor
from schemas import (
    ManualDocument,
    RiskAssessment,
    RetrievedCandidate,
    SearchResult,
    TurnResult,
)
from taxonomy import (
    MAX_FAULT_ASKS,
    MAX_FEEDBACK_ASKS,
    MAX_MODEL_ASKS,
    PRINT_INTERRUPT_HINT_CODE,
    SUPPORTED_MODEL,
    SYMPTOM_LABELS,
    ExtractedInfo,
    FeedbackAnalysis,
    analyze_feedback,
    compact_text,
    detect_print_interrupt,
    extract_error_codes,
    extract_info,
    has_actionable_symptom,
    merge_symptom_categories,
    normalize_text,
)


REASON_TEXT = {
    "explicit_human_request": "用户明确要求人工处理",
    "high_risk": "检测到冒烟、焦味、漏电、火花等高风险安全信号",
    "unsupported_model": "设备型号超出本手册覆盖范围",
    "model_conflict": "用户描述中出现互相冲突的型号信息",
    "insufficient_info": "型号或故障现象达到各自追问上限后仍无足够信息",
    "code_not_found": "用户给出的故障代码在手册中不存在",
    "code_symptom_conflict": "错误代码与故障现象指向不同章节，存在冲突",
    "feedback_unresolved": "用户反馈上一轮诊断不成立，重新排查后仍无法定位其他安全章节",
    "not_applicable": "没有通过型号与故障现象适用性校验的手册章节",
    "low_confidence": "检索置信度低于安全阈值",
    "ambiguous": "存在多个接近的候选章节，无法确定唯一适用内容",
    "no_manual_steps": "匹配章节没有可安全执行的用户级排查步骤",
    "unsafe_manual_steps": "手册排查步骤涉及拆机、接触电路或拆卸定影组件等专业操作",
    "unsupported_manual": "手册数据不可用",
}

# Queue mapping is an initial heuristic and is intentionally left for
# alignment with the real after-sales system (see README "待对齐").
_QUEUE_BY_REASON = {
    "high_risk": "safety_escalation",
    "unsafe_manual_steps": "safety_escalation",
    "explicit_human_request": "general_after_sales",
    "unsupported_model": "product_scope_review",
    "model_conflict": "product_scope_review",
}

_COST_OR_PROMISE = re.compile(
    r"免费|保修|包修|价格|报价|收费|费用|赔付|赔偿|退款|\d+\s*元|保证|承诺|百分百|一定能"
)

# User-level steps must not instruct the user to open the chassis, touch live
# circuits or remove fuser/heating/mechanical assemblies.
_UNSAFE_STEP_ACTION = re.compile(
    r"(?:"
    r"拆机"
    r"|短接"
    r"|(?:拆开|拆下|拆卸|拆除|拆解|取下|移除|打开)"
    r".{0,8}(?:外壳|机壳|电源模块|电源板|电路板|高压板|定影|加热组件|后盖|侧盖|传动机构|电机|齿轮)"
    r"|(?:接触|触碰|触摸|测量|检测|检查)"
    r".{0,8}(?:带电|电路|电源板|高压|定影|加热)"
    r"|(?:将手|把手|伸手)"
    r".{0,8}(?:伸入|放入|进入)"
    r")",
    re.IGNORECASE,
)
_STEP_NEGATION = re.compile(
    r"(?:禁止|严禁|不要|不得|不可|切勿|勿|避免)[^，。；！？,.;!?]{0,8}$"
)
_STEP_PROFESSIONAL = re.compile(
    r"(?:联系|安排|由|请|交由|委托)[^，。；！？,.;!?]{0,12}$"
)
_CODE_CORRECTION = re.compile(
    r"看错|记错|不是|没有代码|没代码|不显示|无代码|其实是|应该是|更正|改成|重新看"
)
# A short confirmation at the start of a turn, e.g. "是", "对，是的",
# "是，打印".  It must be followed by punctuation/space/end so "是不是..."
# is not treated as confirmation.
_AFFIRMATIVE_REPLY = re.compile(
    r"^\s*(?:是的|对的|是|对|嗯|没错|有的|有|好的|好)"
    r"(?:\s*[，,。.!！]|\s+|$)"
)
# When the previous answer is denied, the retrieval query needs the *positive*
# alternative vocabulary; merely excluding E08 is not enough for BM25 to find
# E19.  Keys are the diagnosed code, with category fallbacks below.
_FEEDBACK_QUERY_EXPANSIONS = {
    "E08": "打印中断 打到一半停住 纸张皱 卡下一张",
    "E02": "卡纸 纸卡 卡住 拉不出来 皱 进纸",
    "E10": "异响 震动 咔咔 嗡嗡",
}
_FEEDBACK_CATEGORY_EXPANSIONS = {
    "paper_mismatch": "打印中断 打到一半停住 纸张皱 卡下一张",
    "paper_jam": "卡纸 纸卡 卡住 拉不出来 皱 进纸",
    "noise": "异响 震动 咔咔 嗡嗡",
}


def steps_require_professional(steps: list[str]) -> bool:
    """True if a manual step asks the user to do a professional-only action.

    Negated safety instructions ("不要拆机") and instructions to contact a
    professional are safe and therefore ignored.
    """

    for step in steps:
        for match in _UNSAFE_STEP_ACTION.finditer(step):
            prefix = step[max(0, match.start() - 16) : match.start()]
            if _STEP_NEGATION.search(prefix):
                continue
            if _STEP_PROFESSIONAL.search(prefix):
                continue
            return True
    return False


@dataclass
class AgentState:
    """Conversation state with per-slot follow-up budgets.

    ``model_ask_count`` and ``fault_ask_count`` are the only routing budgets.
    ``follow_up_count`` is a derived audit value for tickets/evaluation and is
    never used to decide whether the conversation may continue.
    """

    session_id: str = field(default_factory=lambda: f"S-{uuid.uuid4().hex[:10].upper()}")
    messages: list[str] = field(default_factory=list)
    history: list[dict[str, str]] = field(default_factory=list)
    model_ask_count: int = 0
    fault_ask_count: int = 0
    feedback_ask_count: int = 0
    asked_slots: list[str] = field(default_factory=list)
    pending_slot: str | None = None
    last_question: str | None = None
    conflict_follow_up_asked: bool = False
    # A short confirmation ("是") resolves the model only for that turn unless
    # it is persisted here.  Extraction runs over the raw user history, which
    # never contains "LX-2000" when the user only answered "是"; without this
    # the next turn forgets the confirmation and asks for the model again.
    confirmed_model: str | None = None
    confirmed_model_status: str | None = None
    # A diagnosis opens a follow-up loop instead of ending the session.  The
    # loop closes only on an explicit resolution, a handoff, or budget
    # exhaustion.
    in_follow_up: bool = False
    diagnosis_count: int = 0
    last_diagnosis_code: str | None = None
    last_diagnosis_source_id: str | None = None
    last_diagnosis_title: str | None = None
    last_diagnosis_categories: list[str] = field(default_factory=list)
    last_diagnosis_confidence: float = 0.0
    excluded_codes: list[str] = field(default_factory=list)
    excluded_source_ids: list[str] = field(default_factory=list)
    feedback_mode: str | None = None
    finished: bool = False

    @property
    def follow_up_count(self) -> int:
        return self.model_ask_count + self.fault_ask_count + self.feedback_ask_count


class TriageAgent:
    def __init__(
        self,
        manual_path: str | Path,
        *,
        llm_extractor: RiskSignalExtractor | None = None,
        use_llm: bool = True,
        llm_note: str = "",
        ticket_sink: Callable[[dict], None] | None = None,
    ):
        self.index = ManualIndex.from_file(manual_path)
        self.llm_extractor = llm_extractor if use_llm else None
        self.risk_engine = RiskEngine(self.llm_extractor)
        self.use_llm = use_llm
        self.llm_note = llm_note
        self.ticket_sink = ticket_sink
        self._session = AgentState()

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def finished(self) -> bool:
        return self._session.finished

    def reset(self) -> None:
        self._session = AgentState()

    def handle(self, user_text: str) -> TurnResult:
        text = (user_text or "").strip()
        if not text:
            return TurnResult(
                action="ask",
                message="请描述打印机目前的具体现象。",
                session_id=self.session_id,
                stage_trace=["A:empty_input"],
            )

        if self._session.finished:
            self.reset()

        self._session.messages.append(text)
        self._session.history.append({"role": "user", "content": text})
        trace = ["A:input_accumulated"]
        # Re-extract from the *entire* user history every turn, then apply
        # deterministic coreference for short confirmations ("是", "对").
        info = extract_info(self._session.messages)
        info = self._apply_code_correction(info, text)
        info = self._restore_confirmed_model(info, trace)
        info = self._resolve_contextual_slots(info, text, trace)
        keyword_risk = self.risk_engine.assess_keywords(info.user_text)

        # Any node: an explicit request for a human wins immediately.
        if info.requested_human:
            trace.append("I:explicit_human_request")
            return self._handoff(
                "explicit_human_request",
                info,
                keyword_risk,
                trace,
            )

        # Safety preemption: dangerous cases count as sufficient for escalation.
        # We do not delay a smoke/shock report while asking for a model number.
        if keyword_risk.level == "high":
            trace.append("B:safety_sufficient")
            return self._risk_and_maybe_handoff(info, trace, force_handoff_reason="high_risk")

        # Scope gate before symptom sufficiency: "LX-2001 卡纸了" must hand off
        # as out_of_scope, never continue just because "卡纸" is a strong symptom.
        if info.model_status == "unsupported":
            trace.append("I:unsupported_model")
            return self._handoff("unsupported_model", info, keyword_risk, trace)
        if info.model_status == "conflict":
            trace.append("I:model_conflict")
            return self._handoff("model_conflict", info, keyword_risk, trace)

        # Post-diagnosis loop: re-enter check_info with the user's feedback
        # instead of resetting the session.  Safety and explicit-human checks
        # above still preempt everything.
        feedback_result = self._handle_feedback(info, keyword_risk, trace)
        if feedback_result is not None:
            return feedback_result

        # Model scope is decided first.  Only for an in-scope model do we ask
        # the LLM to fill the symptom slot, and only when deterministic
        # extraction has not already found an actionable symptom.
        if info.model_status == "known":
            self._augment_symptoms_with_llm(info, trace)

        # Model escape hatch: after the independent model budget is exhausted,
        # use LX-2000 as a *clearly labelled assumption* whenever the history
        # contains a code or an actionable symptom.  Handoff only when there is
        # no usable fault information at all.
        if (
            info.model_status == "unknown"
            and self._session.model_ask_count >= MAX_MODEL_ASKS
        ):
            if self._has_usable_fault_info(info):
                self._assume_supported_model(info, trace)
                self._augment_symptoms_with_llm(info, trace)
            else:
                trace.append("C:model_exhausted_without_fault_info")
                return self._handoff(
                    "insufficient_info",
                    info,
                    keyword_risk,
                    trace,
                )

        if not self._info_sufficient(info):
            trace.append("B:insufficient")
            # A confirmed model with an exhausted symptom budget ends the
            # session.  This check is per-slot; the model counter is irrelevant.
            if (
                info.model_status in {"known", "assumed"}
                and not self._has_symptom_or_code(info)
                and self._session.fault_ask_count >= MAX_FAULT_ASKS
            ):
                trace.append("C:fault_limit_reached")
                return self._handoff(
                    "insufficient_info",
                    info,
                    keyword_risk,
                    trace,
                )

            slot, question = self._next_question(info)
            if slot is None or question is None:
                trace.append("C:no_question_available")
                return self._handoff(
                    "insufficient_info",
                    info,
                    keyword_risk,
                    trace,
                )
            return self._ask_result(slot, question, info, trace)

        trace.append("B:sufficient")
        return self._risk_and_maybe_handoff(info, trace)

    def _remember_model(self, status: str) -> None:
        self._session.confirmed_model = SUPPORTED_MODEL
        self._session.confirmed_model_status = status

    def _restore_confirmed_model(
        self,
        info: ExtractedInfo,
        trace: list[str],
    ) -> ExtractedInfo:
        """Reapply a model confirmation that is not present in the raw history.

        Explicit evidence in the current turn always wins: if extraction says
        ``unsupported`` or ``conflict``, the persisted value is not applied.
        """

        if info.model_status != "unknown" or not self._session.confirmed_model:
            return info
        status = self._session.confirmed_model_status or "known"
        info.model = self._session.confirmed_model
        info.model_status = status
        info.model_assumed = status == "assumed"
        info.unsupported_hint = None
        info.out_of_scope = False
        trace.append(f"A:model_context({info.model},{status})")
        return info

    def _resolve_contextual_slots(
        self,
        info: ExtractedInfo,
        latest_text: str,
        trace: list[str],
    ) -> ExtractedInfo:
        """Resolve a short confirmation against the previous assistant turn.

        Only the *assistant question* is used for coreference; assistant text
        is never fed into slot extraction, because the question itself
        contains "LX-2000" and would create a false positive.
        """

        if not _AFFIRMATIVE_REPLY.match(normalize_text(latest_text)):
            return info
        if self._session.pending_slot not in {"model", "model_repeat"}:
            return info
        if info.model_status != "unknown":
            return info

        # The model question always asks whether the device is LX-2000.
        info.model = SUPPORTED_MODEL
        info.model_status = "known"
        info.model_assumed = False
        info.unsupported_hint = None
        info.out_of_scope = False
        self._remember_model("known")
        trace.append("A:coreference(model=LX-2000)")
        return info

    # ------------------------------------------------------------------
    # Post-diagnosis feedback loop
    # ------------------------------------------------------------------

    def _handle_feedback(
        self,
        info: ExtractedInfo,
        risk: RiskAssessment,
        trace: list[str],
    ) -> TurnResult | None:
        """Re-enter check_info after a diagnosis instead of resetting.

        Returns ``None`` when the session is not in the post-diagnosis loop so
        the normal first-pass workflow continues unchanged.
        """

        if not self._session.in_follow_up:
            return None

        feedback = analyze_feedback(self._session.messages[-1])
        trace.append(
            "A:feedback("
            f"negative={str(feedback.negative).lower()},"
            f"resolved={str(feedback.resolved).lower()},"
            f"progress={str(feedback.progress).lower()},"
            f"denied_codes={','.join(feedback.denied_codes) or '-'},"
            f"denied_categories={','.join(feedback.denied_categories) or '-'},"
            f"denied_text={feedback.denied_text or '-'})"
        )

        if feedback.resolved:
            return self._resolved_result(info, risk, trace)

        # "拿出来了，然后呢" is not a denial.  Keep the current diagnosis
        # context and continue from the same manual chapter without retrieval.
        if feedback.progress:
            return self._progress_result(info, risk, feedback, trace)

        # "故障发生在打印的时候" yields only the weak `vague` category; it is
        # context, not a new fault symptom, and must not exclude the current
        # diagnosis as if the user had denied it.
        has_new_info = bool(
            feedback.explicit_codes
            or has_actionable_symptom(feedback.new_categories)
        )
        if not feedback.negative and (feedback.is_question or not has_new_info):
            trace.append("A:feedback_ambiguous")
            if feedback.is_question:
                trace.append("A:feedback_question_requires_confirmation")
            # Let the LLM recover paraphrased progress intent that the
            # deterministic patterns did not recognize.  It may only select
            # existing manual steps; routing stays with this workflow.
            llm_intent, llm_steps = self._llm_feedback_intent(trace)
            if llm_intent == "progress":
                return self._progress_result(
                    info,
                    risk,
                    feedback,
                    trace,
                    llm_steps=llm_steps,
                )
            if self._session.feedback_ask_count < MAX_FEEDBACK_ASKS:
                return self._ask_result(
                    "feedback",
                    self._feedback_question(),
                    info,
                    trace,
                    risk=risk,
                )
            trace.append("C:feedback_limit_reached")
            return self._handoff("feedback_unresolved", info, risk, trace)

        self._session.feedback_mode = (
            "negative" if feedback.negative else "new_info"
        )
        excluded_codes, excluded_sources = self._apply_feedback_exclusions(
            info, feedback, trace
        )
        return self._risk_and_maybe_handoff(
            info,
            trace,
            feedback=feedback,
            query_suffix=self._feedback_query_suffix(info),
            categories=self._feedback_categories(info, feedback),
            exclude_codes=excluded_codes,
            exclude_source_ids=excluded_sources,
        )

    def _apply_feedback_exclusions(
        self,
        info: ExtractedInfo,
        feedback: FeedbackAnalysis,
        trace: list[str],
    ) -> tuple[list[str], list[str]]:
        """Exclude the rejected chapter and any code the user explicitly denied."""

        codes: list[str] = []
        source_ids: list[str] = []
        if self._session.last_diagnosis_code:
            codes.append(self._session.last_diagnosis_code.upper())
        if self._session.last_diagnosis_source_id:
            source_ids.append(self._session.last_diagnosis_source_id)
        codes.extend(code.upper() for code in feedback.denied_codes)

        for code in codes:
            if code not in self._session.excluded_codes:
                self._session.excluded_codes.append(code)
        for source_id in source_ids:
            if source_id not in self._session.excluded_source_ids:
                self._session.excluded_source_ids.append(source_id)

        suppressed = set(self._session.excluded_codes)
        if info.error_codes:
            info.error_codes = [
                code for code in info.error_codes if code.upper() not in suppressed
            ]
        if (
            info.inferred_code_hint
            and info.inferred_code_hint.upper() in suppressed
        ):
            info.inferred_code_hint = None

        trace.append(
            "A:feedback_exclude("
            f"codes={','.join(sorted(set(codes))) or '-'},"
            f"sources={','.join(source_ids) or '-'})"
        )
        return codes, source_ids

    def _feedback_query_suffix(self, info: ExtractedInfo) -> str:
        """Add the alternative-symptom vocabulary needed by BM25.

        Exclusion alone leaves E08 as the only high-scoring section; the
        expansion lets E19/E10 compete on their actual manual wording.
        """

        last_code = (self._session.last_diagnosis_code or "").upper()
        if last_code in _FEEDBACK_QUERY_EXPANSIONS:
            return _FEEDBACK_QUERY_EXPANSIONS[last_code]
        categories = set(info.categories) | set(
            self._session.last_diagnosis_categories
        )
        for category in ("paper_mismatch", "paper_jam", "noise"):
            if category in categories:
                return _FEEDBACK_CATEGORY_EXPANSIONS[category]
        return ""

    @staticmethod
    def _feedback_categories(
        info: ExtractedInfo, feedback: FeedbackAnalysis
    ) -> list[str]:
        """Drop categories the user explicitly denied, keeping any new symptom."""

        if not feedback.denied_categories:
            return list(info.categories)
        denied = set(feedback.denied_categories)
        return [category for category in info.categories if category not in denied]

    def _feedback_question(self) -> str:
        title = self._session.last_diagnosis_title or "上一轮的诊断结论"
        labels = "、".join(
            SYMPTOM_LABELS.get(category, category)
            for category in self._session.last_diagnosis_categories[:3]
        )
        detail = f"“{title}”"
        if labels:
            detail += f"（涉及{labels}）"
        return (
            f"收到。您是说上一轮关于{detail}的判断不符合实际情况吗？"
            "请具体告诉我哪一点不对：例如没有出现屏幕提示、纸张状态不同、"
            "或者按步骤操作后仍然不行。也可以直接描述一个新现象。"
        )

    def _resolved_result(
        self,
        info: ExtractedInfo,
        risk: RiskAssessment,
        trace: list[str],
    ) -> TurnResult:
        message = (
            "好的，已记录本轮问题为已解决。"
            "如果之后再次出现，可以随时描述现象重新排查。"
        )
        self._session.in_follow_up = False
        self._session.pending_slot = None
        self._session.last_question = None
        self._session.feedback_mode = "resolved"
        self._session.finished = True
        self._session.history.append({"role": "assistant", "content": message})
        trace.append("H:session_resolved")
        return TurnResult(
            action="resolved",
            message=message,
            session_id=self.session_id,
            stage_trace=trace,
            info=info.to_ticket_dict(),
            risk=risk.to_dict(),
            debug={
                "llm": self.llm_note or "not_configured",
                "diagnosis_count": self._session.diagnosis_count,
            },
        )

    def _diagnosis_document(self) -> ManualDocument | None:
        """Return the manual chapter behind the most recent diagnosis."""

        source_id = self._session.last_diagnosis_source_id
        code = self._session.last_diagnosis_code
        if source_id:
            for document in self.index.documents:
                if document.source_id == source_id:
                    return document
        if code:
            for document in self.index.documents:
                if document.code == code:
                    return document
        return None

    @staticmethod
    def _progress_step_indexes(
        document: ManualDocument,
        stage: str | None,
    ) -> list[int]:
        """Choose the remaining manual steps for a progress reply.

        The index is zero-based; the caller keeps the manual's original
        numbering when displaying the steps.
        """

        steps = document.steps
        if not steps:
            return []
        start = 0
        if stage == "obstruction_removed":
            anchor = None
            for index, step in enumerate(steps):
                if (
                    ("取出" in step or "清理" in step or "拿出" in step)
                    and any(term in step for term in ("纸", "卡", "碎"))
                ):
                    anchor = index
            if anchor is not None:
                start = anchor + 1
        elif stage == "cover_opened":
            anchor = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if "取出" in step or "清理" in step
                ),
                None,
            )
            if anchor is not None:
                start = anchor
            else:
                opened = next(
                    (
                        index
                        for index, step in enumerate(steps)
                        if "打开" in step or "开启" in step
                    ),
                    None,
                )
                if opened is not None:
                    start = opened + 1
        return list(range(min(start, len(steps) - 1), len(steps)))

    def _llm_feedback_intent(
        self,
        trace: list[str],
    ) -> tuple[str | None, list[int]]:
        """Ask the LLM for a bounded post-diagnosis intent classification.

        The model receives only the current diagnosis title, the manual steps
        and the user's latest message.  It returns an intent plus zero-based
        step indexes that are validated against the manual before use.
        """

        extractor = self.llm_extractor
        method = getattr(extractor, "classify_feedback_intent", None)
        document = self._diagnosis_document()
        if method is None or document is None or not self._session.messages:
            return None, []
        try:
            payload, notes = method(
                self._session.messages[-1],
                diagnosis_title=document.phenomenon,
                steps=document.steps,
                warning=document.warning or "",
            )
        except Exception:
            trace.append("L:feedback_intent=fallback:exception")
            return None, []
        if payload is None:
            trace.append("L:feedback_intent=fallback")
            if notes:
                trace.append(f"L:feedback_note={notes[0]}")
            return None, []

        intent = str(payload.get("intent", ""))
        if intent not in {"solved", "denial", "progress", "unclear"}:
            trace.append("L:feedback_intent=fallback:invalid_intent")
            return None, []
        evidence = str(payload.get("evidence", ""))
        compacted_evidence = compact_text(evidence)
        compacted_user = compact_text(self._session.messages[-1])
        if intent in {"solved", "denial", "progress"} and (
            not compacted_evidence or compacted_evidence not in compacted_user
        ):
            trace.append("L:feedback_intent=fallback:no_evidence")
            return None, []
        trace.append(f"L:feedback_intent={intent}")
        raw_numbers = payload.get("next_step_numbers", [])
        if not isinstance(raw_numbers, list):
            return intent, []
        indexes: list[int] = []
        for value in raw_numbers:
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 1 <= value <= len(document.steps):
                indexes.append(value - 1)
        return intent, sorted(set(indexes))

    def _progress_result(
        self,
        info: ExtractedInfo,
        risk: RiskAssessment,
        feedback: FeedbackAnalysis,
        trace: list[str],
        *,
        llm_steps: list[int] | None = None,
    ) -> TurnResult:
        """Answer a "what next?" reply from the current manual chapter.

        This path deliberately does not call the retriever.  The previous
        diagnosis remains authoritative; the user only needs the remaining
        steps for the action they just completed.
        """

        document = self._diagnosis_document()
        trace.append(
            "A:progress("
            f"stage={feedback.progress_stage or 'generic'},"
            f"text={feedback.progress_text or '-'})"
        )
        if document is None:
            # Defensive fallback: without the previous chapter there is no
            # grounded way to continue, so use the normal clarification path.
            if self._session.feedback_ask_count < MAX_FEEDBACK_ASKS:
                return self._ask_result(
                    "feedback",
                    self._feedback_question(),
                    info,
                    trace,
                    risk=risk,
                )
            return self._handoff("feedback_unresolved", info, risk, trace)

        indexes = self._progress_step_indexes(
            document, feedback.progress_stage
        )
        if llm_steps is None:
            llm_intent, llm_steps = self._llm_feedback_intent(trace)
            if llm_intent not in {None, "progress"}:
                llm_steps = []
        if llm_steps:
            if feedback.progress_stage is None:
                # The deterministic patterns did not recognize this paraphrase;
                # the LLM-proposed subset is the only concrete progress signal.
                indexes = llm_steps
            else:
                # Never rewind past the deterministic "already done" boundary.
                # The model may add a later valid step, but cannot re-include
                # the action the user just completed.
                floor = min(indexes)
                llm_steps = [index for index in llm_steps if index >= floor]
                indexes = sorted(set(indexes) | set(llm_steps))
            trace.append(
                "L:progress_steps=" + ",".join(str(index + 1) for index in indexes)
            )
        if not indexes:
            return self._handoff("no_manual_steps", info, risk, trace)

        code_label = document.code or "手册未标注代码"
        manual_high_risk = "高" in (document.manual_risk or "")
        has_warning = bool(
            document.warning
            and document.warning not in {"无", "无。", "低风险。无"}
        )
        lines = [
            f"继续按上一轮诊断（{code_label}：{document.phenomenon}）处理。",
            "已记录您反馈的进度，本轮不重新检索，直接给出后续步骤。",
            "",
        ]
        if manual_high_risk:
            if has_warning:
                lines.append(f"【安全警告】{document.warning}")
            else:
                lines.append(
                    "【安全警告】本手册章节标注为高风险，请严格按步骤操作；"
                    "如涉及拆机、电路或定影组件，请停止并联系专业人员。"
                )
            lines.append(f"手册风险标注：{document.manual_risk}。")
            lines.append("")

        lines.append("接下来请继续：")
        lines.extend(
            f"{index + 1}. {document.steps[index]}" for index in indexes
        )
        if has_warning and not manual_high_risk:
            lines.extend(["", f"安全提示：{document.warning}"])
        if (
            document.manual_risk
            and document.manual_risk != "未知"
            and not manual_high_risk
        ):
            lines.append(f"手册风险标注：{document.manual_risk}。")
        lines.extend(
            [
                "",
                f"来源：{document.citation}",
                "如已完成某一步可跳过；若仍无改善，请说明卡在哪一步。"
                "问题解决可直接回复“解决了”。",
            ]
        )
        message = "\n".join(lines)
        if _COST_OR_PROMISE.search(message):
            return self._handoff("no_manual_steps", info, risk, trace)

        self._session.feedback_mode = "progress"
        self._session.pending_slot = None
        self._session.last_question = None
        self._session.in_follow_up = True
        self._session.history.append({"role": "assistant", "content": message})
        trace.append(
            f"H:progress_guidance({code_label},steps="
            f"{','.join(str(index + 1) for index in indexes)})"
        )
        trace.append("H:follow_up_open")
        return TurnResult(
            action="guide",
            message=message,
            session_id=self.session_id,
            stage_trace=trace,
            info=info.to_ticket_dict(),
            risk=risk.to_dict(),
            sources=[
                {
                    "source_id": document.source_id,
                    "code": document.code,
                    "title": document.phenomenon,
                    "citation": document.citation,
                    "confidence": round(
                        self._session.last_diagnosis_confidence, 4
                    ),
                }
            ],
            debug={
                "llm": self.llm_note or "not_configured",
                "feedback_mode": "progress",
                "diagnosis_count": self._session.diagnosis_count,
                "excluded_codes": list(self._session.excluded_codes),
                "progress_steps": [index + 1 for index in indexes],
            },
        )

    @staticmethod
    def _has_symptom_or_code(info: ExtractedInfo) -> bool:
        return bool(
            info.error_codes
            or info.inferred_code_hint
            or has_actionable_symptom(info.categories)
        )

    @classmethod
    def _has_usable_fault_info(cls, info: ExtractedInfo) -> bool:
        return cls._has_symptom_or_code(info)

    @staticmethod
    def _retrieval_codes(info: ExtractedInfo) -> list[str]:
        """Explicit codes win; otherwise use the deterministic E08 alias."""

        if info.error_codes:
            return list(info.error_codes)
        if info.inferred_code_hint:
            return [info.inferred_code_hint]
        return []

    def _assume_supported_model(self, info: ExtractedInfo, trace: list[str]) -> None:
        info.model = SUPPORTED_MODEL
        info.model_status = "assumed"
        info.model_assumed = True
        self._remember_model("assumed")
        trace.append("B:soft_downgrade(model=LX-2000)")

    def _ask_result(
        self,
        slot: str,
        question: str,
        info: ExtractedInfo,
        trace: list[str],
        *,
        risk: RiskAssessment | None = None,
        candidates: list[RetrievedCandidate] | None = None,
        debug: dict | None = None,
    ) -> TurnResult:
        """Record a follow-up, update the correct per-slot counter, and ask."""

        if slot in {"model", "model_repeat"}:
            self._session.model_ask_count += 1
            counter = self._session.model_ask_count
            pending = "model"
            trace_step = f"C:ask(model#{counter})"
        elif slot == "feedback":
            self._session.feedback_ask_count += 1
            counter = self._session.feedback_ask_count
            pending = "feedback"
            trace_step = f"C:ask(feedback#{counter})"
            self._session.feedback_mode = "clarify"
        elif slot == "conflict":
            self._session.fault_ask_count += 1
            self._session.conflict_follow_up_asked = True
            counter = self._session.fault_ask_count
            pending = "conflict"
            trace_step = f"C:ask(fault#{counter},conflict)"
        else:
            self._session.fault_ask_count += 1
            counter = self._session.fault_ask_count
            pending = "symptom" if slot == "symptom" else "detail"
            trace_step = f"C:ask(fault#{counter})"

        self._session.asked_slots.append(slot)
        self._session.pending_slot = pending
        self._session.last_question = question
        self._session.history.append({"role": "assistant", "content": question})
        trace.append(trace_step)

        payload_debug = {"llm": self.llm_note or "not_configured"}
        if debug:
            payload_debug.update(debug)
        return TurnResult(
            action="ask",
            message=question,
            session_id=self.session_id,
            stage_trace=trace,
            info=info.to_ticket_dict(),
            risk=risk.to_dict() if risk is not None else None,
            candidates=(
                [candidate.to_brief() for candidate in candidates[:3]]
                if candidates
                else []
            ),
            debug=payload_debug,
        )

    # ------------------------------------------------------------------
    # Risk / retrieval / generation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_code_correction(info: ExtractedInfo, latest_text: str) -> ExtractedInfo:
        """Allow a conflict follow-up to override an earlier wrong code.

        Extraction is cumulative, so without this a correction such as
        "看错了，其实是 E02" would keep the old E01 and remain conflicted.
        """

        normalized = normalize_text(latest_text)
        if not _CODE_CORRECTION.search(normalized):
            return info
        latest_codes = extract_error_codes(latest_text)
        if latest_codes:
            info.error_codes = latest_codes
            info.inferred_code_hint = None
            return info
        for code in list(info.error_codes):
            if re.search(
                rf"(?:不是|没有|没显示|不显示|并非)\s*{re.escape(code)}",
                normalized,
                re.IGNORECASE,
            ):
                info.error_codes.remove(code)
        if re.search(r"(?:没有|没|不显示|看不到|无)代码", normalized):
            info.error_codes = []
        if not info.error_codes and detect_print_interrupt(info.user_text):
            info.inferred_code_hint = PRINT_INTERRUPT_HINT_CODE
        return info

    def _risk_and_maybe_handoff(
        self,
        info: ExtractedInfo,
        trace: list[str],
        *,
        force_handoff_reason: str | None = None,
        feedback: FeedbackAnalysis | None = None,
        query_suffix: str = "",
        categories: list[str] | None = None,
        exclude_codes: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
    ) -> TurnResult:
        risk = self.risk_engine.assess(info.user_text, use_llm=self.use_llm)
        trace.append(f"D1:llm_signals={'used' if risk.llm_used else 'fallback'}")
        trace.append(f"D1b:keyword_signals={sum(1 for s in risk.signals if s.source == 'keyword')}")
        trace.append(f"D2:risk={risk.level}")

        if force_handoff_reason or risk.level == "high":
            trace.append("I:high_risk")
            return self._handoff("high_risk", info, risk, trace)

        if not info.model:
            # This should only be reachable through the safety path.
            trace.append("I:missing_model")
            return self._handoff("insufficient_info", info, risk, trace)

        evaluation_categories = (
            list(categories) if categories is not None else list(info.categories)
        )
        # Exclusions are cumulative: a user who rejects E08 and then E19 must
        # never be bounced back to E08 on the next "还是不行".
        excluded_codes = list(
            dict.fromkeys(
                [
                    *self._session.excluded_codes,
                    *(code.upper() for code in (exclude_codes or [])),
                ]
            )
        )
        excluded_sources = list(
            dict.fromkeys(
                [*self._session.excluded_source_ids, *(exclude_source_ids or [])]
            )
        )
        retrieval_codes = [
            code
            for code in self._retrieval_codes(info)
            if code.upper() not in set(excluded_codes)
        ]
        retrieval_query = (
            f"{info.user_text} {query_suffix}" if query_suffix else info.user_text
        )
        search = self.index.search(
            retrieval_query,
            model=info.model,
            error_codes=retrieval_codes,
            categories=evaluation_categories,
            exclude_codes=excluded_codes,
            exclude_source_ids=excluded_sources,
            top_k=5,
            min_confidence=(
                FEEDBACK_MIN_CONFIDENCE if feedback is not None else None
            ),
            min_margin=FEEDBACK_MIN_MARGIN if feedback is not None else None,
        )
        hint_suffix = (
            f",hint={info.inferred_code_hint}" if info.inferred_code_hint else ""
        )
        exclude_suffix = (
            f",exclude={','.join(excluded_codes) or '-'}"
            if excluded_codes or excluded_sources
            else ""
        )
        trace.append(
            f"F:retrieve(model={info.model},"
            f"codes={','.join(retrieval_codes) or '-'}{hint_suffix}"
            f"{exclude_suffix})"
        )
        if not search.ok or search.winner is None:
            failure = search.failure_code or "not_applicable"
            # Code/symptom conflict gets one clarification attempt.  It uses
            # the fault budget, because it is a fault-slot clarification.
            if (
                failure == "code_symptom_conflict"
                and not self._session.conflict_follow_up_asked
                and self._session.fault_ask_count < MAX_FAULT_ASKS
            ):
                return self._ask_result(
                    "conflict",
                    self._conflict_question(info),
                    info,
                    trace,
                    risk=risk,
                    candidates=search.candidates,
                    debug={"retrieval_failure": failure},
                )
            # A failed re-retrieval after feedback gets a bounded clarification
            # chance instead of immediately handing off or repeating E08.
            if (
                feedback is not None
                and self._session.feedback_ask_count < MAX_FEEDBACK_ASKS
            ):
                trace.append(f"G:feedback_retry({failure})")
                return self._ask_result(
                    "feedback",
                    self._feedback_question(),
                    info,
                    trace,
                    risk=risk,
                    candidates=search.candidates,
                    debug={
                        "retrieval_failure": failure,
                        "feedback_mode": self._session.feedback_mode,
                    },
                )
            if feedback is not None:
                trace.append("C:feedback_limit_reached")
            trace.append(f"G:rejected({failure})")
            trace.append("I:retrieval_failed")
            return self._handoff(
                "feedback_unresolved" if feedback is not None else failure,
                info,
                risk,
                trace,
                search=search,
            )
        trace.append(
            f"G:accepted({search.winner.document.code or 'no-code'},"
            f"confidence={search.confidence:.2f})"
        )

        if steps_require_professional(search.winner.document.steps):
            trace.append("I:unsafe_manual_steps")
            return self._handoff(
                "unsafe_manual_steps",
                info,
                risk,
                trace,
                search=search,
            )

        message = self._build_grounded_answer(
            search.winner,
            model_assumed=info.model_assumed,
            excluded_codes=excluded_codes,
        )
        if not message:
            trace.append("I:no_manual_steps")
            return self._handoff("no_manual_steps", info, risk, trace, search=search)

        trace.append(
            "H:answer_grounded"
            + (":assumed_model" if info.model_assumed else "")
        )
        self._record_diagnosis(search.winner)
        # Any diagnosed session knows which model produced the answer.  This
        # keeps the post-diagnosis feedback loop from losing the model when the
        # user confirmed it only with "是".
        if info.model:
            self._remember_model("assumed" if info.model_assumed else "known")
        if feedback is not None:
            trace.append("H:feedback_rediagnosed")
        trace.append("H:follow_up_open")
        self._session.history.append({"role": "assistant", "content": message})
        return TurnResult(
            action="diagnose",
            message=message,
            session_id=self.session_id,
            stage_trace=trace,
            info=info.to_ticket_dict(),
            risk=risk.to_dict(),
            sources=[self._source_dict(search.winner)],
            candidates=[candidate.to_brief() for candidate in search.candidates],
            debug={
                "confidence": round(search.confidence, 4),
                "retrieval_failure": search.failure_code,
                "llm": self.llm_note or "not_configured",
                "diagnosis_count": self._session.diagnosis_count,
                "feedback_mode": self._session.feedback_mode,
                "excluded_codes": list(self._session.excluded_codes),
            },
        )

    def _record_diagnosis(self, candidate: RetrievedCandidate) -> None:
        """Open the follow-up loop and remember what to exclude if denied."""

        document = candidate.document
        self._session.in_follow_up = True
        self._session.diagnosis_count += 1
        self._session.last_diagnosis_code = document.code
        self._session.last_diagnosis_source_id = document.source_id
        self._session.last_diagnosis_title = document.phenomenon
        self._session.last_diagnosis_categories = list(document.categories)
        self._session.last_diagnosis_confidence = candidate.confidence
        self._session.pending_slot = None
        self._session.last_question = None

    # ------------------------------------------------------------------
    # Symptom extraction and follow-up questions
    # ------------------------------------------------------------------

    def _augment_symptoms_with_llm(
        self, info: ExtractedInfo, trace: list[str]
    ) -> None:
        """Use DeepSeek as a semantic symptom extractor, never as a decider.

        Deterministic keyword extraction always runs first.  The LLM is only
        consulted when no error code and no actionable keyword symptom exists,
        so common phrase variants cost no extra request.  Any model or network
        failure leaves the deterministic result untouched.
        """

        if self.llm_extractor is None:
            return
        if info.error_codes or has_actionable_symptom(info.categories):
            return

        extract_symptoms = getattr(self.llm_extractor, "extract_symptoms", None)
        if not callable(extract_symptoms):
            return

        try:
            categories, notes = extract_symptoms(info.user_text)
        except Exception as exc:
            categories, notes = [], [
                f"故障现象抽取失败，保留关键词兜底：{type(exc).__name__}"
            ]

        if categories:
            merge_symptom_categories(info, categories)
            trace.append(f"D0:symptom_llm={','.join(categories)}")
        else:
            trace.append("D0:symptom_llm=none")
        for note in notes:
            trace.append(f"D0:note={note}")

    @classmethod
    def _info_sufficient(cls, info: ExtractedInfo) -> bool:
        return info.model_status in {"known", "assumed"} and cls._has_symptom_or_code(
            info
        )

    def _next_question(self, info: ExtractedInfo) -> tuple[str | None, str | None]:
        """Choose the next question from per-slot budgets, never a global cap.

        Ordering rationale:
        * ask the model first if it has never been asked;
        * if both slots are missing, ask the symptom next (it is often the
          information users can provide most easily);
        * return to the model only for its second and final attempt;
        * an exhausted model budget never blocks a new symptom answer because
          the model has its own escape hatch in ``handle``.
        """

        missing_model = info.model_status == "unknown"
        missing_symptom = not self._has_symptom_or_code(info)
        model_asks = self._session.model_ask_count
        fault_asks = self._session.fault_ask_count

        if missing_model and model_asks == 0:
            return "model", (
                "请先确认设备型号：可以查看机身铭牌或包装标签，"
                "上面是否有“LX-2000”字样？"
            )
        if missing_symptom and fault_asks == 0:
            return "symptom", self._symptom_question()
        if missing_model and model_asks < MAX_MODEL_ASKS:
            return "model_repeat", (
                "型号仍未确认。请直接回答：设备上是否有“LX-2000”字样？"
                "如果不是，请告诉我完整型号；不确定也可以直接说明。"
            )
        if missing_symptom and fault_asks < MAX_FAULT_ASKS:
            return "detail", self._detail_question(info)
        return None, None

    @staticmethod
    def _conflict_question(info: ExtractedInfo) -> str:
        codes = "、".join(info.error_codes) or "该错误代码"
        return (
            f"您提到的 {codes} 与描述的现象可能不是同一个故障。请再确认一次："
            "屏幕显示的代码具体是什么？纸张是完全不出，还是卡在机器内部/纸盒里？"
            "如果之前说错了，请直接更正代码或现象。"
        )

    @staticmethod
    def _symptom_question() -> str:
        return (
            "请具体描述故障现象：是开不了机、卡纸/不出纸、打印到一半停住、"
            "打印字迹异常、异响、连不上电脑，还是屏幕显示了错误代码？"
        )

    @staticmethod
    def _detail_question(info: ExtractedInfo) -> str:
        categories = set(info.categories)
        if categories & {"no_power"}:
            detail = "按电源键后，指示灯或屏幕是否有任何反应？电源线两端是否插紧？"
        elif categories & {
            "paper_jam",
            "paper_feed",
            "paper_output",
            "paper_mismatch",
        }:
            detail = (
                "屏幕是否显示 E01–E20 代码？卡纸、不出纸或打印中断发生在"
                "纸盒、机身内部还是出纸口？"
            )
        elif categories & {"print_quality", "toner"}:
            detail = "面板是否提示墨粉不足？字迹异常是固定位置还是整页都有？"
        elif categories & {"noise"}:
            detail = "异响来自纸盒、机身内部还是出纸口？是否伴随震动、焦味或冒烟？"
        elif categories & {"connectivity", "wireless"}:
            detail = "电脑/手机与打印机是如何连接的？打印机面板或网络状态页是否有异常提示？"
        else:
            detail = "面板是否有错误代码、指示灯是否闪烁？故障发生在开机、打印还是连接电脑时？"
        return f"请再补充一个细节：{detail}"

    # ------------------------------------------------------------------
    # Handoff / ticket
    # ------------------------------------------------------------------

    def _handoff(
        self,
        reason_code: str,
        info: ExtractedInfo,
        risk: RiskAssessment,
        trace: list[str],
        *,
        search: SearchResult | None = None,
    ) -> TurnResult:
        if reason_code not in REASON_TEXT:
            reason_code = "not_applicable"
        candidates = (
            search.candidates
            if search is not None
            else self._retrieve_candidates_for_ticket(info, risk)
        )
        candidates = self._trim_candidates(candidates, info.categories)
        ticket = self._build_ticket(reason_code, info, risk, candidates)
        self._session.finished = True
        self._session.in_follow_up = False
        self._session.pending_slot = None
        self._session.last_question = None
        self._session.feedback_mode = "handoff"
        if self.ticket_sink is not None:
            self.ticket_sink(ticket)

        return TurnResult(
            action="handoff",
            message=self._handoff_message(reason_code, info, risk),
            session_id=self.session_id,
            stage_trace=trace,
            info=info.to_ticket_dict(),
            risk=risk.to_dict(),
            candidates=[candidate.to_brief() for candidate in candidates],
            ticket=ticket,
            debug={
                "llm": self.llm_note or "not_configured",
                "risk_notes": risk.llm_notes,
            },
        )

    @staticmethod
    def _trim_candidates(
        candidates: list[RetrievedCandidate], categories: list[str]
    ) -> list[RetrievedCandidate]:
        """Keep only candidates that are plausible enough to help a human.

        Generic manual sections mention "冒烟" inside their safety warnings, so
        a raw character match alone can pull them into a high-risk ticket.
        Requiring half the top score (and a symptom-category match when the
        user gave one) removes that noise without hiding genuine alternatives.
        """

        if not candidates:
            return []
        top_score = max(candidate.raw_score for candidate in candidates)
        threshold = max(0.05, top_score * 0.5)
        trimmed: list[RetrievedCandidate] = []
        for candidate in candidates:
            if candidate.raw_score < threshold:
                continue
            if (
                categories
                and candidate.category_match == 0.0
                and not candidate.code_match
            ):
                continue
            trimmed.append(candidate)
        return trimmed[:3]

    def _retrieve_candidates_for_ticket(
        self, info: ExtractedInfo, risk: RiskAssessment
    ) -> list[RetrievedCandidate]:
        # Candidates from a manual scoped to LX-2000 would be misleading for an
        # unsupported or contradictory model, so the ticket stays honest.
        if info.model_status in {"unsupported", "conflict"}:
            return []
        should_retrieve = bool(
            info.error_codes
            or info.inferred_code_hint
            or info.strong_symptom
            or info.categories
            or risk.level == "high"
            or info.model
        )
        if not should_retrieve:
            return []
        enforce_model = info.model_status in {"known", "assumed"}
        excluded_codes = set(self._session.excluded_codes)
        retrieval_codes = [
            code
            for code in self._retrieval_codes(info)
            if code.upper() not in excluded_codes
        ]
        query = info.user_text
        if self._session.in_follow_up:
            suffix = self._feedback_query_suffix(info)
            if suffix:
                query = f"{query} {suffix}"
        try:
            result = self.index.search(
                query,
                model=info.model if enforce_model else None,
                error_codes=retrieval_codes,
                categories=info.categories,
                exclude_codes=list(self._session.excluded_codes),
                exclude_source_ids=list(self._session.excluded_source_ids),
                top_k=5,
                enforce_model=enforce_model,
            )
        except Exception:
            return []
        candidates = sorted(
            result.candidates,
            key=lambda candidate: (candidate.code_match, candidate.raw_score),
            reverse=True,
        )
        return candidates[:3]

    def _build_ticket(
        self,
        reason_code: str,
        info: ExtractedInfo,
        risk: RiskAssessment,
        candidates: list[RetrievedCandidate],
    ) -> dict:
        now = datetime.now(timezone.utc)
        ticket_id = f"HO-{now:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6].upper()}"
        extracted = info.to_ticket_dict()
        # Per-slot budgets are authoritative.  follow_up_count is the derived
        # audit total for evaluators and human reviewers, not a routing cap.
        extracted["model_ask_count"] = self._session.model_ask_count
        extracted["fault_ask_count"] = self._session.fault_ask_count
        extracted["feedback_ask_count"] = self._session.feedback_ask_count
        extracted["follow_up_count"] = self._session.follow_up_count
        extracted["asked_slots"] = list(self._session.asked_slots)
        extracted["conflict_follow_up_asked"] = (
            self._session.conflict_follow_up_asked
        )
        extracted["diagnosis_count"] = self._session.diagnosis_count
        extracted["last_diagnosis_code"] = self._session.last_diagnosis_code
        extracted["excluded_codes"] = list(self._session.excluded_codes)
        extracted["excluded_source_ids"] = list(self._session.excluded_source_ids)
        # A dangerous case must reach the safety queue even when the user also
        # explicitly asks for a human; the explicit request is not a downgrade.
        queue = (
            "safety_escalation"
            if risk.level == "high"
            else _QUEUE_BY_REASON.get(reason_code, "general_after_sales")
        )
        return {
            "ticket_id": ticket_id,
            "created_at": now.isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "reason_code": reason_code,
            "handoff_reason": REASON_TEXT[reason_code],
            "recommended_queue": queue,
            "user_description": info.user_text,
            "user_messages": list(self._session.messages),
            "extracted": extracted,
            "risk": risk.to_dict(),
            "candidate_documents": [candidate.to_brief() for candidate in candidates],
            "manual_scope": SUPPORTED_MODEL,
            "manual_source": self.index.source_path,
        }

    @staticmethod
    def _handoff_message(
        reason_code: str, info: ExtractedInfo, risk: RiskAssessment
    ) -> str:
        if risk.level == "high":
            return (
                "检测到可能涉及冒烟、焦味、漏电、火花或异常高温的高风险信号。"
                "请立即停止使用并远离设备；只有在无需接近烟雾或危险源、且可以安全操作时，"
                "才切断电源。不要再次通电、不要自行拆机。若出现明火、持续烟雾或有人受伤，"
                "请撤离并联系消防或急救。已转人工并生成交接单。"
            )
        if reason_code == "explicit_human_request":
            return "好的，已按您的要求转人工，并生成交接单。"
        if reason_code == "insufficient_info":
            return (
                "现有信息仍不足以安全定位故障，继续猜测可能带来误操作风险。"
                "已停止追问并转人工，交接单中保留了目前已提取的信息。"
            )
        if reason_code == "feedback_unresolved":
            return (
                "上一轮诊断已被您否定。重新匹配后仍无法确定其他安全适用的手册章节，"
                "继续尝试可能造成误判。已转人工，并在交接单中注明已排除的章节。"
            )
        if reason_code == "unsupported_model":
            spoken_model = info.unsupported_hint or "您提到的型号"
            return (
                "抱歉，本分诊助手仅支持 LX-2000 型号。"
                f"您提到的 {spoken_model} 暂不在支持范围内，"
                "正在为您转接人工客服。"
            )
        if reason_code == "model_conflict":
            return (
                "该维修手册只适用于 LX-2000，当前型号信息无法确认或不在覆盖范围内。"
                "为避免套用错误步骤，已转人工核对机型。"
            )
        if reason_code == "code_symptom_conflict":
            return (
                "您提供的错误代码与故障现象指向不同故障，且补充确认后仍不一致。"
                "当前无法确定唯一适用的手册章节，已转人工核对代码和实际现象。"
            )
        if reason_code == "unsafe_manual_steps":
            return (
                "该手册章节的排查步骤涉及拆机、接触电路或拆卸定影组件等专业操作，"
                "超出用户可安全自助的范围。为避免误操作造成伤害，已转人工，"
                "不提供这些步骤。"
            )
        if reason_code == "code_not_found":
            codes = "、".join(info.error_codes) or "该代码"
            return (
                f"维修手册中未找到 {codes} 对应且可安全执行的章节。"
                "已转人工进一步确认。"
            )
        if reason_code in {"low_confidence", "ambiguous", "not_applicable"}:
            return (
                "维修手册中没有足够确定、且同时满足型号与故障现象匹配的排查内容。"
                "为避免给出错误步骤，已转人工。"
            )
        return "当前情况需要人工进一步确认，已生成交接单并转人工。"

    # ------------------------------------------------------------------
    # Grounded answer
    # ------------------------------------------------------------------

    def _build_grounded_answer(
        self,
        candidate: RetrievedCandidate,
        *,
        model_assumed: bool = False,
        excluded_codes: list[str] | None = None,
    ) -> str:
        document = candidate.document
        steps = [step for step in document.steps if step and not _COST_OR_PROMISE.search(step)]
        if not steps:
            return ""

        code_label = document.code or "手册未标注代码"
        manual_high_risk = "高" in (document.manual_risk or "")
        has_warning = bool(
            document.warning
            and document.warning not in {"无", "无。", "低风险。无"}
        )
        lines: list[str] = []
        if excluded_codes:
            excluded = "、".join(dict.fromkeys(excluded_codes))
            lines.extend(
                [
                    f"已根据您的反馈排除上一轮结论（{excluded}），"
                    "并按最新现象重新匹配维修手册。",
                    "",
                ]
            )
        if model_assumed:
            lines.extend(
                [
                    "基于 LX-2000 型号为您提供参考（型号尚未确认；"
                    "如实际机型不符，请停止操作并转人工核对）。",
                    "",
                ]
            )
        model_applicability = (
            "型号按 LX-2000 假定" if model_assumed else f"型号 {document.model} 匹配"
        )
        lines.extend(
            [
                f"根据维修手册，您的情况匹配：{document.phenomenon}（{code_label}）。",
                (
                    f"适用性：{model_applicability}；故障现象匹配；"
                    f"置信度 {candidate.confidence:.0%}。"
                ),
            ]
        )

        # Confirmed product decision: for chapters the manual marks high risk,
        # put a prominent warning *before* any user-facing steps.
        if manual_high_risk:
            lines.append("")
            if has_warning:
                lines.append(f"【安全警告】{document.warning}")
            else:
                lines.append(
                    "【安全警告】本手册章节标注为高风险，请严格按步骤操作；"
                    "如涉及拆机、电路或定影组件，请停止并联系专业人员。"
                )
            lines.append(f"手册风险标注：{document.manual_risk}。")
            lines.append("")

        lines.append("建议按以下顺序排查：")
        lines.extend(f"{index}. {step}" for index, step in enumerate(steps, start=1))

        if has_warning and not manual_high_risk:
            lines.extend(["", f"安全提示：{document.warning}"])
        if (
            document.manual_risk
            and document.manual_risk != "未知"
            and not manual_high_risk
        ):
            lines.append(f"手册风险标注：{document.manual_risk}。")

        lines.extend(
            [
                "",
                f"来源：{document.citation}",
                "以上内容仅依据上述手册章节整理；如步骤提示停止使用或联系专业人员，请照做。",
            ]
        )
        answer = "\n".join(lines)
        # Defence in depth: never return a fee/warranty promise even if a future
        # manual revision accidentally contains one.
        if _COST_OR_PROMISE.search(answer):
            return ""
        return answer

    @staticmethod
    def _source_dict(candidate: RetrievedCandidate) -> dict:
        document = candidate.document
        return {
            "source_id": document.source_id,
            "code": document.code,
            "title": document.phenomenon,
            "citation": document.citation,
            "confidence": round(candidate.confidence, 4),
        }
