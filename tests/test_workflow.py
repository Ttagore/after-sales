from __future__ import annotations

import re
import unittest
from pathlib import Path

from agent_workflow import TriageAgent, steps_require_professional
from taxonomy import requests_human


MANUAL = Path(__file__).resolve().parents[1] / "printer_LX2000.md"
COST_OR_PROMISE = re.compile(
    r"免费|保修|包修|价格|报价|收费|费用|赔付|赔偿|退款|\d+\s*元|保证|承诺|百分百|一定能"
)


class WorkflowTests(unittest.TestCase):
    def make_agent(self, tickets: list[dict] | None = None) -> TriageAgent:
        sink = tickets.append if tickets is not None else None
        return TriageAgent(MANUAL, use_llm=False, ticket_sink=sink)

    def test_smoke_hands_off_immediately_without_model(self) -> None:
        agent = self.make_agent()
        result = agent.handle("冒烟了")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "high_risk")
        self.assertEqual(result.ticket["recommended_queue"], "safety_escalation")
        self.assertIn("B:safety_sufficient", result.stage_trace)
        self.assertIn("停止使用", result.message)
        self.assertNotIn("1.", result.message)  # no repair steps for danger
        candidate_codes = {
            candidate["code"] for candidate in result.ticket["candidate_documents"]
        }
        # Generic sections mention "冒烟" in warnings; they must not pollute
        # the safety ticket's candidate list.
        self.assertEqual(candidate_codes, {"E11"})

    def test_explicit_human_request_any_time(self) -> None:
        agent = self.make_agent()
        result = agent.handle("先别问了，我要转人工")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "explicit_human_request")

    def test_negated_human_request_does_not_handoff(self) -> None:
        agent = self.make_agent()
        result = agent.handle("不要转人工")
        self.assertEqual(result.action, "ask")
        self.assertNotIn("I:explicit_human_request", result.stage_trace)
        self.assertFalse(requests_human("不要人工"))
        self.assertFalse(requests_human("不需要人工客服"))
        self.assertFalse(requests_human("人工智能"))
        self.assertTrue(requests_human("急需人工帮忙"))

    def test_later_human_request_after_earlier_refusal(self) -> None:
        agent = self.make_agent()
        first = agent.handle("不要转人工")
        second = agent.handle("我要人工")
        self.assertEqual(first.action, "ask")
        self.assertEqual(second.action, "handoff")
        self.assertEqual(second.ticket["reason_code"], "explicit_human_request")

    def test_explicit_human_request_with_known_case_keeps_candidates(self) -> None:
        agent = self.make_agent()
        result = agent.handle("LX-2000 卡纸了，帮我转人工")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "explicit_human_request")
        codes = {
            candidate["code"] for candidate in result.ticket["candidate_documents"]
        }
        self.assertIn("E02", codes)

    def test_danger_outranks_explicit_human_for_queue(self) -> None:
        agent = self.make_agent()
        result = agent.handle("冒烟了，帮我转人工")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "explicit_human_request")
        self.assertEqual(result.ticket["recommended_queue"], "safety_escalation")
        self.assertEqual(result.risk["level"], "high")

    def test_independent_slot_budgets_then_handoff(self) -> None:
        agent = self.make_agent()
        messages = ["机器坏了", "LX-2000", "就是坏了", "不知道"]
        results = [agent.handle(message) for message in messages]
        self.assertEqual([item.action for item in results[:3]], ["ask", "ask", "ask"])
        self.assertEqual(results[3].action, "handoff")
        self.assertEqual(results[3].ticket["reason_code"], "insufficient_info")
        extracted = results[3].ticket["extracted"]
        self.assertEqual(extracted["follow_up_count"], 3)
        self.assertEqual(extracted["model_ask_count"], 1)
        self.assertEqual(extracted["fault_ask_count"], 2)
        self.assertEqual(len(extracted["asked_slots"]), 3)

    def test_affirmative_reply_resolves_model_and_uses_prior_symptom(self) -> None:
        agent = self.make_agent()
        first = agent.handle("打到一半出不来了")
        self.assertEqual(first.action, "ask")
        self.assertIn("C:ask(model#1)", first.stage_trace)

        second = agent.handle("是")
        self.assertEqual(second.action, "diagnose")
        self.assertIn("A:coreference(model=LX-2000)", second.stage_trace)
        self.assertEqual(second.sources[0]["code"], "E08")
        self.assertIn("来源：printer_LX2000.md", second.message)
        self.assertNotIn("基于 LX-2000 型号为您提供参考", second.message)

    def test_final_answer_after_model_questions_never_hands_off(self) -> None:
        agent = self.make_agent()
        messages = ["机器坏了", "去年买的", "卡纸", "是，打印"]
        results = [agent.handle(message) for message in messages]
        self.assertEqual([item.action for item in results[:3]], ["ask", "ask", "ask"])
        self.assertEqual(results[3].action, "diagnose")
        self.assertIn(results[3].sources[0]["code"], {"E02", "E08"})
        self.assertEqual(results[3].info["model"], "LX-2000")

    def test_affirmative_model_confirmation_survives_the_next_turn(self) -> None:
        agent = self.make_agent()
        first = agent.handle("打印机坏了")
        self.assertEqual(first.action, "ask")
        self.assertIn("C:ask(model#1)", first.stage_trace)

        second = agent.handle("是")
        self.assertEqual(second.action, "ask")
        self.assertIn("A:coreference(model=LX-2000)", second.stage_trace)
        self.assertIn("C:ask(fault#1)", second.stage_trace)

        third = agent.handle("打印一半停了")
        self.assertEqual(third.action, "diagnose")
        self.assertIn("A:model_context(LX-2000,known)", third.stage_trace)
        self.assertNotIn("C:ask(model#2)", third.stage_trace)
        self.assertEqual(third.sources[0]["code"], "E08")
        self.assertEqual(third.info["model"], "LX-2000")
        self.assertEqual(third.info["model_status"], "known")
        self.assertFalse(third.info["model_assumed"])
        self.assertEqual(agent._session.model_ask_count, 1)

    def test_later_explicit_unsupported_model_overrides_confirmation(self) -> None:
        agent = self.make_agent()
        agent.handle("打印机坏了")
        agent.handle("是")
        result = agent.handle("其实型号是 HP M100")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "unsupported_model")

    def test_model_budget_exhausted_soft_degrades_with_symptom(self) -> None:
        agent = self.make_agent()
        messages = ["机器坏了", "去年买的", "卡纸", "不知道"]
        results = [agent.handle(message) for message in messages]
        final = results[-1]
        self.assertEqual(final.action, "diagnose")
        self.assertIn("B:soft_downgrade(model=LX-2000)", final.stage_trace)
        self.assertIn("H:answer_grounded:assumed_model", final.stage_trace)
        self.assertEqual(final.sources[0]["code"], "E02")
        self.assertTrue(final.info["model_assumed"])
        self.assertEqual(final.info["model_status"], "assumed")
        self.assertTrue(
            final.message.startswith("基于 LX-2000 型号为您提供参考")
        )
        self.assertIn("型号按 LX-2000 假定", final.message)

    def test_model_budget_exhausted_soft_degrades_with_error_code(self) -> None:
        agent = self.make_agent()
        messages = ["机器坏了", "去年买的", "显示E01", "不知道"]
        results = [agent.handle(message) for message in messages]
        final = results[-1]
        self.assertEqual(final.action, "diagnose")
        self.assertIn("B:soft_downgrade(model=LX-2000)", final.stage_trace)
        self.assertEqual(final.sources[0]["code"], "E01")
        self.assertTrue(final.info["model_assumed"])
        self.assertTrue(
            final.message.startswith("基于 LX-2000 型号为您提供参考")
        )

    def test_model_budget_without_any_fault_info_still_hands_off(self) -> None:
        agent = self.make_agent()
        messages = ["机器坏了", "去年买的", "就是坏了", "不知道"]
        results = [agent.handle(message) for message in messages]
        final = results[-1]
        self.assertEqual(final.action, "handoff")
        self.assertEqual(final.ticket["reason_code"], "insufficient_info")
        self.assertIn("C:model_exhausted_without_fault_info", final.stage_trace)
        self.assertFalse(final.ticket["extracted"]["model_assumed"])

    def test_normal_case_returns_grounded_steps_with_citation(self) -> None:
        agent = self.make_agent()
        result = agent.handle("LX-2000 卡纸了")
        self.assertEqual(result.action, "diagnose")
        self.assertIn("E02", result.message)
        self.assertIn("来源：printer_LX2000.md", result.message)
        self.assertIn("第20-35行", result.message)
        self.assertIn("关闭打印机并拔掉电源线", result.message)
        self.assertNotRegex(result.message, COST_OR_PROMISE)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0]["code"], "E02")
        # E02 is a manual high-risk chapter: warning must precede the steps.
        self.assertLess(
            result.message.index("【安全警告】"),
            result.message.index("建议按以下顺序排查"),
        )
        self.assertIn("手册风险标注：高", result.message)

    def test_error_code_and_symptom_conflict_asks_once_then_hands_off(self) -> None:
        agent = self.make_agent()
        first = agent.handle("LX-2000 显示E01不出纸")
        self.assertEqual(first.action, "ask")
        self.assertIn("C:ask(fault#1,conflict)", first.stage_trace)
        second = agent.handle("屏幕就是E01，纸也完全不出来")
        self.assertEqual(second.action, "handoff")
        self.assertEqual(second.ticket["reason_code"], "code_symptom_conflict")
        self.assertTrue(second.ticket["candidate_documents"])

    def test_conflict_follow_up_accepts_a_code_correction(self) -> None:
        agent = self.make_agent()
        first = agent.handle("LX-2000 显示E01不出纸")
        self.assertEqual(first.action, "ask")
        second = agent.handle("看错了，其实是E02")
        self.assertEqual(second.action, "diagnose")
        self.assertIn("E02", second.message)
        self.assertEqual(second.sources[0]["code"], "E02")

    def test_heat_is_safety_escalated(self) -> None:
        agent = self.make_agent()
        result = agent.handle("机器很烫")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "high_risk")
        self.assertIn("B:safety_sufficient", result.stage_trace)

    def test_noise_with_vibration_is_safety_escalated_without_model(self) -> None:
        agent = self.make_agent()
        result = agent.handle("异响带震动")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "high_risk")
        self.assertIn("B:safety_sufficient", result.stage_trace)

    def test_unsafe_manual_steps_are_detected(self) -> None:
        self.assertTrue(steps_require_professional(["拆开外壳，检查电源板"]))
        self.assertTrue(steps_require_professional(["拆卸定影组件"]))
        self.assertFalse(
            steps_require_professional(["联系专业维修人员检查电源模块"])
        )
        self.assertFalse(steps_require_professional(["不要触碰内部电路"]))
        self.assertFalse(
            steps_require_professional(["打开用户手册允许开启的纸盒和检修盖"])
        )

    def test_unsupported_model_hands_off_without_foreign_candidates(self) -> None:
        agent = self.make_agent()
        result = agent.handle("HP 打印机卡纸了")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "unsupported_model")
        self.assertEqual(result.ticket["candidate_documents"], [])

    def test_conflicting_model_hands_off(self) -> None:
        agent = self.make_agent()
        result = agent.handle("LX-2000 和 HP 打印机都卡纸了")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "model_conflict")

    def test_negated_smoke_does_not_escalate(self) -> None:
        agent = self.make_agent()
        result = agent.handle("没有冒烟，LX-2000 卡纸了")
        self.assertEqual(result.action, "diagnose")
        self.assertEqual(result.risk["level"], "low")

    def test_ticket_contains_required_fields(self) -> None:
        tickets: list[dict] = []
        agent = self.make_agent(tickets)
        result = agent.handle("冒烟了")
        self.assertEqual(len(tickets), 1)
        ticket = tickets[0]
        self.assertEqual(ticket, result.ticket)
        for key in (
            "user_description",
            "extracted",
            "risk",
            "candidate_documents",
            "handoff_reason",
            "ticket_id",
        ):
            self.assertIn(key, ticket)
        self.assertEqual(ticket["user_description"], "冒烟了")
        self.assertEqual(ticket["risk"]["level"], "high")

    def test_no_fee_or_warranty_promise_in_any_message(self) -> None:
        agent = self.make_agent()
        first = agent.handle("冒烟了").message
        agent.reset()
        second = agent.handle("LX-2000 卡纸了").message
        for message in (first, second):
            self.assertNotRegex(message, COST_OR_PROMISE)

    # ------------------------------------------------------------------
    # Post-diagnosis feedback loop
    # ------------------------------------------------------------------

    def test_diagnosis_keeps_session_open_for_negative_feedback(self) -> None:
        agent = self.make_agent()
        first = agent.handle("LX-2000 打印一半停了")
        self.assertEqual(first.action, "diagnose")
        self.assertEqual(first.sources[0]["code"], "E08")
        self.assertFalse(agent.finished)
        self.assertIn("H:follow_up_open", first.stage_trace)

        second = agent.handle("没提示纸张错")
        self.assertEqual(second.action, "diagnose")
        self.assertFalse(agent.finished)
        self.assertTrue(
            second.stage_trace[1].startswith("A:feedback(negative=true")
        )
        self.assertIn(
            "A:feedback_exclude(codes=E08,sources=LX2000-E08)",
            second.stage_trace,
        )
        self.assertIn(
            "F:retrieve(model=LX-2000,codes=-,exclude=E08)",
            second.stage_trace,
        )
        self.assertEqual(second.sources[0]["code"], "E19")
        self.assertIn("排除上一轮结论（E08）", second.message)
        self.assertEqual(second.debug["excluded_codes"], ["E08"])
        self.assertEqual(second.debug["diagnosis_count"], 2)

    def test_denial_after_model_confirmed_by_yes_reretrieves(self) -> None:
        # Exact reported path: the symptom is given first, the model is
        # confirmed only with "是", and the diagnosis happens on that same
        # turn.  The next denial must still know the model and re-retrieve,
        # never hand off as insufficient_info.
        agent = self.make_agent()
        first = agent.handle("打印一半停了")
        self.assertEqual(first.action, "ask")
        self.assertIn("C:ask(model#1)", first.stage_trace)

        second = agent.handle("是")
        self.assertEqual(second.action, "diagnose")
        self.assertEqual(second.sources[0]["code"], "E08")

        third = agent.handle("没显示纸张不对")
        self.assertEqual(third.action, "diagnose")
        self.assertEqual(third.sources[0]["code"], "E19")
        self.assertFalse(agent.finished)
        self.assertIn(
            "A:feedback_exclude(codes=E08,sources=LX2000-E08)",
            third.stage_trace,
        )
        self.assertEqual(third.info["model"], "LX-2000")
        self.assertEqual(third.debug["excluded_codes"], ["E08"])

    def test_colloquial_print_interrupt_diagnoses_on_the_answer_turn(self) -> None:
        # Exact reported path: "打了一半停了" (not "打印一半停了") must be
        # recognized as E08 on the turn it is supplied, so the second fault
        # follow-up and the subsequent handoff never happen.
        agent = self.make_agent()
        first = agent.handle("打印机坏了")
        self.assertEqual(first.action, "ask")
        self.assertIn("C:ask(model#1)", first.stage_trace)

        second = agent.handle("是")
        self.assertEqual(second.action, "ask")
        self.assertIn("C:ask(fault#1)", second.stage_trace)

        third = agent.handle("打了一半停了")
        self.assertEqual(third.action, "diagnose")
        self.assertEqual(third.sources[0]["code"], "E08")
        self.assertEqual(third.debug["diagnosis_count"], 1)
        self.assertFalse(
            any("fault#2" in step for step in third.stage_trace)
        )
        self.assertFalse(agent.finished)
        self.assertIsNone(third.ticket)

    def test_vague_stage_statement_after_diagnosis_does_not_exclude_chapter(self) -> None:
        # "故障发生在打印的时候" is context (vague), not a denial and not a
        # new actionable symptom.  It must not exclude E08 and re-retrieve.
        agent = self.make_agent()
        first = agent.handle("LX-2000 打了一半停了")
        self.assertEqual(first.sources[0]["code"], "E08")

        second = agent.handle("故障发生在打印的时候")
        self.assertEqual(second.action, "ask")
        self.assertIn("C:ask(feedback#1)", second.stage_trace)
        self.assertFalse(
            any(step.startswith("A:feedback_exclude") for step in second.stage_trace)
        )
        self.assertFalse(
            any(step.startswith("F:retrieve") for step in second.stage_trace)
        )
        self.assertEqual(second.debug.get("excluded_codes", []), [])

    def test_denied_category_with_new_symptom_changes_diagnosis(self) -> None:
        agent = self.make_agent()
        first = agent.handle("LX-2000 卡纸了")
        self.assertEqual(first.sources[0]["code"], "E02")

        second = agent.handle("没卡纸，就是一直响")
        self.assertEqual(second.action, "diagnose")
        self.assertEqual(second.sources[0]["code"], "E10")
        self.assertIn("排除上一轮结论（E02）", second.message)

    def test_repeated_negative_feedback_never_bounces_back(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 打印一半停了")
        second = agent.handle("还是不行")
        self.assertEqual(second.sources[0]["code"], "E19")

        third = agent.handle("还是不行")
        # E08 and E19 are now both excluded.  A low-confidence re-retrieval
        # must ask for feedback detail, never diagnose E08 again.
        self.assertEqual(third.action, "ask")
        self.assertTrue(
            any(step.startswith("C:ask(feedback#") for step in third.stage_trace)
        )
        self.assertNotIn("E08", third.message)

    def test_resolution_explicitly_ends_the_session(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 卡纸了")
        result = agent.handle("解决了")
        self.assertEqual(result.action, "resolved")
        self.assertTrue(agent.finished)
        self.assertIsNone(result.ticket)
        self.assertIn("已记录本轮问题为已解决", result.message)
        self.assertIn("H:session_resolved", result.stage_trace)

    def test_next_input_after_resolution_starts_a_new_session(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 卡纸了")
        agent.handle("解决了")
        old_session = agent.session_id
        result = agent.handle("机器坏了")
        self.assertEqual(result.action, "ask")
        self.assertIn("型号", result.message)
        self.assertNotEqual(result.session_id, old_session)
        self.assertFalse(agent.finished)

    def test_feedback_clarification_budget_then_handoff(self) -> None:
        agent = self.make_agent()
        messages = ["LX-2000 卡纸了", "有点道理", "我也不知道", "随便吧"]
        results = [agent.handle(message) for message in messages]
        self.assertEqual(
            [result.action for result in results],
            ["diagnose", "ask", "ask", "handoff"],
        )
        self.assertIn("C:ask(feedback#1)", results[1].stage_trace)
        self.assertIn("C:ask(feedback#2)", results[2].stage_trace)
        final = results[3]
        self.assertEqual(final.ticket["reason_code"], "feedback_unresolved")
        extracted = final.ticket["extracted"]
        self.assertEqual(extracted["feedback_ask_count"], 2)
        self.assertEqual(extracted["model_ask_count"], 0)
        self.assertEqual(extracted["follow_up_count"], 2)

    def test_safety_still_preempts_negative_feedback(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 卡纸了")
        result = agent.handle("没解决，现在冒烟了")
        self.assertEqual(result.action, "handoff")
        self.assertEqual(result.ticket["reason_code"], "high_risk")
        self.assertEqual(result.ticket["recommended_queue"], "safety_escalation")

    def test_progress_reply_continues_current_chapter_without_retrieval(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 打印一半停了")
        second = agent.handle("没显示纸张不对")
        self.assertEqual(second.sources[0]["code"], "E19")
        diagnosis_count = second.debug["diagnosis_count"]

        result = agent.handle("拿出来了，然后呢，怎么做")
        self.assertEqual(result.action, "guide")
        self.assertIn(
            "A:progress(stage=obstruction_removed,text=拿出来了,然后呢,怎么做)",
            result.stage_trace,
        )
        self.assertIn("H:progress_guidance(E19,steps=3,4,5)", result.stage_trace)
        self.assertFalse(
            any(step.startswith("F:retrieve") for step in result.stage_trace)
        )
        self.assertFalse(
            any(step.startswith("C:ask(feedback#") for step in result.stage_trace)
        )
        self.assertIn("3. 检查纸张是否受潮", result.message)
        self.assertIn("4. 整理纸张并调整纸盒导板", result.message)
        self.assertEqual(result.sources[0]["code"], "E19")
        self.assertEqual(result.debug["diagnosis_count"], diagnosis_count)
        self.assertFalse(agent.finished)
        self.assertEqual(
            agent.handle("解决了").action,
            "resolved",
        )

    def test_progress_phrase_with_question_is_not_feedback_clarification(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 卡纸了")
        result = agent.handle("打开了，下一步怎么弄")
        self.assertEqual(result.action, "guide")
        self.assertIn("A:progress(stage=cover_opened", result.stage_trace[2])
        self.assertIn("E02", result.sources[0]["code"])

    def test_stuck_removal_is_still_a_denial_not_progress(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 打印一半停了")
        result = agent.handle("拿不出来，怎么办")
        self.assertNotEqual(result.action, "guide")
        self.assertEqual(result.action, "diagnose")
        self.assertTrue(
            any(step.startswith("F:retrieve") for step in result.stage_trace)
        )

    def test_question_after_diagnosis_is_not_treated_as_denial(self) -> None:
        agent = self.make_agent()
        agent.handle("LX-2000 打印一半停了")
        result = agent.handle("是不是纸张不对？")
        self.assertEqual(result.action, "ask")
        self.assertIn("C:ask(feedback#1)", result.stage_trace)
        self.assertFalse(
            any(step.startswith("A:feedback_exclude") for step in result.stage_trace)
        )
        self.assertNotIn("E19", result.message)


if __name__ == "__main__":
    unittest.main()
