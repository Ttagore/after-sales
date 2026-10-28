from __future__ import annotations

import unittest

from risk_engine import RiskEngine
from schemas import RiskSignal as SchemaRiskSignal


class _FakeExtractor:
    name = "fake"

    def __init__(self, signals: list[SchemaRiskSignal]):
        self.signals = signals

    def extract(self, text: str):
        return list(self.signals), []


def signal(category: str, evidence: str, *, negated: bool = False) -> SchemaRiskSignal:
    return SchemaRiskSignal(
        category=category,
        label=category,
        evidence=evidence,
        source="llm",
        negated=negated,
    )


class KeywordRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine()

    def test_smoke_is_high(self) -> None:
        result = self.engine.assess_keywords("打印机冒烟了")
        self.assertEqual(result.level, "high")
        self.assertIn("RISK-HIGH-DIRECT", result.rule_ids)

    def test_electric_shock_is_high(self) -> None:
        result = self.engine.assess_keywords("我碰了一下外壳，手麻了一下，好像有电")
        self.assertEqual(result.level, "high")

    def test_pop_then_dead_is_high(self) -> None:
        result = self.engine.assess_keywords(
            "打印的时候突然啪地响了一声，然后整台机器就没反应了"
        )
        self.assertEqual(result.level, "high")
        self.assertIn("RISK-HIGH-POP-AND-DEAD", result.rule_ids)

    def test_negated_danger_is_low(self) -> None:
        result = self.engine.assess_keywords("没有冒烟，也没有焦味，就是卡纸了")
        self.assertEqual(result.level, "low")
        self.assertEqual(result.signals, [])

    def test_plain_fault_is_low(self) -> None:
        result = self.engine.assess_keywords("LX-2000 卡纸了")
        self.assertEqual(result.level, "low")

    def test_heat_is_high(self) -> None:
        result = self.engine.assess_keywords("机器外壳很烫")
        self.assertEqual(result.level, "high")

    def test_isolated_electrical_pop_is_medium(self) -> None:
        result = self.engine.assess_keywords("打印的时候啪地响了一声")
        self.assertEqual(result.level, "medium")

    def test_noise_with_vibration_is_high(self) -> None:
        result = self.engine.assess_keywords("打印机一直响，还带着震动")
        self.assertEqual(result.level, "high")
        self.assertIn("RISK-HIGH-NOISE-VIBRATION", result.rule_ids)

    def test_noise_without_vibration_stays_low(self) -> None:
        result = self.engine.assess_keywords("打印机一直响")
        self.assertEqual(result.level, "low")


class LlmSignalValidationTests(unittest.TestCase):
    def test_verbatim_signal_is_accepted(self) -> None:
        engine = RiskEngine(_FakeExtractor([signal("smoke", "冒烟")]))
        result = engine.assess("机器冒烟了", use_llm=True)
        self.assertEqual(result.level, "high")
        self.assertTrue(result.llm_used)
        self.assertTrue(any(item.source == "llm" for item in result.signals))

    def test_hallucinated_signal_is_dropped(self) -> None:
        engine = RiskEngine(_FakeExtractor([signal("smoke", "冒烟")]))
        result = engine.assess("LX-2000 卡纸了", use_llm=True)
        self.assertEqual(result.level, "low")
        self.assertTrue(any("无原文依据" in note for note in result.llm_notes))

    def test_model_marked_negated_is_dropped(self) -> None:
        engine = RiskEngine(
            _FakeExtractor([signal("smoke", "冒烟", negated=True)])
        )
        result = engine.assess("没有冒烟", use_llm=True)
        self.assertEqual(result.level, "low")
        self.assertEqual(result.signals, [])

    def test_quote_from_negated_context_is_dropped(self) -> None:
        # The model quotes "冒烟" but omits the negation; the source context
        # must still override an incorrect negated=false.
        engine = RiskEngine(
            _FakeExtractor([signal("smoke", "冒烟", negated=False)])
        )
        result = engine.assess("没有冒烟，只是卡纸", use_llm=True)
        self.assertEqual(result.level, "low")
        self.assertTrue(
            any("否定语境" in note for note in result.llm_notes),
            result.llm_notes,
        )

    def test_llm_failure_does_not_block_keywords(self) -> None:
        class BrokenExtractor:
            name = "broken"

            def extract(self, text: str):
                raise RuntimeError("network down")

        result = RiskEngine(BrokenExtractor()).assess("冒烟了", use_llm=True)
        self.assertEqual(result.level, "high")
        self.assertTrue(any("关键词兜底" in note for note in result.llm_notes))


if __name__ == "__main__":
    unittest.main()
