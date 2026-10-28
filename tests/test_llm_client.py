from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DeepSeekRiskExtractor,
    build_default_risk_extractor,
)


class _FakeCompletions:
    def __init__(self, content: str, finish_reason: str = "stop", error=None):
        self.content = content
        self.finish_reason = finish_reason
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=self.finish_reason,
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class _FakeClient:
    def __init__(self, content: str, finish_reason: str = "stop", error=None):
        self.completions = _FakeCompletions(content, finish_reason, error)
        self.chat = SimpleNamespace(completions=self.completions)


class DeepSeekExtractorTests(unittest.TestCase):
    def test_parses_json_signals(self) -> None:
        client = _FakeClient(
            '{"signals":[{"category":"smoke","evidence":"冒烟了","negated":false}],'
            '"uncertain":false}'
        )
        extractor = DeepSeekRiskExtractor(client=client)
        signals, notes = extractor.extract("冒烟了")
        self.assertEqual(notes, [])
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].category, "smoke")
        self.assertEqual(signals[0].evidence, "冒烟了")
        self.assertEqual(signals[0].source, "llm")

    def test_accepts_markdown_fenced_json(self) -> None:
        client = _FakeClient(
            '```json\n{"signals":[],"uncertain":true}\n```'
        )
        signals, notes = DeepSeekRiskExtractor(client=client).extract("正常")
        self.assertEqual(signals, [])
        self.assertTrue(any("不确定" in note for note in notes))

    def test_invalid_json_falls_back(self) -> None:
        client = _FakeClient("not json")
        signals, notes = DeepSeekRiskExtractor(client=client).extract("冒烟了")
        self.assertEqual(signals, [])
        self.assertTrue(any("非 JSON" in note for note in notes))

    def test_length_finish_reason_falls_back(self) -> None:
        client = _FakeClient("{}", finish_reason="length")
        signals, notes = DeepSeekRiskExtractor(client=client).extract("冒烟了")
        self.assertEqual(signals, [])
        self.assertTrue(any("截断" in note for note in notes))

    def test_api_error_falls_back(self) -> None:
        client = _FakeClient("", error=RuntimeError("network down"))
        signals, notes = DeepSeekRiskExtractor(client=client).extract("冒烟了")
        self.assertEqual(signals, [])
        self.assertTrue(any("关键词兜底" in note for note in notes))

    def test_request_uses_deepseek_openai_compatible_shape(self) -> None:
        client = _FakeClient('{"signals":[],"uncertain":false}')
        extractor = DeepSeekRiskExtractor(client=client)
        extractor.extract("冒烟了")
        kwargs = client.completions.kwargs
        self.assertEqual(kwargs["model"], DEFAULT_MODEL)
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertIn("json", kwargs["messages"][0]["content"].lower())

    def test_feedback_intent_prompt_distinguishes_three_outcomes(self) -> None:
        client = _FakeClient(
            '{"intent":"progress","evidence":"拿出来了",'
            '"next_step_numbers":[3,4,5]}'
        )
        extractor = DeepSeekRiskExtractor(client=client)
        payload, notes = extractor.classify_feedback_intent(
            "拿出来了，然后呢，怎么做",
            diagnosis_title="每次打到一半就停住，纸还皱成一团",
            steps=[
                "停止打印并关闭设备。",
                "取出卡纸。",
                "检查纸张是否受潮。",
                "调整纸盒导板。",
                "打印测试页。",
            ],
            warning="须断电并等待冷却。",
        )
        self.assertEqual(notes, [])
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["intent"], "progress")
        self.assertEqual(payload["next_step_numbers"], [3, 4, 5])
        system_prompt = client.completions.kwargs["messages"][0]["content"]
        for word in ("solved", "denial", "progress", "unclear"):
            self.assertIn(word, system_prompt)
        self.assertIn("拿出来了，然后呢，怎么做", system_prompt)

    def test_default_base_url(self) -> None:
        self.assertEqual(DEFAULT_BASE_URL, "https://api.deepseek.com/v1")

    def test_missing_api_key_disables_llm(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            extractor, note = build_default_risk_extractor()
        self.assertIsNone(extractor)
        self.assertIn("DEEPSEEK_API_KEY", note)

    def test_disable_switch(self) -> None:
        with patch.dict(os.environ, {"TRIAGE_DISABLE_LLM": "1"}, clear=True):
            extractor, note = build_default_risk_extractor()
        self.assertIsNone(extractor)
        self.assertIn("TRIAGE_DISABLE_LLM", note)


if __name__ == "__main__":
    unittest.main()
