from __future__ import annotations

import unittest
from pathlib import Path

from rag_engine import ManualIndex
from taxonomy import extract_info


MANUAL = Path(__file__).resolve().parents[1] / "printer_LX2000.md"


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = ManualIndex.from_file(MANUAL)

    def search(self, query: str, **kwargs):
        info = extract_info([query])
        return self.index.search(
            query,
            model=info.model,
            error_codes=info.error_codes,
            categories=info.categories,
            top_k=5,
            **kwargs,
        )

    def assert_code(self, query: str, expected: str) -> None:
        result = self.search(query)
        self.assertTrue(result.ok, result.failure_reason)
        assert result.winner is not None
        self.assertEqual(result.winner.document.code, expected)

    def test_manual_parses_expected_sections(self) -> None:
        self.assertEqual(len(self.index.documents), 25)
        codes = {document.code for document in self.index.documents}
        for code in {f"E{number:02d}" for number in range(1, 21)}:
            self.assertIn(code, codes)

    def test_paper_jam(self) -> None:
        self.assert_code("LX-2000 卡纸了", "E02")

    def test_print_sent_but_no_response(self) -> None:
        self.assert_code("LX-2000 电脑上点了打印没反应", "E05")

    def test_phone_wireless(self) -> None:
        self.assert_code("LX-2000 手机搜不到打印机", "E15")

    def test_black_lines(self) -> None:
        self.assert_code("LX-2000 纸上有黑线", "E07")

    def test_faint_print(self) -> None:
        self.assert_code("LX-2000 字很浅", "E13")

    def test_network_not_found(self) -> None:
        self.assert_code("LX-2000 打印机连不上网", "E14")

    def test_no_power(self) -> None:
        self.assert_code("LX-2000 无法开机", "E01")

    def test_fuzzy_print(self) -> None:
        self.assert_code("LX-2000 打印字迹模糊", "E03")

    def test_exact_error_code(self) -> None:
        self.assert_code("LX-2000 显示E04", "E04")

    def test_code_and_symptom_conflict(self) -> None:
        result = self.search("LX-2000 显示E01不出纸")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, "code_symptom_conflict")
        self.assertLessEqual(result.confidence, 0.35)

    def test_code_not_in_manual(self) -> None:
        result = self.search("LX-2000 显示E99")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, "code_not_found")

    def test_wrong_model_filters_everything(self) -> None:
        result = self.index.search("卡纸了", model="HP-1234")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, "no_model_docs")

    def test_irrelevant_symptom_is_not_forced_through(self) -> None:
        result = self.search("LX-2000 今天天气不错")
        self.assertFalse(result.ok)
        self.assertIn(result.failure_code, {"low_confidence", "not_applicable"})

    def test_sources_are_traceable_to_manual_lines(self) -> None:
        result = self.search("LX-2000 卡纸了")
        assert result.winner is not None
        document = result.winner.document
        self.assertEqual(document.source_path, "printer_LX2000.md")
        self.assertGreater(document.start_line, 0)
        self.assertGreaterEqual(document.end_line, document.start_line)
        self.assertIn("第20-35行", document.citation)

    def test_exclude_codes_rejects_previous_diagnosis(self) -> None:
        query = (
            "LX-2000 打印一半停了 没提示纸张错 "
            "打印中断 打到一半停住 纸张皱 卡下一张"
        )
        result = self.index.search(
            query,
            model="LX-2000",
            categories=[],
            exclude_codes=["E08"],
            top_k=5,
        )
        self.assertTrue(result.ok, result.failure_reason)
        assert result.winner is not None
        self.assertEqual(result.winner.document.code, "E19")
        self.assertNotIn(
            "E08", {candidate.document.code for candidate in result.candidates}
        )

    def test_exclude_source_ids_filters_the_same_chapter(self) -> None:
        result = self.index.search(
            "LX-2000 卡纸了 还是不行 皱 进纸",
            model="LX-2000",
            categories=["paper_jam"],
            exclude_source_ids=["LX2000-E02"],
            top_k=5,
        )
        self.assertTrue(result.ok, result.failure_reason)
        assert result.winner is not None
        self.assertNotEqual(result.winner.document.source_id, "LX2000-E02")
        self.assertNotIn(
            "LX2000-E02",
            {candidate.document.source_id for candidate in result.candidates},
        )

    def test_feedback_second_pass_accepts_lower_confidence(self) -> None:
        strict = self.search("LX-2000 打印不清晰")
        self.assertFalse(strict.ok)
        self.assertIn(strict.failure_code, {"low_confidence", "ambiguous"})

        relaxed = self.search(
            "LX-2000 打印不清晰",
            min_confidence=0.4,
            min_margin=0.0,
        )
        self.assertTrue(relaxed.ok, relaxed.failure_reason)
        self.assertIsNotNone(relaxed.winner)
        self.assertLess(relaxed.confidence, 0.68)

    def test_all_documents_excluded_has_a_distinct_failure(self) -> None:
        result = self.index.search(
            "LX-2000 卡纸了",
            model="LX-2000",
            categories=["paper_jam"],
            exclude_source_ids=[
                document.source_id for document in self.index.documents
            ],
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.failure_code, "all_excluded")


if __name__ == "__main__":
    unittest.main()
