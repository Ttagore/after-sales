#!/usr/bin/env python3
"""Compare the offline rule channel with the real DeepSeek channel.

This script is intentionally separate from ``evaluate.py``.  It forces
``use_llm=True`` and runs every case once.  The model is only used for
bounded extraction/classification tasks:

* risk signal extraction;
* fault-symptom extraction;
* post-diagnosis feedback intent classification and manual-step selection
  (only when the deterministic progress patterns need a fallback).

Intent/action selection (ask / diagnose / guide / handoff / resolved), risk
level, retrieval and applicability decisions remain deterministic.  Therefore
the "LLM raw output" recorded below is the raw JSON object returned by
DeepSeek for those calls, not a model-authored diagnosis.

Usage:
    python evaluation/evaluate_llm.py
    python evaluation/evaluate_llm.py --json
    python evaluation/evaluate_llm.py --case BOUND-04 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(EVALUATION_DIR))

from agent_workflow import TriageAgent  # noqa: E402
from evaluate import (  # noqa: E402
    CATEGORY_LABELS,
    DEFAULT_CASES,
    DEFAULT_MANUAL,
    _actual_snapshot,
    _add_failure,
    _check_expectations,
    _check_invariants,
    _load_suite,
    _short,
)
from llm_client import build_default_risk_extractor  # noqa: E402
from taxonomy import MAX_TOTAL_FOLLOW_UPS  # noqa: E402


DEFAULT_REPORT = EVALUATION_DIR / "llm_report.json"


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _load_dotenv() -> None:
    """Load project-local .env without overriding real environment vars."""

    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            os.environ.setdefault(key, value)


def _extract_user_text(user_content: str) -> str:
    marker = "<user_text>\n"
    closing = "\n</user_text>"
    if marker in user_content and closing in user_content:
        return user_content.split(marker, 1)[1].rsplit(closing, 1)[0]
    return user_content


class RecordingExtractor:
    """Wrap a DeepSeek extractor and record every parsed JSON payload.

    The production adapter exposes ``_call_json`` with the parsed object; we
    wrap that one method per instance.  No production file is modified.
    """

    def __init__(self, inner: Any):
        self.inner = inner
        self.name = getattr(inner, "name", "llm")
        self.calls: list[dict[str, Any]] = []
        self._original_call_json = inner._call_json

        def wrapped_call_json(
            *,
            system_prompt: str,
            schema: dict[str, Any],
            user_content: str,
        ) -> tuple[dict[str, Any] | None, list[str]]:
            payload, notes = self._original_call_json(
                system_prompt=system_prompt,
                schema=schema,
                user_content=user_content,
            )
            properties = schema.get("properties", {})
            kind = "risk" if "signals" in properties else "symptom"
            self.calls.append(
                {
                    "kind": kind,
                    "input": _extract_user_text(user_content),
                    "raw": payload,
                    "notes": notes,
                    "ok": payload is not None,
                }
            )
            return payload, notes

        inner._call_json = wrapped_call_json

    def extract(self, text: str):
        return self.inner.extract(text)

    def extract_symptoms(self, text: str):
        return self.inner.extract_symptoms(text)


def _run_rule_case(case: dict[str, Any], manual_path: Path) -> dict[str, Any]:
    """Run one case through the original deterministic channel."""

    from evaluate import _run_case

    return _run_case(case, manual_path)


def _run_llm_case(
    case: dict[str, Any],
    manual_path: Path,
    extractor: RecordingExtractor,
    llm_note: str,
) -> dict[str, Any]:
    """Run one case once through the real LLM channel."""

    failures: list[dict[str, Any]] = []
    results: list[Any] = []
    extractor.calls = []

    try:
        agent = TriageAgent(
            manual_path,
            llm_extractor=extractor,
            use_llm=True,
            llm_note=llm_note,
        )
    except Exception as exc:
        _add_failure(
            failures,
            "agent.init",
            "TriageAgent 初始化成功",
            f"{type(exc).__name__}: {exc}",
        )
        return {
            "actual": {},
            "failures": failures,
            "passed": False,
            "calls": list(extractor.calls),
        }

    for message in case["messages"]:
        try:
            results.append(agent.handle(message))
        except Exception as exc:
            _add_failure(
                failures,
                "agent.handle",
                "无异常",
                f"{type(exc).__name__}: {exc}",
            )
            break

    if not results:
        return {
            "actual": {},
            "failures": failures,
            "passed": False,
            "calls": list(extractor.calls),
        }

    actual = _actual_snapshot(results)
    expected = case["expected"]

    accepted_actions = expected.get("accepted_final_actions")
    if accepted_actions:
        # Boundary case: the requirement is "classify it or ask explicitly,
        # never loop".  Either a diagnosis or one clear follow-up is accepted;
        # a second repeated question or a silent handoff is not.
        if actual["final_action"] not in accepted_actions:
            _add_failure(
                failures,
                "final_action",
                f"任意之一：{accepted_actions}",
                actual["final_action"],
            )
        max_ask_count = int(
            expected.get("max_ask_count", MAX_TOTAL_FOLLOW_UPS)
        )
        if actual["ask_count"] > max_ask_count:
            _add_failure(
                failures,
                "ask_count",
                f"<= {max_ask_count}（不得重复追问）",
                actual["ask_count"],
            )
        if actual["final_action"] == "ask" and "故障现象" not in actual["message"]:
            _add_failure(
                failures,
                "message_contains",
                "追问时必须明确提到故障现象",
                actual["message"],
            )
        if actual["final_action"] == "diagnose" and actual["source_count"] == 0:
            _add_failure(
                failures,
                "diagnose.sources",
                "诊断时应至少有一个来源",
                actual["source_count"],
            )

        # The exact turn_actions/final_action/ask_count/message check would
        # reject the alternative accepted behavior above.
        expected_copy = {
            key: value
            for key, value in expected.items()
            if key
            not in {
                "turn_actions",
                "final_action",
                "ask_count",
                "message_contains",
                "accepted_final_actions",
                "max_ask_count",
            }
        }
        _check_expectations(expected_copy, actual, failures)
    else:
        _check_expectations(expected, actual, failures)

    _check_invariants(results, actual, failures)

    final_info = results[-1].info if isinstance(results[-1].info, dict) else {}
    final_risk = results[-1].risk if isinstance(results[-1].risk, dict) else {}
    accepted_risk = [
        signal
        for signal in final_risk.get("signals", [])
        if isinstance(signal, dict) and signal.get("source") == "llm"
    ]

    return {
        "actual": actual,
        "failures": failures,
        "passed": not failures,
        "calls": list(extractor.calls),
        "accepted_symptom_categories": final_info.get("symptom_categories", []),
        "accepted_llm_risk_signals": accepted_risk,
        "stage_trace": actual.get("stage_trace", []),
    }


def _behavior_delta(rule: dict[str, Any], llm: dict[str, Any]) -> dict[str, Any]:
    rule_actual = rule.get("actual", {})
    llm_actual = llm.get("actual", {})
    keys = (
        "final_action",
        "reason_code",
        "code",
        "ask_count",
        "feedback_ask_count",
        "diagnosis_count",
        "excluded_codes",
        "risk_level",
    )
    differences = {
        key: {"rule": rule_actual.get(key), "llm": llm_actual.get(key)}
        for key in keys
        if rule_actual.get(key) != llm_actual.get(key)
    }
    return {
        "changed": bool(differences),
        "differences": differences,
    }


def _call_summary(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return "无（确定性前置节点未调用 LLM）"
    parts = []
    for call in calls:
        raw = call.get("raw")
        status = "ok" if call.get("ok") else "FAILED"
        parts.append(f"{call['kind']}:{status} {_short(raw, 180)}")
    return " | ".join(parts)


def _print_case(
    item: dict[str, Any],
    *,
    verbose: bool,
) -> None:
    rule = item["rule"]
    llm = item["llm"]
    actual_rule = rule.get("actual", {})
    actual_llm = llm.get("actual", {})
    changed = item["delta"]["changed"]
    status = "DIFF" if changed else "SAME"
    if llm.get("passed"):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    print(
        f"[{verdict} {status}] {item['id']} "
        f"rule={actual_rule.get('final_action')}/"
        f"{actual_rule.get('reason_code') or '-'}/"
        f"{actual_rule.get('code') or '-'}/"
        f"asks={actual_rule.get('ask_count')} | "
        f"llm={actual_llm.get('final_action')}/"
        f"{actual_llm.get('reason_code') or '-'}/"
        f"{actual_llm.get('code') or '-'}/"
        f"asks={actual_llm.get('ask_count')}"
    )
    if changed:
        for key, values in item["delta"]["differences"].items():
            print(f"    Δ{key}: rule={values['rule']!r} -> llm={values['llm']!r}")
    if verbose or changed or not llm.get("passed"):
        for call in llm.get("calls", []):
            print(f"    LLM {call['kind']} raw: {_short(call.get('raw'))}")
        for failure in llm.get("failures", []):
            print(
                f"    ! {failure['path']}: "
                f"expected={_short(failure['expected'])}, "
                f"actual={_short(failure['actual'])}"
            )
    else:
        print(f"    LLM calls: {_call_summary(llm.get('calls', []))}")


def _build_report(
    cases_path: Path,
    manual_path: Path,
    model: str,
    cases: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, Any]:
    rule_passed = sum(1 for item in items if item["rule"].get("passed"))
    llm_passed = sum(1 for item in items if item["llm"].get("passed"))
    changed = sum(1 for item in items if item["delta"]["changed"])
    call_errors = sum(
        1
        for item in items
        for call in item["llm"].get("calls", [])
        if not call.get("ok")
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "llm_vs_rule",
        "model": model,
        "cases_file": str(cases_path),
        "manual": str(manual_path),
        "total": len(items),
        "summary": {
            "rule_passed": rule_passed,
            "rule_failed": len(items) - rule_passed,
            "llm_passed": llm_passed,
            "llm_failed": len(items) - llm_passed,
            "behavior_changed": changed,
            "llm_call_errors": call_errors,
        },
        "cases": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用真实 DeepSeek 通道复跑 34 条用例，并与规则通道对比"
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES),
        help="用例 JSON 路径（默认：evaluation/test_cases.json）",
    )
    parser.add_argument(
        "--manual",
        default=str(DEFAULT_MANUAL),
        help="维修手册路径（默认：项目根目录 printer_LX2000.md）",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
        help="只运行指定用例 ID，可重复使用",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="完整 JSON 报告输出路径（默认：evaluation/llm_report.json）",
    )
    parser.add_argument("--json", action="store_true", help="把完整报告输出到 stdout")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="即使行为相同，也打印每条 LLM 原始输出",
    )
    return parser


def main() -> int:
    _configure_output()
    _load_dotenv()
    args = build_parser().parse_args()

    cases_path = Path(args.cases).expanduser().resolve()
    manual_path = Path(args.manual).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    cases, _meta, suite_errors, _counts = _load_suite(cases_path)
    if not manual_path.exists():
        suite_errors.append(f"找不到维修手册：{manual_path}")

    if args.case_ids:
        known_ids = {case.get("id") for case in cases}
        unknown_ids = sorted(set(args.case_ids) - known_ids)
        if unknown_ids:
            suite_errors.append("未知用例 ID：" + "、".join(unknown_ids))
        selected = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in selected]

    if suite_errors:
        print("用例集校验失败：")
        for error in suite_errors:
            print(f"  - {error}")
        return 2

    extractor, llm_note = build_default_risk_extractor()
    if extractor is None:
        print(f"无法启用真实 LLM：{llm_note}")
        print("请确认 DEEPSEEK_API_KEY 已配置、openai SDK 已安装，且未设置 TRIAGE_DISABLE_LLM。")
        return 2
    if not getattr(extractor, "enabled", False):
        print("DeepSeek 客户端未启用，无法进行 LLM 通道评测。")
        return 2

    recorder = RecordingExtractor(extractor)
    items: list[dict[str, Any]] = []

    for case in cases:
        rule_result = _run_rule_case(case, manual_path)
        llm_result = _run_llm_case(
            case,
            manual_path,
            recorder,
            f"DeepSeek {getattr(extractor, 'model', 'deepseek-chat')}",
        )
        items.append(
            {
                "id": case["id"],
                "category": case["category"],
                "description": case.get("description", ""),
                "messages": case["messages"],
                "expected": case["expected"],
                "rule": rule_result,
                "llm": llm_result,
                "delta": _behavior_delta(rule_result, llm_result),
            }
        )

    report = _build_report(
        cases_path,
        manual_path,
        getattr(extractor, "model", "deepseek-chat"),
        cases,
        items,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = report["summary"]
        print("=== LX-2000 售后分诊：规则通道 vs 真实 LLM 通道 ===")
        print(f"模型：{report['model']} | 用例：{report['total']}")
        print(
            f"规则通过：{summary['rule_passed']}/{report['total']} | "
            f"LLM 通过：{summary['llm_passed']}/{report['total']} | "
            f"行为变化：{summary['behavior_changed']} | "
            f"LLM 调用失败：{summary['llm_call_errors']}"
        )
        print("说明：LLM 只输出风险/现象抽取 JSON；追问/诊断/转人工由确定性状态机决定。")
        print(f"完整报告：{report_path}")
        print("-" * 78)
        for item in items:
            _print_case(item, verbose=args.verbose)
        print("-" * 78)
        print(
            f"结果：规则 {summary['rule_passed']}/{report['total']}，"
            f"LLM {summary['llm_passed']}/{report['total']}，"
            f"行为变化 {summary['behavior_changed']} 条"
        )

    return 0 if report["summary"]["llm_passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
