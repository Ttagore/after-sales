#!/usr/bin/env python3
"""Behavioral evaluation for the LX-2000 after-sales triage agent.

The suite is deterministic: every case is run with ``TriageAgent(use_llm=False)``
so keyword extraction, risk rules, retrieval and handoff decisions are
reproducible.  The evaluator deliberately compares the structured fields of
``TurnResult`` -- ``action``, ``ticket.reason_code``, ``risk.level`` and
``sources`` -- rather than guessing from words inside the answer text.

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --json
    python evaluation/evaluate.py --case RISK-01 --case NOMATCH-01
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_workflow import TriageAgent  # noqa: E402
from taxonomy import (  # noqa: E402
    MAX_FAULT_ASKS,
    MAX_FEEDBACK_ASKS,
    MAX_MODEL_ASKS,
    MAX_TOTAL_FOLLOW_UPS,
)


DEFAULT_CASES = Path(__file__).with_name("test_cases.json")
DEFAULT_MANUAL = PROJECT_ROOT / "printer_LX2000.md"

EXPECTED_TOTAL = 34
VALID_ACTIONS = {"ask", "diagnose", "guide", "handoff", "resolved"}
CATEGORY_LABELS = {
    "insufficient_info": "信息不足",
    "sufficient_info": "信息足够",
    "high_risk": "高风险",
    "unsupported_or_no_match": "不支持/无匹配",
    "feedback_loop": "诊断后反馈闭环",
}
REQUIRED_CATEGORIES = tuple(CATEGORY_LABELS)

# The same product rule used by the workflow: never promise free repair,
# warranty coverage, price, compensation, etc.
COST_OR_PROMISE = re.compile(
    r"免费|保修|包修|价格|报价|收费|费用|赔付|赔偿|退款|\d+\s*元|保证|承诺|百分百|一定能"
)

TICKET_REQUIRED_FIELDS = (
    "ticket_id",
    "reason_code",
    "handoff_reason",
    "recommended_queue",
    "user_description",
    "user_messages",
    "extracted",
    "risk",
    "candidate_documents",
)


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _short(value: Any, limit: int = 220) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _add_failure(
    failures: list[dict[str, Any]], path: str, expected: Any, actual: Any
) -> None:
    failures.append({"path": path, "expected": expected, "actual": actual})


def _load_suite(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str], Counter[str]]:
    """Load and structurally validate the case file."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return [], {}, [f"找不到用例文件：{path}"], Counter()
    except json.JSONDecodeError as exc:
        return [], {}, [f"用例文件不是合法 JSON：{exc}"], Counter()
    except OSError as exc:
        return [], {}, [f"读取用例文件失败：{exc}"], Counter()

    if isinstance(raw, dict):
        cases = raw.get("cases")
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    else:
        cases = raw
        meta = {}

    errors: list[str] = []
    if not isinstance(cases, list):
        return [], meta, ["用例文件缺少 cases 数组"], Counter()

    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            errors.append(f"第 {index} 条用例不是对象")
            continue

        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"第 {index} 条用例缺少 id")
        elif case_id in seen_ids:
            errors.append(f"用例 id 重复：{case_id}")
        else:
            seen_ids.add(case_id)

        category = case.get("category")
        if category not in CATEGORY_LABELS:
            errors.append(f"{case_id or index}: category 非法：{category!r}")
        else:
            counts[category] += 1

        messages = case.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or not all(isinstance(item, str) and item.strip() for item in messages)
        ):
            errors.append(f"{case_id or index}: messages 必须是非空字符串数组")

        expected = case.get("expected")
        if not isinstance(expected, dict):
            errors.append(f"{case_id or index}: 缺少 expected 对象")
            continue

        final_action = expected.get("final_action")
        if final_action not in VALID_ACTIONS:
            errors.append(
                f"{case_id or index}: expected.final_action 非法：{final_action!r}"
            )

        turn_actions = expected.get("turn_actions")
        if turn_actions is not None:
            if not isinstance(turn_actions, list) or not all(
                action in VALID_ACTIONS for action in turn_actions
            ):
                errors.append(
                    f"{case_id or index}: expected.turn_actions 含非法动作"
                )
            elif turn_actions and turn_actions[-1] != final_action:
                errors.append(
                    f"{case_id or index}: turn_actions 最后一项与 final_action 不一致"
                )

    if len(cases) != EXPECTED_TOTAL:
        errors.append(
            f"用例数量应为 {EXPECTED_TOTAL}，实际为 {len(cases)}"
        )

    meta_total = meta.get("total")
    if meta_total is not None and meta_total != len(cases):
        errors.append(
            f"meta.total={meta_total!r} 与实际用例数 {len(cases)} 不一致"
        )

    missing_categories = [
        CATEGORY_LABELS[category]
        for category in REQUIRED_CATEGORIES
        if counts[category] == 0
    ]
    if missing_categories:
        errors.append("缺少场景类别：" + "、".join(missing_categories))

    return cases, meta, errors, counts


def _actual_snapshot(results: list[Any]) -> dict[str, Any]:
    """Extract the fields the evaluator is allowed to judge.

    ``action`` is the authoritative behavior signal.  Message wording is only
    checked when a case explicitly asks for a stable phrase.
    """

    last = results[-1]
    ticket = last.ticket if isinstance(last.ticket, dict) else {}
    extracted = ticket.get("extracted")
    if not isinstance(extracted, dict):
        extracted = last.info if isinstance(last.info, dict) else {}
    sources = last.sources if isinstance(last.sources, list) else []
    candidates = ticket.get("candidate_documents")
    if not isinstance(candidates, list):
        candidates = []

    model_ask_count = sum(
        1
        for result in results
        for step in result.stage_trace
        if step.startswith("C:ask(model#")
    )
    fault_ask_count = sum(
        1
        for result in results
        for step in result.stage_trace
        if step.startswith("C:ask(fault#")
    )
    feedback_ask_count = sum(
        1
        for result in results
        for step in result.stage_trace
        if step.startswith("C:ask(feedback#")
    )
    excluded_codes: list[str] = []
    for result in results:
        debug = result.debug if isinstance(result.debug, dict) else {}
        if isinstance(debug.get("excluded_codes"), list):
            excluded_codes = list(debug["excluded_codes"])
        ticket_extracted = (
            result.ticket.get("extracted")
            if isinstance(result.ticket, dict)
            else None
        )
        if isinstance(ticket_extracted, dict) and isinstance(
            ticket_extracted.get("excluded_codes"), list
        ):
            excluded_codes = list(ticket_extracted["excluded_codes"])

    return {
        "turn_actions": [result.action for result in results],
        "final_action": last.action,
        "ask_count": sum(1 for result in results if result.action == "ask"),
        "model_ask_count": model_ask_count,
        "fault_ask_count": fault_ask_count,
        "feedback_ask_count": feedback_ask_count,
        "diagnosis_count": sum(
            1 for result in results if result.action == "diagnose"
        ),
        "excluded_codes": excluded_codes,
        "reason_code": ticket.get("reason_code"),
        "follow_up_count": extracted.get("follow_up_count"),
        "asked_slots_count": (
            len(extracted.get("asked_slots", []))
            if isinstance(extracted.get("asked_slots"), list)
            else None
        ),
        "risk_level": (last.risk or {}).get("level")
        if isinstance(last.risk, dict)
        else None,
        "queue": ticket.get("recommended_queue"),
        "out_of_scope": extracted.get("out_of_scope"),
        "code": sources[0].get("code") if sources else None,
        "source_count": len(sources),
        "candidate_codes": [
            candidate.get("code") if isinstance(candidate, dict) else None
            for candidate in candidates
        ],
        "ticket_present": isinstance(last.ticket, dict),
        "message": last.message,
        "stage_trace": list(last.stage_trace),
    }


def _check_message_order(
    message: str,
    tokens: list[str],
    failures: list[dict[str, Any]],
) -> None:
    cursor = -1
    for token in tokens:
        index = message.find(token, cursor + 1)
        if index < 0:
            _add_failure(
                failures,
                "message_order",
                f"按顺序出现：{tokens}",
                f"缺少或顺序错误：{token}",
            )
            return
        cursor = index


def _check_expectations(
    expected: dict[str, Any],
    actual: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    exact_fields = (
        "turn_actions",
        "final_action",
        "ask_count",
        "model_ask_count",
        "fault_ask_count",
        "feedback_ask_count",
        "diagnosis_count",
        "excluded_codes",
        "reason_code",
        "follow_up_count",
        "asked_slots_count",
        "risk_level",
        "queue",
        "out_of_scope",
        "code",
    )
    for field in exact_fields:
        if field not in expected:
            continue
        if actual.get(field) != expected[field]:
            _add_failure(failures, field, expected[field], actual.get(field))

    if expected.get("require_sources") and actual["source_count"] == 0:
        _add_failure(failures, "require_sources", True, actual["source_count"])

    if "candidate_codes" in expected:
        expected_codes = sorted(
            "None" if code is None else str(code)
            for code in expected["candidate_codes"]
        )
        actual_codes = sorted(
            "None" if code is None else str(code)
            for code in actual["candidate_codes"]
        )
        if actual_codes != expected_codes:
            _add_failure(
                failures,
                "candidate_codes",
                expected_codes,
                actual_codes,
            )

    message = actual["message"]
    for token in expected.get("message_contains", []):
        if token not in message:
            _add_failure(failures, "message_contains", token, message)
    for token in expected.get("message_not_contains", []):
        if token in message:
            _add_failure(failures, "message_not_contains", token, message)
    if expected.get("message_order"):
        _check_message_order(message, expected["message_order"], failures)


def _check_invariants(
    results: list[Any],
    actual: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    """Checks that apply to every case, independent of its expected block."""

    for index, result in enumerate(results, start=1):
        if result.action not in VALID_ACTIONS:
            _add_failure(
                failures,
                f"turn[{index}].action",
                sorted(VALID_ACTIONS),
                result.action,
            )
        if COST_OR_PROMISE.search(result.message or ""):
            _add_failure(
                failures,
                f"turn[{index}].message.cost_or_promise",
                "不包含费用/保修承诺",
                result.message,
            )

    if actual["model_ask_count"] > MAX_MODEL_ASKS:
        _add_failure(
            failures,
            "model_ask_count",
            f"<= {MAX_MODEL_ASKS}",
            actual["model_ask_count"],
        )
    if actual["fault_ask_count"] > MAX_FAULT_ASKS:
        _add_failure(
            failures,
            "fault_ask_count",
            f"<= {MAX_FAULT_ASKS}",
            actual["fault_ask_count"],
        )
    if actual["feedback_ask_count"] > MAX_FEEDBACK_ASKS:
        _add_failure(
            failures,
            "feedback_ask_count",
            f"<= {MAX_FEEDBACK_ASKS}",
            actual["feedback_ask_count"],
        )
    if actual["ask_count"] > MAX_TOTAL_FOLLOW_UPS:
        _add_failure(
            failures,
            "ask_count",
            f"<= {MAX_TOTAL_FOLLOW_UPS}",
            actual["ask_count"],
        )

    # Only an explicit resolution or a handoff ends the session.  A diagnosis
    # keeps the follow-up loop open so the user can deny it or report a new
    # symptom; any later message would otherwise silently start a new session.
    for index, result in enumerate(results[:-1], start=1):
        if result.action in {"handoff", "resolved"}:
            _add_failure(
                failures,
                "premature_session_end",
                "最后一轮才结束会话",
                f"第 {index} 轮已结束（{result.action}）",
            )

    if actual["final_action"] == "diagnose":
        if actual["source_count"] == 0:
            _add_failure(failures, "diagnose.sources", "至少 1 个来源", 0)
        if "H:answer_grounded" not in actual["stage_trace"]:
            _add_failure(
                failures,
                "diagnose.stage_trace",
                "包含 H:answer_grounded",
                actual["stage_trace"],
            )

    if actual["final_action"] == "handoff":
        if not actual["ticket_present"]:
            _add_failure(failures, "handoff.ticket", "交接单非空", None)
            return

        ticket = results[-1].ticket
        for field in TICKET_REQUIRED_FIELDS:
            if field not in ticket:
                _add_failure(
                    failures,
                    f"ticket.{field}",
                    "字段存在",
                    "缺失",
                )

        extracted = ticket.get("extracted")
        if not isinstance(extracted, dict):
            _add_failure(failures, "ticket.extracted", "对象", extracted)
            return

        if extracted.get("follow_up_count") != actual["ask_count"]:
            _add_failure(
                failures,
                "ticket.extracted.follow_up_count",
                actual["ask_count"],
                extracted.get("follow_up_count"),
            )
        if extracted.get("model_ask_count") != actual["model_ask_count"]:
            _add_failure(
                failures,
                "ticket.extracted.model_ask_count",
                actual["model_ask_count"],
                extracted.get("model_ask_count"),
            )
        if extracted.get("fault_ask_count") != actual["fault_ask_count"]:
            _add_failure(
                failures,
                "ticket.extracted.fault_ask_count",
                actual["fault_ask_count"],
                extracted.get("fault_ask_count"),
            )
        if extracted.get("feedback_ask_count") != actual["feedback_ask_count"]:
            _add_failure(
                failures,
                "ticket.extracted.feedback_ask_count",
                actual["feedback_ask_count"],
                extracted.get("feedback_ask_count"),
            )
        asked_slots = extracted.get("asked_slots")
        if not isinstance(asked_slots, list) or len(asked_slots) != actual["ask_count"]:
            _add_failure(
                failures,
                "ticket.extracted.asked_slots",
                f"长度 {actual['ask_count']}",
                asked_slots,
            )


def _run_case(
    case: dict[str, Any], manual_path: Path
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    results: list[Any] = []

    try:
        agent = TriageAgent(manual_path, use_llm=False)
    except Exception as exc:
        _add_failure(
            failures,
            "agent.init",
            "TriageAgent 初始化成功",
            f"{type(exc).__name__}: {exc}",
        )
        return {
            "id": case.get("id"),
            "category": case.get("category"),
            "description": case.get("description", ""),
            "passed": False,
            "actual": {},
            "failures": failures,
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
            "id": case.get("id"),
            "category": case.get("category"),
            "description": case.get("description", ""),
            "passed": False,
            "actual": {},
            "failures": failures,
        }

    actual = _actual_snapshot(results)
    _check_expectations(case["expected"], actual, failures)
    _check_invariants(results, actual, failures)

    return {
        "id": case.get("id"),
        "category": case.get("category"),
        "description": case.get("description", ""),
        "passed": not failures,
        "actual": actual,
        "failures": failures,
    }


def _print_suite_errors(errors: list[str]) -> None:
    print("用例集校验失败：")
    for error in errors:
        print(f"  - {error}")


def _print_human_report(
    cases_path: Path,
    cases: list[dict[str, Any]],
    counts: Counter[str],
    results: list[dict[str, Any]],
) -> None:
    print("=== LX-2000 售后分诊行为评测 ===")
    print(f"用例文件：{cases_path}")
    print(
        "用例：{}（{}）".format(
            len(cases),
            " / ".join(
                f"{CATEGORY_LABELS[category]} {counts[category]}"
                for category in REQUIRED_CATEGORIES
            ),
        )
    )
    print("-" * 78)

    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        actual = item["actual"]
        if actual:
            summary = (
                f"actions={actual['turn_actions']} "
                f"reason={actual['reason_code']} "
                f"code={actual['code']} "
                f"risk={actual['risk_level']} "
                f"asks={actual['ask_count']}"
            )
        else:
            summary = "no result"

        label = CATEGORY_LABELS.get(item["category"], item["category"])
        print(f"[{status}] {item['id']} ({label}) {summary}")
        if item["description"]:
            print(f"       {item['description']}")
        for failure in item["failures"]:
            print(
                f"       - {failure['path']}: "
                f"expected={_short(failure['expected'])}, "
                f"actual={_short(failure['actual'])}"
            )

    passed = sum(1 for item in results if item["passed"])
    print("-" * 78)
    print(f"结果：{passed}/{len(results)} 通过")
    if passed != len(results):
        print("失败用例见上方 FAIL 行。")


def _build_json_report(
    cases_path: Path,
    manual_path: Path,
    cases: list[dict[str, Any]],
    counts: Counter[str],
    results: list[dict[str, Any]],
    suite_errors: list[str],
) -> dict[str, Any]:
    passed = sum(1 for item in results if item["passed"])
    return {
        "suite": {
            "cases_file": str(cases_path),
            "manual": str(manual_path),
            "total": len(cases),
            "counts": dict(counts),
            "errors": suite_errors,
        },
        "results": results,
        "summary": {
            "passed": passed,
            "failed": len(results) - passed,
            "total": len(results),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 LX-2000 售后分诊的 34 条行为评测用例"
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
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    return parser


def main() -> int:
    _configure_output()
    args = build_parser().parse_args()

    cases_path = Path(args.cases).expanduser().resolve()
    manual_path = Path(args.manual).expanduser().resolve()
    cases, _meta, suite_errors, counts = _load_suite(cases_path)

    if not manual_path.exists():
        suite_errors.append(f"找不到维修手册：{manual_path}")

    if args.case_ids:
        known_ids = {case.get("id") for case in cases}
        unknown_ids = sorted(set(args.case_ids) - known_ids)
        if unknown_ids:
            suite_errors.append("未知用例 ID：" + "、".join(unknown_ids))
        selected = set(args.case_ids)
        cases = [case for case in cases if case.get("id") in selected]
        counts = Counter(case.get("category") for case in cases)

    if suite_errors:
        if args.json:
            print(
                json.dumps(
                    _build_json_report(
                        cases_path,
                        manual_path,
                        cases,
                        counts,
                        [],
                        suite_errors,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            _print_suite_errors(suite_errors)
        return 2

    results = [
        _run_case(case, manual_path)
        for case in cases
    ]

    if args.json:
        print(
            json.dumps(
                _build_json_report(
                    cases_path,
                    manual_path,
                    cases,
                    counts,
                    results,
                    [],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_human_report(cases_path, cases, counts, results)

    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
