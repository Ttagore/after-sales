#!/usr/bin/env python3
"""Command-line entry point for the LX-2000 after-sales triage assistant.

Examples:
    python main.py                  # interactive, uses DeepSeek if configured
    python main.py --no-llm         # keyword-only, fully offline
    python main.py --demo           # scripted multi-turn scenarios
    python main.py --once "冒烟了"   # one turn, useful in scripts
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from agent_workflow import TriageAgent
from llm_client import build_default_risk_extractor
from schemas import TurnResult
from taxonomy import MAX_FAULT_ASKS, MAX_FEEDBACK_ASKS, MAX_MODEL_ASKS


DEMO_SCENARIOS: list[tuple[str, list[str]]] = [
    ("高风险：冒烟，立即转人工", ["冒烟了"]),
    (
        "信息不足：型号与现象各自追问 2 次后转人工",
        ["机器坏了", "LX-2000", "就是坏了", "不知道"],
    ),
    ("正常诊断：卡纸", ["LX-2000 卡纸了"]),
    ("历史合并：肯定词消解型号", ["打到一半出不来了", "是"]),
    (
        "软降级：型号问不出仍可诊断",
        ["机器坏了", "去年买的", "卡纸", "不知道"],
    ),
    ("冲突：代码与现象不一致", ["LX-2000 显示E01不出纸"]),
    ("超范围型号", ["HP 打印机卡纸了"]),
    ("用户明确要求人工", ["我要人工"]),
    ("否定风险信号不应误报", ["没有冒烟，LX-2000 卡纸了"]),
    (
        "诊断后否定反馈：排除 E08 并重新匹配",
        ["LX-2000 打印一半停了", "没提示纸张错"],
    ),
    (
        "诊断后新现象：从卡纸改判异响",
        ["LX-2000 卡纸了", "没卡纸，就是一直响"],
    ),
    (
        "诊断后进度推进：不重新检索，继续下一步",
        ["LX-2000 打印一半停了", "没显示纸张不对", "拿出来了，然后呢，怎么做"],
    ),
    ("诊断后确认解决：会话正常结束", ["LX-2000 卡纸了", "解决了"]),
]


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _load_dotenv() -> None:
    """Load the project-local .env without overriding real environment vars."""

    path = Path(__file__).with_name(".env")
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


def _make_ticket_sink(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)

    def save(ticket: dict) -> None:
        target = directory / f"{ticket['ticket_id']}.json"
        target.write_text(
            json.dumps(ticket, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return save


def _source_lines(result: TurnResult) -> list[str]:
    lines: list[str] = []
    for source in result.sources:
        citation = source["citation"]
        if citation not in result.message:
            lines.append(f"  来源：{citation}")
    return lines


def _follow_up_heading(stage_trace: list[str]) -> str | None:
    """Translate the last per-slot follow-up trace into a display heading."""

    heading = None
    for step in stage_trace:
        model_match = re.fullmatch(r"C:ask\(model#(\d+)\)", step)
        if model_match:
            heading = (
                f"【追问：设备型号 {model_match.group(1)}/{MAX_MODEL_ASKS}】"
            )
            continue
        feedback_match = re.fullmatch(r"C:ask\(feedback#(\d+)\)", step)
        if feedback_match:
            heading = (
                "【追问：诊断反馈 "
                f"{feedback_match.group(1)}/{MAX_FEEDBACK_ASKS}】"
            )
            continue
        conflict_match = re.fullmatch(r"C:ask\(fault#(\d+),conflict\)", step)
        if conflict_match:
            heading = "【追问：故障信息澄清】"
            continue
        fault_match = re.fullmatch(r"C:ask\(fault#(\d+)\)", step)
        if fault_match:
            heading = (
                f"【追问：故障现象 {fault_match.group(1)}/{MAX_FAULT_ASKS}】"
            )
    return heading


def print_result(result: TurnResult, *, trace: bool, show_ticket: bool) -> None:
    follow_up_heading = _follow_up_heading(result.stage_trace)

    if result.action == "ask":
        heading = follow_up_heading or "【追问】"
    elif result.action == "diagnose":
        heading = "【诊断建议】"
    elif result.action == "guide":
        heading = "【下一步指导】"
    elif result.action == "resolved":
        heading = "【已解决】"
    else:
        heading = "【转人工】"

    print(f"\n{heading}\n{result.message}")

    for line in _source_lines(result):
        print(line)

    if result.candidates and result.action == "diagnose":
        winner_ids = {source["source_id"] for source in result.sources}
        others = [
            candidate
            for candidate in result.candidates
            if candidate["source_id"] not in winner_ids and candidate["applicable"]
        ]
        if others:
            top_score = max(candidate["retrieval_score"] for candidate in others)
            has_categories = bool(result.info.get("symptom_categories"))
            plausible = [
                candidate
                for candidate in others
                if candidate["retrieval_score"] >= max(0.05, top_score * 0.5)
                and (
                    not has_categories
                    or candidate["category_match"] > 0
                    or candidate["code_match"]
                )
            ][:3]
            if plausible:
                print("  其他候选：")
                for candidate in plausible:
                    code = candidate["code"] or "无代码"
                    print(
                        f"    - {code} {candidate['title']} "
                        f"(相关度 {candidate['retrieval_score']:.2f})"
                    )

    if result.ticket:
        ticket = result.ticket
        print(
            f"\n交接单：{ticket['ticket_id']} | 原因：{ticket['handoff_reason']} | "
            f"队列：{ticket['recommended_queue']}"
        )
        if ticket["candidate_documents"]:
            codes = "、".join(
                candidate["code"] or candidate["source_id"]
                for candidate in ticket["candidate_documents"]
            )
            print(f"候选文档：{codes}")
        if show_ticket:
            print(json.dumps(ticket, ensure_ascii=False, indent=2))

    if trace:
        print(f"状态轨迹：{' → '.join(result.stage_trace)}")


def _build_agent(args: argparse.Namespace, *, demo: bool) -> TriageAgent:
    manual = Path(args.manual).expanduser().resolve()
    if not manual.exists():
        raise SystemExit(f"找不到维修手册：{manual}")

    use_llm = not args.no_llm and (not demo or args.llm)
    extractor = None
    llm_note = "已关闭 LLM，仅使用关键词规则"
    if use_llm:
        extractor, llm_note = build_default_risk_extractor()
        if extractor is None:
            use_llm = False

    sink = _make_ticket_sink(Path(args.save_tickets)) if args.save_tickets else None
    return TriageAgent(
        manual,
        llm_extractor=extractor,
        use_llm=use_llm,
        llm_note=llm_note,
        ticket_sink=sink,
    )


def run_once(args: argparse.Namespace) -> int:
    agent = _build_agent(args, demo=False)
    result = agent.handle(args.once)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print_result(result, trace=args.trace, show_ticket=args.show_ticket)
    return 0


def run_demo(args: argparse.Namespace) -> int:
    agent = _build_agent(args, demo=True)
    print(f"LX-2000 售后分诊演示 | 会话 {agent.session_id}")
    print(f"LLM：{agent.llm_note}")
    for index, (title, messages) in enumerate(DEMO_SCENARIOS, start=1):
        print("\n" + "=" * 72)
        print(f"场景 {index}：{title}")
        agent.reset()
        for message in messages:
            print(f"\n用户> {message}")
            result = agent.handle(message)
            print_result(result, trace=args.trace, show_ticket=args.show_ticket)
        print("\n" + "=" * 72)
    return 0


def run_interactive(args: argparse.Namespace) -> int:
    agent = _build_agent(args, demo=False)
    print("LX-2000 售后分诊助手")
    print(f"手册：printer_LX2000.md | 会话 {agent.session_id}")
    print(f"LLM：{agent.llm_note}")
    print("输入 /new 新建会话，/quit 退出。")

    while True:
        try:
            text = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not text:
            continue
        if text.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("已退出。")
            return 0
        if text.lower() == "/new":
            agent.reset()
            print(f"已新建会话 {agent.session_id}。")
            continue
        if text.lower() == "/help":
            print("描述打印机现象即可；可随时说“我要人工”。/new 新建会话，/quit 退出。")
            continue

        result = agent.handle(text)
        print_result(result, trace=args.trace, show_ticket=args.show_ticket)
        if result.action in {"diagnose", "guide"}:
            # Keep the session open: the user may deny the conclusion, report a
            # new symptom, ask for the next step, or confirm resolution.
            print(
                "\n如果以上判断不符合实际，或按步骤操作后仍然不行，"
                "请直接继续输入反馈；也可以说“拿出来了，然后呢”继续问下一步；"
                "问题解决可直接说“解决了”。"
            )
        elif result.action in {"handoff", "resolved"}:
            print("\n本轮会话已结束；继续输入将自动开启新会话。")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LX-2000 打印机售后工单分诊助手（RAG + 规则风控）"
    )
    parser.add_argument(
        "--manual",
        default=str(Path(__file__).with_name("printer_LX2000.md")),
        help="维修手册路径（默认：同目录 printer_LX2000.md）",
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="关闭 DeepSeek，仅使用关键词规则"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="演示模式下也启用 DeepSeek（默认演示只用关键词通道）",
    )
    parser.add_argument("--demo", action="store_true", help="运行内置多轮演示场景")
    parser.add_argument("--once", metavar="TEXT", help="只处理一轮用户输入后退出")
    parser.add_argument("--json", action="store_true", help="与 --once 搭配，输出 JSON")
    parser.add_argument("--trace", action="store_true", help="显示状态机轨迹")
    parser.add_argument("--show-ticket", action="store_true", help="打印完整交接单 JSON")
    parser.add_argument(
        "--save-tickets", metavar="DIR", help="把交接单 JSON 写入指定目录"
    )
    return parser


def main() -> int:
    _configure_output()
    _load_dotenv()
    args = build_parser().parse_args()
    if args.once:
        return run_once(args)
    if args.demo:
        return run_demo(args)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
