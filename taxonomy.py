"""Shared language rules for the LX-2000 triage assistant.

All matching is deterministic and offline.  The rules here only *extract*
observable facts (model, error code, symptom, explicit handoff request); they
do not decide whether a case is safe or which manual section is applicable.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable


SUPPORTED_MODEL = "LX-2000"
# Exact-match key after normalization (lowercase, whitespace/hyphens removed).
_SUPPORTED_MODEL_KEY = "lx2000"

# Follow-ups are budgeted independently per slot.  There is deliberately no
# global "ask_count" used for routing: having already asked about the model
# must never prevent a final symptom answer from being used.
MAX_MODEL_ASKS = 2
MAX_FAULT_ASKS = 2
# Post-diagnosis feedback is a third, independent clarification budget.  It
# keeps the "diagnose -> user feedback -> re-check" loop bounded.
MAX_FEEDBACK_ASKS = 2
# Derived only for audit/evaluation bounds; never used as a routing gate.
MAX_TOTAL_FOLLOW_UPS = MAX_MODEL_ASKS + MAX_FAULT_ASKS + MAX_FEEDBACK_ASKS

# Categories that describe a sufficiently concrete, observable symptom.
STRONG_SYMPTOM_CATEGORIES = frozenset(
    {
        "no_power",
        "paper_jam",
        "paper_feed",
        "no_print_response",
        "print_quality",
        "toner",
        "noise",
        "connectivity",
        "wireless",
        "cover",
        "paper_mismatch",
        "unsupported_function",
    }
)

WEAK_SYMPTOM_CATEGORIES = frozenset({"paper_output", "indicator", "vague"})

SYMPTOM_LABELS = {
    "no_power": "无法开机/不通电",
    "paper_jam": "卡纸/纸张卡住",
    "paper_feed": "进纸/缺纸检测异常",
    "paper_output": "不出纸（信息较宽泛）",
    "no_print_response": "发送打印后设备无响应",
    "print_quality": "打印字迹/图像异常",
    "toner": "墨粉/耗材相关",
    "noise": "异响/震动",
    "connectivity": "USB/网络连接异常",
    "wireless": "无线/手机连接",
    "cover": "盖板/舱门",
    "paper_mismatch": "纸张设置异常/打印中断",
    "unsupported_function": "使用本机不支持的功能",
    "indicator": "仅有指示灯/屏幕提示",
    "vague": "描述笼统",
}


class _PatternGroup:
    """Compile a group of patterns and expose non-negated matches."""

    def __init__(self, patterns: Iterable[str]):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    def find(self, text: str) -> re.Match[str] | None:
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                if not is_negated(text, match.start()):
                    return match
        return None


# A short negation immediately before a keyword, stopping at punctuation.
_NEGATION_TAIL = re.compile(
    r"(?:没有|没|未|无|不|并没有|并未|没有发现|未发现|没发现)"
    r"[^，。；！？,.;!?]{0,4}$"
)


def normalize_text(text: str) -> str:
    """Normalize full-width characters, dashes and whitespace for matching."""

    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("—", "-").replace("–", "-").replace("－", "-")
    text = text.replace("＿", "_")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def compact_text(text: str) -> str:
    """Remove punctuation/whitespace; useful for quote validation."""

    text = normalize_text(text)
    return re.sub(r"[^0-9a-z一-鿿]+", "", text)


def is_negated(text: str, start: int) -> bool:
    """Best-effort local negation check used by keyword extraction.

    This intentionally looks only a few characters back so a negation in an
    earlier clause does not suppress a later, separate warning.
    """

    prefix = text[max(0, start - 12) : start]
    return bool(_NEGATION_TAIL.search(prefix))


def model_key(value: str | None) -> str | None:
    """Return the exact-match key used by the model scope gate.

    Normalization rules (deliberately simple and deterministic):
    1. NFKC normalization (converts full-width characters such as "ＬＸ－２０００");
    2. lowercase;
    3. remove all whitespace characters;
    4. remove all hyphen variants (``-``, ``－``, ``—``, ``–`` after NFKC).

    The result is compared for *exact equality* with ``"lx2000"``.  There is no
    substring matching: ``LX-2000s`` -> ``lx2000s`` and ``LX-2001`` ->
    ``lx2001`` are both out of scope.
    """

    if not value:
        return None
    return re.sub(r"[\s-]+", "", normalize_text(value)) or None


def normalize_model(value: str | None) -> str | None:
    """Canonicalize a model string for display and retrieval filtering.

    Supported models are returned in their canonical form (``LX-2000``).
    Unsupported values are returned as an uppercase compact token; callers
    that need a scope decision must use :func:`model_key` instead of guessing
    from this display value.
    """

    key = model_key(value)
    if key is None:
        return None
    if key == _SUPPORTED_MODEL_KEY:
        return SUPPORTED_MODEL
    return key.upper()


@dataclass
class ModelDetection:
    model: str | None = None
    unsupported_hint: str | None = None
    conflict: bool = False
    out_of_scope: bool = False


_KNOWN_BRANDS = (
    "hp",
    "惠普",
    "canon",
    "佳能",
    "epson",
    "爱普生",
    "brother",
    "兄弟",
    "lenovo",
    "联想",
    "pantum",
    "奔图",
    "xerox",
    "施乐",
    "samsung",
    "三星",
    "xiaomi",
    "小米",
    "huawei",
    "华为",
    "ricoh",
    "理光",
    "kyocera",
    "京瓷",
    "sharp",
    "夏普",
    "dell",
    "戴尔",
)

_MODEL_CONTEXT = re.compile(
    r"(?:型号|model(?:\s*number)?|机器(?:型号)?(?:是|为|：|:)|"
    r"打印机(?:型号)?(?:是|为|：|:)|设备(?:型号)?(?:是|为|：|:))"
    r"\s*[:：]?\s*([a-z][a-z0-9]*(?:[\s-]?[a-z0-9]+){0,4})",
    re.IGNORECASE,
)

# Model recognition is REGEX-BASED, not LLM-based, because model scope is a
# hard safety gate and must be deterministic, offline, and unit-testable.
# This token pattern intentionally recognizes *any* LX-prefixed model-like
# string (LX + optional separator + digits + optional suffix), not only the
# supported model. detect_model() then compares each token by exact
# normalized equality, so LX-2001 / LX-2000s / LX-20 are out of scope rather
# than being silently treated as LX-2000.
_LX_MODEL_TOKEN = re.compile(
    r"(?<![a-z0-9])lx[\s_-]*\d[a-z0-9_-]*",
    re.IGNORECASE,
)


def _brand_hint(text: str) -> str | None:
    for brand in _KNOWN_BRANDS:
        if brand.isascii():
            if re.search(rf"(?<![a-z0-9]){re.escape(brand)}(?![a-z0-9])", text):
                return brand.upper() if len(brand) <= 3 else brand
        elif brand in text:
            return brand
    return None


def detect_model(text: str) -> ModelDetection:
    """Return a three-state model decision using deterministic regexes.

    Implementation method: regex, not LLM.  The scope gate must not depend on
    model inference or network availability.

    States:
    1. No model-like token at all -> ``ModelDetection()`` (status "unknown").
       The workflow asks for the model.
    2. One or more model-like tokens, none exactly "lx2000" after
       normalization -> ``unsupported_hint`` + ``out_of_scope=True``.
       The workflow hands off immediately and never asks about symptoms.
    3. At least one token exactly equal to "lx2000", with no other model
       mentioned -> ``model=LX-2000`` and the workflow may continue.

    A mix of supported and unsupported models is treated as a conflict
    (existing behavior), because the target device cannot be identified.

    This function scans the *cumulative* user text, since the user may answer
    the model question in any turn.  Once an out-of-scope token appears, it
    remains in the accumulated notes and will still be caught.
    """

    normalized = normalize_text(text)
    lx_tokens = [
        match.group(0).strip() for match in _LX_MODEL_TOKEN.finditer(normalized)
    ]
    supported_lx = [
        token for token in lx_tokens if model_key(token) == _SUPPORTED_MODEL_KEY
    ]
    unsupported_lx = [
        token for token in lx_tokens if model_key(token) != _SUPPORTED_MODEL_KEY
    ]

    # Echo the first unsupported model-like token back to the user.  Uppercase
    # only for display: the scope decision already used model_key().
    unsupported_hint = unsupported_lx[0].upper() if unsupported_lx else None

    # Preserve the existing brand-only path (for example "HP 打印机卡纸了").
    # A brand is model information even when no LX-style token is present.
    brand_hint = _brand_hint(normalized)
    if brand_hint and unsupported_hint is None:
        unsupported_hint = brand_hint

    # Context form such as "型号是 HP M100" when no LX token exists.  This is
    # still regex-based and only supplies the display hint; the exact scope
    # comparison remains model_key(candidate) == "lx2000".
    if not lx_tokens:
        context_match = _MODEL_CONTEXT.search(normalized)
        if context_match:
            candidate = context_match.group(1).strip(" .，,;；")
            if (
                candidate
                and not re.fullmatch(r"e\d{1,2}", candidate)
                and model_key(candidate) != _SUPPORTED_MODEL_KEY
            ):
                unsupported_hint = unsupported_hint or candidate.upper()

    if supported_lx and unsupported_hint:
        return ModelDetection(
            unsupported_hint=unsupported_hint,
            conflict=True,
        )
    if supported_lx:
        return ModelDetection(model=SUPPORTED_MODEL)
    if unsupported_hint:
        return ModelDetection(
            unsupported_hint=unsupported_hint,
            out_of_scope=True,
        )
    return ModelDetection()


_ERROR_CODE = re.compile(r"(?<![a-z0-9])e\s?(\d{1,2})(?!\d)", re.IGNORECASE)


def extract_error_codes(text: str) -> list[str]:
    normalized = normalize_text(text)
    codes = {
        f"E{int(match.group(1)):02d}"
        for match in _ERROR_CODE.finditer(normalized)
        if not is_negated(normalized, match.start())
    }
    return sorted(codes)


# Ordered so the most specific / actionable category appears first.
CATEGORY_PATTERNS: dict[str, _PatternGroup] = {
    "no_power": _PatternGroup(
        [
            r"不亮",
            r"开不了机",
            r"无法开机",
            r"启动不了",
            r"点不亮",
            r"通不了电",
            r"按.{0,8}(?:开关|电源).{0,8}(?:没|无).{0,4}反应",
            r"电源.{0,6}(?:没|无).{0,4}反应",
            r"(?:机器|设备).{0,4}没电",
        ]
    ),
    "paper_jam": _PatternGroup(
        [
            r"卡纸",
            r"纸卡",
            r"卡在.{0,8}(?:里头|里面|内部|纸盒|通道|机器|滚轮|入口|出口)",
            r"拉不出来",
            r"纸张.{0,4}卡",
            r"卷纸",
            r"皱成一团",
            r"纸.{0,4}皱",
            # Paraphrases that do not contain the literal word "卡纸".
            # The example "突然卡住，纸出不来了" must count as a symptom
            # without asking the user to repeat it.
            r"(?:纸|纸张).{0,8}(?:出不来|出不了|下不来|进不去|进不了)",
            r"(?:纸|纸张).{0,8}(?:卡住|卡了|堵住|堵了|堵在)",
            r"(?:卡住|卡了|堵住|堵了).{0,8}(?:纸|纸张)",
            r"(?:打印机|机器).{0,8}(?:卡住|卡了|堵住|堵了)",
        ]
    ),
    "paper_feed": _PatternGroup(
        [
            r"有纸.{0,12}(?:没纸|缺纸)",
            r"说没纸",
            r"提示.{0,6}(?:没纸|缺纸)",
            r"不进纸",
            r"纸盒.{0,10}纸",
        ]
    ),
    "no_print_response": _PatternGroup(
        [
            r"(?:点|按|发送|提交).{0,10}打印.{0,12}(?:没|无).{0,5}(?:动静|反应)",
            r"电脑.{0,12}打印.{0,12}(?:没|无).{0,5}(?:动静|反应)",
            r"打印.{0,10}(?:没|无).{0,5}(?:动静|反应)",
            r"打印任务.{0,8}(?:卡|排队|不动)",
            r"(?:机器|打印机).{0,8}(?:动静|反应).{0,4}(?:都|也)?(?:没有|没)",
            r"一点(?:动静|反应)都(?:没有|没)",
        ]
    ),
    "print_quality": _PatternGroup(
        [
            r"模糊",
            r"不清晰",
            # "不清楚哪坏了/不清楚怎么用" is missing information, not a
            # print-quality symptom.  Only a bare/print-context "不清楚"
            # counts as blurry output.
            r"不清楚(?!哪|什么|怎么|啥|是否|是不是|为何|为什么)",
            r"灰蒙蒙",
            r"黑线",
            r"白块",
            r"白纸",
            r"空白页",
            r"缺字",
            r"字.{0,4}缺",
            r"掉字",
            r"掉粉",
            r"一摸就掉",
            r"字.{0,6}掉",
            r"蹭",
            r"脏",
            r"一道",
            r"色块",
            r"颜色.{0,4}浅",
            r"字.{0,4}浅",
            r"越来越浅",
            r"淡",
        ]
    ),
    "toner": _PatternGroup(
        [r"墨粉", r"碳粉", r"墨盒", r"粉盒", r"耗材", r"加粉", r"没墨", r"快没墨"]
    ),
    "noise": _PatternGroup(
        [r"异响", r"咔咔", r"噪音", r"嗡嗡", r"震动", r"一直响", r"响个不停", r"响"]
    ),
    "connectivity": _PatternGroup(
        [
            r"连不上",
            r"连接不上",
            r"找不到.{0,8}(?:打印机|电脑)",
            r"电脑.{0,10}找不到",
            r"离线",
            r"网络",
            r"网线",
            r"usb",
            r"驱动",
            r"端口",
        ]
    ),
    "wireless": _PatternGroup([r"手机", r"无线", r"wifi", r"wi-fi", r"蓝牙"]),
    "cover": _PatternGroup([r"盖子", r"盖板", r"前盖", r"检修盖", r"舱门", r"卡扣"]),
    "paper_mismatch": _PatternGroup(
        [
            r"纸张.{0,6}(?:不对|尺寸|类型)",
            r"尺寸.{0,4}(?:不对|不符)",
            r"\ba4\b",
            r"\bletter\b",
            # Product-defined aliases for E08.  These phrases describe the
            # print being interrupted midway rather than a literal paper-size
            # error, so they are mapped to E08's manual category.
            r"(?:打(?:印)?(?:到|了)?一半|印(?:到|了)?一半)"
            r"[^，。；！？,.;!?]{0,8}"
            r"(?:停住|停了|停|卡住|卡了|断了|断|出不来了|出不来|不出纸|不动)",
            r"出(?:到|了)?一半[^，。；！？,.;!?]{0,6}"
            r"(?:卡住|卡了|停住|停了|断了|断|出不来|出不来了)",
        ]
    ),
    "unsupported_function": _PatternGroup([r"扫描", r"传真"]),
    "paper_output": _PatternGroup(
        [r"不出纸", r"出不来纸", r"没出纸", r"纸张.{0,4}出不来", r"打不出来", r"不打印", r"打印不了"]
    ),
    "indicator": _PatternGroup(
        [r"灯.{0,8}闪", r"指示灯", r"屏幕.{0,6}(?:亮|显示)", r"一直闪", r"灯一直"]
    ),
    "vague": _PatternGroup(
        [
            r"坏了",
            r"不能用",
            r"不工作",
            r"不好使",
            r"故障",
            r"出问题",
            r"罢工",
            r"刚还好好的",
            r"突然停了",
            r"不动了",
            r"没法用",
            r"没反应",
        ]
    ),
}


def extract_symptom_categories(text: str) -> list[str]:
    normalized = normalize_text(text)
    categories: list[str] = []
    for category, group in CATEGORY_PATTERNS.items():
        if group.find(normalized):
            categories.append(category)
    return categories


# Product-defined symptom -> code alias.  The user does not need to have seen
# the code on screen for these phrases; the manual mapping is deterministic and
# is recorded separately as an *inferred* hint in the ticket.
PRINT_INTERRUPT_HINT_CODE = "E08"

_PRINT_INTERRUPT = _PatternGroup(
    [
        r"(?:打(?:印)?(?:到|了)?一半|印(?:到|了)?一半)"
        r"[^，。；！？,.;!?]{0,8}"
        r"(?:停住|停了|停|卡住|卡了|断了|断|出不来了|出不来|不出纸|不动)",
        r"出(?:到|了)?一半[^，。；！？,.;!?]{0,6}"
        r"(?:卡住|卡了|停住|停了|断了|断|出不来|出不来了)",
    ]
)

# A phrase such as "打到一半停住，纸还皱成一团，又卡下一张" is a paper-jam
# description (E19), not the E08 paper-setting/interruption case.  Do not
# infer E08 when explicit paper-jam evidence is present.
_PRINT_INTERRUPT_PAPER_JAM_EVIDENCE = re.compile(
    r"皱|卡下一张|卡纸|纸卡|拉不出来|纸盒|进纸"
)


def detect_print_interrupt(text: str) -> bool:
    """True for the E08 synonym family, unless paper-jam evidence dominates."""

    normalized = normalize_text(text)
    if not _PRINT_INTERRUPT.find(normalized):
        return False
    return not _PRINT_INTERRUPT_PAPER_JAM_EVIDENCE.search(normalized)


def strong_symptom(categories: Iterable[str]) -> bool:
    return any(category in STRONG_SYMPTOM_CATEGORIES for category in categories)


# "vague" is not a symptom at all. "indicator" only says a light/screen is
# active; it does not identify a fault. Everything else is actionable enough
# to enter retrieval (paper_output is intentionally included even though it
# remains "weak" for retrieval specificity scoring).
_NON_ACTIONABLE_SYMPTOMS = frozenset({"vague", "indicator"})

# Categories the symptom-extraction LLM is allowed to return. "vague" is
# excluded because it cannot help the retrieval applicability gate.
SYMPTOM_EXTRACTION_CATEGORIES = tuple(
    category for category in SYMPTOM_LABELS if category != "vague"
)


def has_actionable_symptom(categories: Iterable[str]) -> bool:
    """True when at least one extracted category describes a real fault."""

    return any(
        category not in _NON_ACTIONABLE_SYMPTOMS for category in categories
    )


_HUMAN_PATTERNS = _PatternGroup(
    [
        r"转人工",
        r"人工客服",
        r"真人客服",
        r"真人",
        r"找人工",
        r"接人工",
        r"人工服务",
        r"人工处理",
        r"售后客服",
        r"联系客服",
        r"要人工",
        r"叫人工",
        r"人工(?!智能|呼吸|干预|合成)",
        r"live\s+agent",
        r"human\s+agent",
        r"real\s+person",
    ]
)
# Refusal immediately before the matched request, e.g. "不要(给我)转人工".
_IMMEDIATE_HUMAN_REFUSAL = re.compile(
    r"(?:不需要|不需|不用|不要|无需|不想|别|甭|不)(?:再)?[^，。；！？,.;!?]{0,4}$"
)


def requests_human(text: str) -> bool:
    """Return True if *any* non-refused human-handoff request appears."""

    normalized = normalize_text(text)
    for pattern in _HUMAN_PATTERNS.patterns:
        for match in pattern.finditer(normalized):
            prefix = normalized[max(0, match.start() - 12) : match.start()]
            if _IMMEDIATE_HUMAN_REFUSAL.search(prefix):
                continue
            return True
    return False


_RESOLVED_REPLY = _PatternGroup(
    [
        r"(?:已经|已)(?:解决|修好|弄好|搞好|恢复(?:正常)?|正常|可以|没问题)",
        r"(?:解决|修好|弄好|搞好|恢复(?:正常)?|正常|可以|没问题|好)了"
        r"(?!点|些|一点|不|吗|没|么)",
        r"没问题(?:了)?",
        r"搞定了|行了|没事了",
        r"谢谢|多谢|感谢",
    ]
)
_CONTRAST = re.compile(r"但是|但|不过|可是|然而|只是")
_STILL_BAD = re.compile(
    r"(?:还是|仍然|依然|依旧|照旧|又|仍旧)?"
    r"(?:不行|不好|没好|没解决|未解决|没修好|没弄好|不管用|不管事|"
    r"没用|没效果|有问题|故障还在|问题还在|老样子|一样)"
)
# Explicit denial of what the assistant just said.  "是不是..." / "对不对..."
# are questions, not denials, so the lookbehind guards those forms.
_NEGATIVE_GENERIC = re.compile(
    r"(?:没|没有|未)(?:显示|提示|出现|看到|发现)"
    r"|(?<!是)(?:不是|并非)(?:这个|这样|这么回事)?"
    r"|(?<!对)不对|不匹配|不符合"
    r"|跟(?:我|我的|我说的)?(?:的)?(?:情况|现象)?(?:不一样|不符)"
    r"|没有(?:你说的)?(?:这个)?(?:现象|情况)"
)
_DENIED_CODE = re.compile(
    r"(?:(?<!是)不是|没有|没显示|没出现|未显示|未出现|并非|而不是)\s*"
    r"e\s?(\d{1,2})",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"[?？]|吗|呢|是不是|有没有|对不对|是否")
# Post-diagnosis progress: the user has completed a step and asks what comes
# next.  These are *not* denials and must not trigger a new retrieval.  The
# positive forms ("拿出来了") are deliberately distinct from failure forms
# such as "拿不出来", which stay in the denial/still-bad path.
_PROGRESS_ACTIONS = (
    (
        "obstruction_removed",
        re.compile(
            r"(?:拿|取|弄|拔|抽|掏)出来了|取出(?:了|卡纸|异物|纸屑)|"
            r"清理(?:了|掉)|清掉了|拿掉了|已经拿出来"
        ),
    ),
    (
        "cover_opened",
        re.compile(
            r"(?:打开|掀开|开启)(?:了|盖子|盖板|纸盒|检修盖)|"
            r"开了(?:盖子|盖板|纸盒|检修盖)"
        ),
    ),
    (
        "button_pressed",
        re.compile(
            r"(?:按了|按下|按过|重启了|关机了|断电了|拔掉电源了|已经重启)"
        ),
    ),
    (
        "generic",
        re.compile(
            r"然后呢|下一步|接下来|还要(?:做|检查|干|注意)(?:什么|啥)|"
            r"怎么(?:做|弄|继续|处理)|如何继续|按钮(?:在哪|在哪里)|"
            r"(?:还|下一步)?(?:要|该)做什么"
        ),
    ),
)
_STUCK_ACTION = re.compile(
    r"(?:拿|取|弄|拔|抽|掏)不出来|打不开|卡住(?:了)?|还是卡|仍然卡"
)
_DENIED_FEATURE = re.compile(
    r"(?:没|没有|未)(?:显示|提示|出现|看到)"
    r"[^，。；！？,.;!?]{0,10}(?:纸张|报错|错误|代码|e\s?\d{1,2}|不对|提示)"
    r"|(?:没有|没|未见|未)(?:纸张|报错|错误|代码)"
    r"[^，。；！？,.;!?]{0,6}(?:提示|显示|报错)",
    re.IGNORECASE,
)
_DENIED_CATEGORY_PATTERNS = {
    "paper_jam": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未|没看到|没发现|没有发现)"
        r"[^，。；！？,.;!?]{0,6}"
        r"(?:卡纸|纸卡|卡住|卡在|皱|拉不出来)"
    ),
    "paper_mismatch": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未)"
        r"[^，。；！？,.;!?]{0,6}"
        r"(?:纸张.{0,2}(?:不对|错|尺寸|类型|设置)|a4|letter)",
        re.IGNORECASE,
    ),
    "noise": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未|(?<!是)不)"
        r"[^，。；！？,.;!?]{0,6}(?:异响|响|噪音|震动)"
    ),
    "no_power": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未)"
        r"[^，。；！？,.;!?]{0,6}(?:不亮|开不了机|电源|通电)"
    ),
    "print_quality": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未)"
        r"[^，。；！？,.;!?]{0,6}(?:黑线|模糊|缺字|白块|字迹)"
    ),
    "connectivity": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未)"
        r"[^，。；！？,.;!?]{0,6}(?:连不上|网络|usb|驱动)"
    ),
    "toner": re.compile(
        r"(?:(?<!是)不是|没有|没|并非|未)"
        r"[^，。；！？,.;!?]{0,6}(?:墨粉|碳粉|墨盒|粉盒|耗材)"
    ),
}


@dataclass
class FeedbackAnalysis:
    """Deterministic interpretation of a post-diagnosis user reply."""

    resolved: bool = False
    negative: bool = False
    is_question: bool = False
    still_bad: bool = False
    denied_feature: bool = False
    denied_codes: list[str] = field(default_factory=list)
    denied_categories: list[str] = field(default_factory=list)
    denied_text: str = ""
    new_categories: list[str] = field(default_factory=list)
    explicit_codes: list[str] = field(default_factory=list)
    # A progress reply means "I did the step, what next?", not a denial.
    # The workflow answers from the current manual chapter without retrieval.
    progress: bool = False
    progress_stage: str | None = None
    progress_text: str = ""


def analyze_feedback(text: str) -> FeedbackAnalysis:
    """Classify a reply to the previous diagnosis.

    This only extracts language evidence.  Whether to re-retrieve, ask a
    clarification or end the session remains a deterministic workflow rule.
    """

    normalized = normalize_text(text)
    denied_codes = sorted(
        {
            f"E{int(match.group(1)):02d}"
            for match in _DENIED_CODE.finditer(normalized)
        }
    )
    denied_categories = [
        category
        for category, pattern in _DENIED_CATEGORY_PATTERNS.items()
        if pattern.search(normalized)
    ]
    still_bad = bool(_STILL_BAD.search(normalized) or _STUCK_ACTION.search(normalized))
    question = bool(_QUESTION.search(normalized))
    denied_feature = bool(_DENIED_FEATURE.search(normalized))
    progress_stage = next(
        (
            name
            for name, pattern in _PROGRESS_ACTIONS
            if pattern.search(normalized)
        ),
        None,
    )
    explicit_negative = bool(
        still_bad or denied_feature or denied_codes or denied_categories
    )
    # "是不是纸张不对？" is a question, not a denial.  A question that also
    # contains an explicit denial ("是不是没提示纸张错？") still counts.
    negative = explicit_negative or bool(
        _NEGATIVE_GENERIC.search(normalized) and not question
    )
    resolved = bool(
        _RESOLVED_REPLY.find(normalized)
        and not negative
        and not question
        and not _CONTRAST.search(normalized)
    )
    progress = bool(progress_stage and not negative and not resolved)
    new_categories = extract_symptom_categories(text)
    if question and not explicit_negative:
        new_categories = []
    return FeedbackAnalysis(
        resolved=resolved,
        negative=negative,
        is_question=question,
        still_bad=still_bad,
        denied_feature=denied_feature,
        denied_codes=denied_codes,
        denied_categories=denied_categories,
        denied_text=normalized if negative else "",
        new_categories=new_categories,
        explicit_codes=extract_error_codes(text),
        progress=progress,
        progress_stage=progress_stage if progress else None,
        progress_text=normalized if progress else "",
    )


def detect_power_state(text: str) -> str:
    normalized = normalize_text(text)
    if _PatternGroup([r"不亮", r"没反应", r"不通电", r"开不了机"]).find(normalized):
        return "off_or_no_response"
    if _PatternGroup([r"灯亮", r"屏幕亮", r"能开机", r"已开机", r"通电"]).find(normalized):
        return "on"
    return "unknown"


def detect_stage(text: str) -> str:
    normalized = normalize_text(text)
    if re.search(r"开机|启动|电源", normalized):
        return "startup"
    if re.search(r"打印|出纸|进纸|卡纸|纸盒", normalized):
        return "printing"
    if re.search(r"电脑|连接|网络|usb|驱动|手机", normalized):
        return "connection"
    return "unknown"


@dataclass
class ExtractedInfo:
    model: str | None
    model_status: str
    unsupported_hint: str | None
    out_of_scope: bool = False
    error_codes: list[str] = field(default_factory=list)
    # Symptom-to-code alias inferred from a deterministic product mapping.
    # It is never merged into ``error_codes`` and is always marked in tickets.
    inferred_code_hint: str | None = None
    model_assumed: bool = False
    categories: list[str] = field(default_factory=list)
    symptom_labels: list[str] = field(default_factory=list)
    strong_symptom: bool = False
    power_state: str = "unknown"
    stage: str = "unknown"
    requested_human: bool = False
    user_text: str = ""

    def to_ticket_dict(self) -> dict:
        return {
            "model": self.model,
            "model_status": self.model_status,
            "model_assumed": self.model_assumed,
            "unsupported_model_hint": self.unsupported_hint,
            "out_of_scope": self.out_of_scope,
            "error_codes": self.error_codes,
            "inferred_code_hint": self.inferred_code_hint,
            "symptom_categories": self.categories,
            "symptoms": self.symptom_labels,
            "strong_symptom": self.strong_symptom,
            "power_state": self.power_state,
            "stage": self.stage,
        }


def merge_symptom_categories(
    info: ExtractedInfo, categories: Iterable[str]
) -> None:
    """Merge LLM-extracted categories into deterministic extraction results.

    Deterministic keyword categories keep their original order; LLM categories
    are appended only when they add something new.  Unknown category names are
    ignored so a malformed model response cannot poison retrieval.
    """

    merged = list(info.categories)
    for category in categories:
        if category in SYMPTOM_LABELS and category not in merged:
            merged.append(category)
    info.categories = merged
    info.symptom_labels = [SYMPTOM_LABELS.get(c, c) for c in merged]
    info.strong_symptom = strong_symptom(merged)


def extract_info(messages: list[str]) -> ExtractedInfo:
    """Extract slots from the *cumulative* user context (and only user turns)."""

    text = "\n".join(message for message in messages if message.strip())
    model_detection = detect_model(text)
    if model_detection.conflict:
        status = "conflict"
    elif model_detection.model:
        status = "known"
    elif model_detection.unsupported_hint:
        status = "unsupported"
    else:
        status = "unknown"

    categories = extract_symptom_categories(text)
    codes = extract_error_codes(text)
    inferred_code_hint = (
        PRINT_INTERRUPT_HINT_CODE
        if not codes and detect_print_interrupt(text)
        else None
    )
    return ExtractedInfo(
        model=model_detection.model,
        model_status=status,
        unsupported_hint=model_detection.unsupported_hint,
        out_of_scope=model_detection.out_of_scope,
        error_codes=codes,
        inferred_code_hint=inferred_code_hint,
        categories=categories,
        symptom_labels=[SYMPTOM_LABELS.get(c, c) for c in categories],
        strong_symptom=strong_symptom(categories),
        power_state=detect_power_state(text),
        stage=detect_stage(text),
        # Evaluate refusal/request per user turn so "不要转人工" in an earlier
        # turn cannot suppress a later "我要人工".
        requested_human=any(
            requests_human(message) for message in messages if message.strip()
        ),
        user_text=text,
    )
