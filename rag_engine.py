"""Manual indexing and hybrid, applicability-aware retrieval.

Retrieval design:
* sparse BM25 over Chinese character bigrams and Latin tokens;
* character trigram TF-IDF cosine (a dependency-free fuzzy/dense-style signal);
* reciprocal-rank fusion plus exact error-code / metadata boosts;
* exclusion filters so a rejected diagnosis cannot be retrieved again;
* a deterministic applicability gate before any answer is produced.

The manual is the only source of diagnostic steps.  The retrieval layer never
invents content that is not present in a parsed section.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from schemas import ManualDocument, RetrievedCandidate, SearchResult
from taxonomy import (
    SUPPORTED_MODEL,
    normalize_model,
    normalize_text,
    extract_symptom_categories,
)
from taxonomy import WEAK_SYMPTOM_CATEGORIES


MIN_CONFIDENCE = 0.68
MIN_MARGIN = 0.05
# After negative feedback we must try an alternative chapter rather than end
# the session.  The initial diagnosis keeps the strict gate; the second pass
# only rejects genuinely weak matches.
FEEDBACK_MIN_CONFIDENCE = 0.4
# The second pass is allowed to choose the best remaining chapter even when
# the top two are close: ending the session is less useful than showing the
# next plausible candidate with its confidence disclosed.
FEEDBACK_MIN_MARGIN = 0.0

CATEGORY_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"paper_jam", "paper_feed", "paper_output", "paper_mismatch", "no_print_response"}),
    frozenset({"print_quality", "toner"}),
    frozenset({"connectivity", "wireless", "no_print_response"}),
    frozenset({"no_power"}),
    frozenset({"noise"}),
    frozenset({"cover"}),
    frozenset({"unsupported_function"}),
)

CATEGORY_QUERY_EXPANSIONS = {
    "no_power": "无法开机 不亮 电源",
    "paper_jam": "卡纸 纸张 通道 卡住 卡在 出不来 拉不出来 进不去 堵住",
    "paper_feed": "纸盒 缺纸 进纸",
    "paper_output": "不出纸 纸张 输出",
    "no_print_response": "打印 无响应 没动静",
    "print_quality": "打印 字迹 异常",
    "toner": "墨粉 耗材",
    "noise": "异响 震动",
    "connectivity": "连接 USB 网络",
    "wireless": "手机 无线",
    "cover": "盖板 盖子",
    "paper_mismatch": "纸张 尺寸",
    "unsupported_function": "扫描 传真",
    "indicator": "指示灯 屏幕 闪烁",
}

DOMAIN_TERMS = (
    "卡纸",
    "纸卡",
    "卡住",
    "卡在",
    "出不来",
    "进不去",
    "堵住",
    "堵了",
    "拉不出来",
    "纸盒",
    "没纸",
    "缺纸",
    "不进纸",
    "不出纸",
    "出不来纸",
    "打不出来",
    "不打印",
    "打印不了",
    "打印任务",
    "没动静",
    "没反应",
    "不亮",
    "开不了机",
    "无法开机",
    "启动不了",
    "不通电",
    "模糊",
    "不清晰",
    "不清楚",
    "灰蒙蒙",
    "黑线",
    "白块",
    "白纸",
    "空白页",
    "缺字",
    "掉字",
    "掉粉",
    "一摸就掉",
    "蹭",
    "脏",
    "一道",
    "色块",
    "浅",
    "淡",
    "墨粉",
    "碳粉",
    "墨盒",
    "粉盒",
    "耗材",
    "没墨",
    "异响",
    "咔咔",
    "噪音",
    "嗡嗡",
    "震动",
    "响",
    "连不上",
    "连接不上",
    "搜不到",
    "找不到",
    "离线",
    "网络",
    "网线",
    "usb",
    "驱动",
    "端口",
    "手机",
    "无线",
    "wifi",
    "蓝牙",
    "盖子",
    "盖板",
    "前盖",
    "卡扣",
    "扫描",
    "传真",
    "纸张不对",
    "纸张尺寸",
    "纸张类型",
    "报错",
    "a4",
    "letter",
    "冒烟",
    "焦味",
    "烧焦",
    "漏电",
    "触电",
    "手麻",
    "火花",
    "进水",
)

_STOP_GRAMS = {
    "我的",
    "我们",
    "的",
    "了",
    "是",
    "在",
    "有",
    "和",
    "与",
    "就",
    "都",
    "也",
    "很",
    "还",
    "这",
    "那",
    "一个",
    "一下",
    "怎么",
    "什么",
    "现在",
    "今天",
    "突然",
    "一直",
}


def _cjk_runs(text: str) -> list[str]:
    return re.findall(r"[一-鿿]+", normalize_text(text))


SYNONYM_TOKEN_GROUPS = (
    ("卡纸", ("卡纸", "纸卡", "纸张卡")),
    ("没反应", ("没反应", "没动静", "没有反应", "一点动静都没有")),
    ("不亮", ("不亮", "点不亮")),
    ("没纸", ("没纸", "缺纸")),
    ("连不上", ("连不上", "连接不上")),
    ("墨粉", ("墨粉", "碳粉")),
)


def lexical_tokens(text: str) -> list[str]:
    """BM25 tokens: Chinese bigrams plus Latin/number tokens.

    A tiny synonym canonicalizer handles character-order variants such as
    卡纸/纸卡, which bigram tokenization alone would treat as unrelated.
    """

    normalized = normalize_text(text)
    tokens: list[str] = []
    for run in _cjk_runs(normalized):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    tokens.extend(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", normalized))
    for canonical, variants in SYNONYM_TOKEN_GROUPS:
        hits = sum(normalized.count(variant) for variant in variants)
        tokens.extend([canonical] * min(hits, 3))
    return tokens


def char_ngrams(text: str, n: int = 3) -> list[str]:
    normalized = normalize_text(text)
    grams: list[str] = []
    for run in _cjk_runs(normalized):
        if len(run) < n:
            grams.append(run)
        else:
            grams.extend(run[index : index + n] for index in range(len(run) - n + 1))
    for token in re.findall(r"[a-z0-9]+", normalized):
        grams.append(f"w:{token}")
    return grams


TERM_ALIASES = {
    "没反应": ("没反应", "没动静", "没有反应", "一点动静都没有"),
    "没动静": ("没动静", "没反应", "没有反应", "一点动静都没有"),
    "不出纸": ("不出纸", "出不来纸", "没出纸", "纸张卡住"),
    "卡纸": ("卡纸", "纸卡", "卡住", "卡在", "纸张卡", "拉不出来"),
    "卡住": ("卡住", "卡在", "卡纸", "纸卡", "拉不出来", "取不出来"),
    "卡在": ("卡在", "卡住", "卡纸", "纸卡", "拉不出来"),
    "出不来": ("出不来", "拉不出来", "取不出来", "卡住", "卡在"),
    "进不去": ("进不去", "不进纸", "卡住", "卡在"),
    "堵住": ("堵住", "堵了", "卡住", "卡在", "卡纸"),
    "堵了": ("堵了", "堵住", "卡住", "卡在", "卡纸"),
    "浅": ("浅", "淡", "灰蒙蒙"),
    "模糊": ("模糊", "不清晰", "不清楚"),
    "黑线": ("黑线", "一道", "线"),
    "墨粉": ("墨粉", "碳粉", "没墨"),
    "粉盒": ("粉盒", "墨盒", "墨粉盒"),
    "响": ("响", "异响", "噪音", "咔咔"),
    "扫描": ("扫描",),
    "传真": ("传真",),
    "报错": ("报错", "错误", "提示"),
    "连不上": ("连不上", "连接不上", "无法连接", "找不到"),
    "连接不上": ("连接不上", "连不上", "无法连接", "找不到"),
    "找不到": ("找不到", "连不上", "无法连接"),
}


def _term_coverage(query: str, document_text: str) -> float:
    normalized_query = normalize_text(query)
    normalized_document = normalize_text(document_text)
    terms = [term for term in DOMAIN_TERMS if term in normalized_query]
    if terms:
        hits = 0
        for term in terms:
            aliases = TERM_ALIASES.get(term, (term,))
            if any(alias in normalized_document for alias in aliases):
                hits += 1
        if hits:
            return hits / len(terms)

    query_grams = {
        gram
        for gram in char_ngrams(query, 2)
        if gram not in _STOP_GRAMS and len(gram) > 1 and not gram.startswith("w:")
    }
    if not query_grams:
        return 0.0
    document_grams = set(char_ngrams(document_text, 2))
    return len(query_grams & document_grams) / len(query_grams)


def _category_match(
    query_categories: list[str],
    doc_categories: list[str],
    *,
    include_weak: bool = False,
) -> float:
    ignored = {"vague", "indicator"}
    if include_weak:
        query_set = set(query_categories) - ignored
    else:
        query_set = set(query_categories) - WEAK_SYMPTOM_CATEGORIES
    doc_set = set(doc_categories)
    if not query_set:
        return 1.0
    if not doc_set:
        return 0.0
    if query_set & doc_set:
        return 1.0
    for group in CATEGORY_GROUPS:
        if (query_set & group) and (doc_set & group):
            return 0.5
    return 0.0


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freqs = [Counter(tokens) for tokens in documents]
        self.doc_lengths = [len(tokens) for tokens in documents]
        self.avg_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for tokens in documents:
            document_frequency.update(set(tokens))
        total = len(documents)
        self.idf = {
            token: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in document_frequency.items()
        }

    def scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for index, frequencies in enumerate(self.doc_freqs):
            score = 0.0
            length = self.doc_lengths[index]
            denominator_norm = self.k1 * (
                1 - self.b + self.b * (length / self.avg_length if self.avg_length else 0)
            )
            for token in query_tokens:
                frequency = frequencies.get(token)
                if not frequency:
                    continue
                idf = self.idf.get(token, 0.0)
                score += idf * (
                    frequency * (self.k1 + 1) / (frequency + denominator_norm)
                )
            scores.append(score)
        return scores


class TfidfNgramIndex:
    def __init__(self, documents: list[str], n: int = 3):
        self.n = n
        self.counters = [Counter(char_ngrams(document, n)) for document in documents]
        document_frequency: Counter[str] = Counter()
        for counter in self.counters:
            document_frequency.update(counter.keys())
        total = len(documents)
        self.idf = {
            token: math.log((1 + total) / (1 + frequency)) + 1
            for token, frequency in document_frequency.items()
        }
        self.vectors = [self._vector(counter) for counter in self.counters]

    def _vector(self, counter: Counter[str]) -> dict[str, float]:
        vector = {
            token: count * self.idf.get(token, 1.0) for token, count in counter.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm:
            vector = {token: value / norm for token, value in vector.items()}
        return vector

    def query_vector(self, query: str) -> dict[str, float]:
        return self._vector(Counter(char_ngrams(query, self.n)))

    def scores(self, query_vector: dict[str, float]) -> list[float]:
        scores: list[float] = []
        for vector in self.vectors:
            if len(vector) > len(query_vector):
                score = sum(value * vector.get(token, 0.0) for token, value in query_vector.items())
            else:
                score = sum(value * query_vector.get(token, 0.0) for token, value in vector.items())
            scores.append(max(0.0, score))
        return scores


@dataclass
class _ParsedFields:
    code: str | None = None
    causes: list[str] | None = None
    steps: list[str] | None = None
    warning: str = ""
    manual_risk: str = "未知"

    def __post_init__(self) -> None:
        self.causes = self.causes or []
        self.steps = self.steps or []


def _parse_section_fields(lines: list[str]) -> _ParsedFields:
    fields = _ParsedFields()
    current: str | None = None
    values: dict[str, list[str]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            key, separator, value = line[2:].partition("：")
            if separator:
                current = key.strip()
                values.setdefault(current, [])
                if value.strip():
                    values[current].append(value.strip())
                continue
        if current and re.match(r"^\d+\.\s*", line):
            values[current].append(re.sub(r"^\d+\.\s*", "", line).strip())
        elif current and values.get(current):
            values[current][-1] += line

    code = " ".join(values.get("故障代码", [])).strip()
    fields.code = None if not code or code == "无" else code.upper().replace(" ", "")
    fields.causes = values.get("可能原因", [])
    fields.steps = values.get("排查步骤", [])
    fields.warning = " ".join(values.get("安全警告", [])).strip() or "无"
    fields.manual_risk = " ".join(values.get("风险等级", [])).strip() or "未知"
    return fields


def _infer_document_categories(title: str) -> list[str]:
    categories = extract_symptom_categories(title)
    return [category for category in categories if category != "vague"]


class ManualIndex:
    def __init__(self, documents: list[ManualDocument], source_path: str):
        self.documents = documents
        self.source_path = source_path
        self._build_retrieval_indexes()

    @classmethod
    def from_file(cls, path: str | Path) -> "ManualIndex":
        manual_path = Path(path).resolve()
        text = manual_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        model = SUPPORTED_MODEL
        header_match = re.search(r"LX[\s-]?2000", text[:600], re.IGNORECASE)
        if header_match:
            model = normalize_model(header_match.group(0)) or SUPPORTED_MODEL
        # If the header is missing, keep the single-model default instead of
        # inferring scope from the filename.  Exact scope decisions for user
        # input are made in taxonomy.detect_model(), not here.

        sections: list[tuple[int, str]] = []
        for line_number, line in enumerate(lines, start=1):
            match = re.match(r"^##\s*故障现象[:：]\s*(.+?)\s*$", line)
            if match:
                sections.append((line_number, match.group(1).strip()))

        documents: list[ManualDocument] = []
        for index, (start_line, title) in enumerate(sections):
            end_line = (
                sections[index + 1][0] - 1 if index + 1 < len(sections) else len(lines)
            )
            body_lines = lines[start_line:end_line]
            fields = _parse_section_fields(body_lines)
            source_id = (
                f"LX2000-{fields.code}"
                if fields.code
                else f"LX2000-SEC{index + 1:02d}"
            )
            documents.append(
                ManualDocument(
                    source_id=source_id,
                    model=model,
                    code=fields.code,
                    phenomenon=title,
                    causes=fields.causes,
                    steps=fields.steps,
                    warning=fields.warning,
                    manual_risk=fields.manual_risk,
                    section_index=index + 1,
                    start_line=start_line,
                    end_line=end_line,
                    source_path=manual_path.name,
                    categories=_infer_document_categories(title),
                )
            )

        if not documents:
            raise ValueError(f"未在 {manual_path} 中解析到任何“## 故障现象”章节")
        return cls(documents, manual_path.name)

    def _build_retrieval_indexes(self) -> None:
        self._doc_texts: list[str] = []
        for document in self.documents:
            title_weight = " ".join([document.phenomenon] * 5)
            code_weight = " ".join([document.code or ""] * 3)
            body = " ".join(
                [
                    *document.causes,
                    *document.steps,
                    document.warning,
                    " ".join(document.categories),
                ]
            )
            self._doc_texts.append(
                normalize_text(f"{title_weight} {code_weight} {body}")
            )

        self._bm25 = BM25([lexical_tokens(text) for text in self._doc_texts])
        self._ngram = TfidfNgramIndex(self._doc_texts)

    def search(
        self,
        query: str,
        *,
        model: str | None,
        error_codes: list[str] | None = None,
        categories: list[str] | None = None,
        exclude_codes: list[str] | None = None,
        exclude_source_ids: list[str] | None = None,
        top_k: int = 5,
        enforce_model: bool = True,
        min_confidence: float | None = None,
        min_margin: float | None = None,
    ) -> SearchResult:
        codes = [code.upper() for code in (error_codes or [])]
        confidence_threshold = (
            MIN_CONFIDENCE if min_confidence is None else min_confidence
        )
        margin_threshold = MIN_MARGIN if min_margin is None else min_margin
        categories = categories or []
        result = SearchResult(query=query, model_filter=model)

        indexed = list(enumerate(self.documents))
        if enforce_model and model:
            normalized_model = normalize_model(model)
            indexed = [
                (index, document)
                for index, document in indexed
                if normalize_model(document.model) == normalized_model
            ]
        if not indexed:
            result.failure_code = "no_model_docs"
            result.failure_reason = f"维修手册中没有型号 {model} 的可检索章节"
            return result

        # Negative feedback ("没有出现 E08 的现象") must not retrieve the same
        # chapter again.  Exclusions are applied before scoring so the excluded
        # section can neither win nor influence the top-score normalization.
        excluded_codes = {code.upper() for code in (exclude_codes or [])}
        excluded_sources = set(exclude_source_ids or [])
        if excluded_codes or excluded_sources:
            indexed = [
                (index, document)
                for index, document in indexed
                if not (
                    (document.code and document.code.upper() in excluded_codes)
                    or document.source_id in excluded_sources
                )
            ]
            if not indexed:
                result.failure_code = "all_excluded"
                result.failure_reason = (
                    "已排除上一轮诊断章节，且手册中没有其他可匹配章节"
                )
                return result

        expansion = " ".join(
            CATEGORY_QUERY_EXPANSIONS.get(category, "")
            for category in categories
            if category not in WEAK_SYMPTOM_CATEGORIES
        )
        retrieval_query = f"{query} {query} {query} {' '.join(codes)} {expansion}"
        bm25_scores = self._bm25.scores(lexical_tokens(retrieval_query))
        cosine_scores = self._ngram.scores(self._ngram.query_vector(retrieval_query))
        max_bm25 = max((bm25_scores[index] for index, _ in indexed), default=0.0)

        scored: list[RetrievedCandidate] = []
        for index, document in indexed:
            bm25 = bm25_scores[index]
            cosine = cosine_scores[index]
            bm25_normalized = bm25 / max_bm25 if max_bm25 else 0.0
            code_match = bool(codes and document.code and document.code in codes)
            code_bonus = 1.0 if code_match else 0.0
            coverage = _term_coverage(query, self._doc_texts[index])
            title_coverage = _term_coverage(query, document.phenomenon)
            raw_score = (
                0.35 * bm25_normalized
                + 0.20 * cosine
                + 0.30 * coverage
                + 0.10 * title_coverage
                + 0.05 * code_bonus
            )
            category_match = _category_match(categories, document.categories)
            conflict_match = _category_match(
                categories, document.categories, include_weak=True
            )
            validation: list[str] = []
            failure_code: str | None = None
            failure_reason: str | None = None

            model_match = bool(
                model and normalize_model(document.model) == normalize_model(model)
            )
            if enforce_model and not model_match:
                validation.append("型号不匹配")
                failure_code = "not_applicable"
                failure_reason = "检索结果与用户型号不匹配"
            else:
                validation.append(f"型号匹配：{document.model}")

            if codes:
                if code_match:
                    validation.append(f"错误代码匹配：{document.code}")
                else:
                    failure_code = "code_not_found"
                    failure_reason = f"手册中未找到错误代码 {', '.join(codes)} 的适用章节"
                    validation.append("错误代码不匹配")
            elif category_match:
                validation.append("故障现象类别匹配")
            else:
                failure_code = "not_applicable"
                failure_reason = "故障现象与手册章节不匹配"
                validation.append("故障现象不匹配")

            if codes and categories and conflict_match == 0.0:
                failure_code = "code_symptom_conflict"
                failure_reason = "用户给出的错误代码与描述的现象指向不同故障，需人工确认"
                validation.append("错误代码与现象冲突")

            scored.append(
                RetrievedCandidate(
                    document=document,
                    raw_score=raw_score,
                    bm25=bm25,
                    cosine=cosine,
                    coverage=coverage,
                    category_match=category_match,
                    code_match=code_match,
                    model_match=model_match,
                    applicable=failure_code is None,
                    validation=validation,
                    failure_code=failure_code,
                    failure_reason=failure_reason,
                )
            )

        scored.sort(key=lambda candidate: candidate.raw_score, reverse=True)
        result.candidates = scored[:top_k]

        eligible: list[RetrievedCandidate]
        if codes:
            eligible = [candidate for candidate in scored if candidate.code_match]
            if not eligible:
                result.failure_code = "code_not_found"
                result.failure_reason = f"手册中未找到错误代码 {', '.join(codes)} 的适用章节"
                return result
            eligible.sort(key=lambda candidate: candidate.raw_score, reverse=True)
            winner = eligible[0]
            if winner.failure_code == "code_symptom_conflict":
                winner.confidence = min(0.35, self._confidence(winner, bool(codes)))
                result.failure_code = "code_symptom_conflict"
                result.failure_reason = winner.failure_reason
                result.confidence = winner.confidence
                return result
        else:
            eligible = [
                candidate for candidate in scored if candidate.category_match > 0.0
            ]
            if not eligible:
                result.failure_code = "not_applicable"
                result.failure_reason = "没有通过型号与故障现象适用性校验的手册章节"
                return result
            eligible.sort(key=lambda candidate: candidate.raw_score, reverse=True)
            winner = eligible[0]

        second_score = eligible[1].raw_score if len(eligible) > 1 else 0.0
        top_score = max(winner.raw_score, 1e-9)
        margin = 1.0 if len(eligible) == 1 else max(0.0, (top_score - second_score) / top_score)
        confidence = self._confidence(winner, bool(codes), margin=margin)
        winner.margin = margin
        winner.confidence = confidence
        winner.applicable = True
        winner.validation.append(f"检索置信度：{confidence:.2f}")

        if confidence < confidence_threshold:
            winner.failure_code = "low_confidence"
            winner.failure_reason = (
                f"最高匹配置信度 {confidence:.2f} 低于阈值 {confidence_threshold:.2f}"
            )
            result.failure_code = "low_confidence"
            result.failure_reason = winner.failure_reason
            result.confidence = confidence
            return result
        if margin < margin_threshold:
            winner.failure_code = "ambiguous"
            winner.failure_reason = (
                f"前两名候选过于接近（margin={margin:.2f}），无法确定唯一适用章节"
            )
            result.failure_code = "ambiguous"
            result.failure_reason = winner.failure_reason
            result.confidence = confidence
            return result

        winner.applicable = True
        winner.failure_code = None
        winner.failure_reason = None
        result.winner = winner
        result.confidence = confidence
        return result

    def _confidence(
        self,
        candidate: RetrievedCandidate,
        has_code: bool,
        *,
        margin: float = 1.0,
    ) -> float:
        if has_code:
            confidence = (
                0.56
                + 0.12 * candidate.category_match
                + 0.12 * candidate.coverage
                + 0.10 * candidate.raw_score
                + 0.10 * margin
            )
        else:
            confidence = (
                0.18
                + 0.32 * candidate.category_match
                + 0.24 * candidate.coverage
                + 0.16 * candidate.raw_score
                + 0.10 * margin
            )
        return round(min(1.0, max(0.0, confidence)), 4)
