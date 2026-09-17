"""中文/英文混合的极简分词。

为什么不用 jieba？
- 面试现场环境不可控（可能没网、pip 装不上）。标准库方案 = 零风险。
- 无词表的中文检索，工业界常用做法就是 CJK 二元组（bigram）切分：
  "保研" -> 保研；"保研政策" -> 保研/研政/政策。
  它比单字精度高、比词典分词召回稳，且对未登录词（新术语）天然友好。

英文/数字走小写化 + 词干还原（简单后缀剥离），与中文的 bigram 落在同一个词表空间里，
因此 BM25 可以直接在混合文本上工作。
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List

# CJK 统一表意文字 + 扩展 A + 常用标点外的日韩汉字范围
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_CJK_RE = re.compile(f"[{_CJK}]")
_CJK_RUN_RE = re.compile(f"[{_CJK}]+")
_LATIN_RE = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")
_TOKEN_RE = re.compile(f"[{_CJK}]|[a-z0-9]+(?:[.\\-_][a-z0-9]+)*")

# 停用词：中英混合。中文停用词按"字"处理，因为 bigram 会把它们带进噪声词。
_STOP = {
    "的", "了", "是", "在", "和", "与", "或", "及", "也", "都", "而", "被", "把", "对",
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那", "这个", "那个",
    "什么", "怎么", "怎样", "如何", "为什么", "哪些", "哪个", "多少", "吗", "呢", "啊",
    "有", "没有", "可以", "需要", "应该", "是否", "一个", "一些", "以及", "并且", "但是",
    "the", "a", "an", "of", "to", "is", "are", "was", "were", "be", "and", "or", "in",
    "on", "for", "with", "how", "what", "which", "why", "do", "does", "did", "can",
    "i", "you", "it", "this", "that", "please", "tell", "me",
}

_SUFFIXES = ("ing", "ers", "er", "ies", "ed", "es", "s")


def is_cjk(ch: str) -> bool:
    return bool(_CJK_RE.match(ch))


def _stem(word: str) -> str:
    """非常轻量的英文词干化：只处理最常见后缀，长度不足时不处理。"""
    if len(word) <= 4 or not word.isalpha():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            base = word[: -len(suf)]
            if suf == "ies":
                base += "y"
            return base
    return word


def bigrams(run: str) -> List[str]:
    """把一段连续汉字切成 bigram；长度为 1 时退化为单字。"""
    if len(run) == 1:
        return [run]
    return [run[i : i + 2] for i in range(len(run) - 1)]


def cjk_tokens(run: str, unigrams: bool = True) -> List[str]:
    """汉字串 -> 检索 token：bigram（主）+ unigram（辅）。

    为什么要加上 unigram？这是实测发现的一个真实缺陷：
    纯 bigram 索引下，**文档里的双字词无法被单字或跨词查询匹配到**。
    例：语料写"奖励标准：8000 元"，用户问"金额是多少"——
    两者没有任何一个共同的 bigram（"金额" 在文档里不存在），直接漏检；
    更糟的是 bigram 会跨越词边界产生"金的""是的"这类噪声词，
    导致"查询里最重要的词"被误判成噪声词。

    unigram 是 Lucene `CJKBigramFilter` 的 `outputUnigrams` 选项（官方做法）：
    补上单字通道能显著提高召回，代价是索引变大。
    由于 unigram 的 IDF 天然很低，它不会污染 BM25 的排序，只是兜底召回。
    """
    if not run:
        return []
    if len(run) == 1:
        return [run]
    grams = [run[i : i + 2] for i in range(len(run) - 1)]
    if unigrams:
        grams.extend(run)
    return grams


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """把任意混合文本切成检索 token 列表。

    >>> tokenize("保研 GPA 要求是多少？")[:4]
    ['保研', '研g', 'gpa', '要求']  # 实际输出顺序以字符流为准
    """
    if not text:
        return []
    text = text.lower()
    tokens: List[str] = []
    for run in _CJK_RUN_RE.findall(text):
        tokens.extend(cjk_tokens(run))
    for word in _LATIN_RE.findall(text):
        tokens.append(_stem(word))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _STOP and len(t) > 0]
    return tokens


def token_counts(text: str, drop_stopwords: bool = True) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for tok in tokenize(text, drop_stopwords=drop_stopwords):
        counts[tok] = counts.get(tok, 0) + 1
    return counts


def char_ngrams(text: str, n: int = 3) -> List[str]:
    """字符 n-gram，用于 TF-IDF 向量通道（对 bigram 切分是有益的补充）。

    bigram 会丢掉三元以上搭配信息，字符 trigram 能补回"跨词边界"的匹配能力，
    这让两条检索通道具有互补性，融合后收益才明显。
    """
    s = re.sub(r"\s+", "", text.lower())
    if len(s) < n:
        return [s] if s else []
    return [s[i : i + n] for i in range(len(s) - n + 1)]


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
