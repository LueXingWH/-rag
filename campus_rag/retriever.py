"""检索层：BM25 词法通道 + TF-IDF 向量通道 + RRF 融合。

为什么两条通道？它们是"正交"的：
- BM25 擅长精确术语（"GPA 3.5"、"IL-6"、课程代码），但对同义改写脆弱。
- TF-IDF 字符 trigram 向量擅长模糊/改写（"保研要什么条件" vs "推免资格"），
  但对精确数字和长术语不如 BM25。
RRF（Reciprocal Rank Fusion）只用排名不用分值，天然免去两条通道分数量纲不一致的问题。

如果配了 embedding API，还可以升级到语义向量通道（见 embed.py），
那时排序会变成 BM25 + 向量，语义泛化能力显著提升——这是可讲清的"下一步"。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .data import Chunk
from .text import char_ngrams, token_counts, tokenize

# 轻量同义词/缩写扩展：零依赖、可解释、可手改。生产环境应换成向量召回或同义词表服务。
# 注意作用范围：扩展查询只作为**额外的低权重排名**参与 RRF 融合（weight=0.4），
# 它提高召回但不会改变主排序；拒答置信度也**不使用**扩展结果，避免"自己给自己加证据"。
SYNONYMS: Dict[str, List[str]] = {
    "保研": ["推免", "免试攻读", "推荐免试"],
    "推免": ["保研", "免试攻读"],
    "奖学金": ["奖助学金", "国家奖学金", "励志奖学金", "助学金"],
    "考试": ["期末", "考核", "笔试"],
    "复习": ["备考", "考试范围"],
    "实验室": ["机房", "实验中心"],
    "报销": ["经费报销", "财务报销"],
    "挂科": ["不及格", "补考", "重修"],
    "绩点": ["GPA", "gpa", "学分绩"],
    "gpa": ["绩点", "学分绩"],
    "申请": ["报名", "提交材料"],
    "竞赛": ["比赛", "学科竞赛", "大赛"],
    "选课": ["课程选择", "抢课"],
    "导师": ["指导教师", "课题组"],
    "借书": ["借阅", "图书借阅"],
    # 口语问法 -> 文档用词（这类"同义不同词"是词法检索最主要的漏检来源）
    "金额": ["奖励标准", "标准", "资助标准", "奖金标准"],
    "多少钱": ["标准", "元", "费用", "收费标准"],
    "多少度": ["温度"],
    "几本": ["可借册数", "册"],
    "多久": ["期限", "学时", "天"],
    "条件": ["申请条件", "要求", "资格"],
    "流程": ["办理流程", "步骤", "程序"],
}


def expand_query(query: str) -> List[str]:
    """返回同义词扩展后的补充查询串（用于并行检索后融合）。"""
    extra: List[str] = []
    toks = tokenize(query, drop_stopwords=False)
    for t in toks:
        for syn in SYNONYMS.get(t, []):
            extra.append(syn)
    if not extra:
        return []
    return [" ".join(extra)]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    lexical: float = 0.0
    vector: float = 0.0
    semantic: float = 0.0
    matched: Tuple[str, ...] = ()
    snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = self.chunk.to_dict()
        d.update(
            {
                "score": round(self.score, 4),
                "lexical": round(self.lexical, 4),
                "vector": round(self.vector, 4),
                "semantic": round(self.semantic, 4),
                "matched": list(self.matched),
                "snippet": self.snippet,
            }
        )
        return d


class BM25:
    """标准 BM25（k1=1.2, b=0.75），纯 Python 倒排索引。"""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.2, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.tf: List[Dict[str, int]] = []
        self.len: List[int] = []
        self.df: Dict[str, int] = {}
        self.postings: Dict[str, List[int]] = {}
        self.avg_len = 1.0

        for i, ch in enumerate(self.chunks):
            # 标题权重：把 heading 重复计入词频（一次廉价但有效的字段加权）
            counts = token_counts(ch.text)
            for tok, c in token_counts(ch.heading).items():
                counts[tok] = counts.get(tok, 0) + 2 * c
            for tok, c in token_counts(ch.title).items():
                counts[tok] = counts.get(tok, 0) + c
            self.tf.append(counts)
            length = sum(counts.values()) or 1
            self.len.append(length)
            for tok in counts:
                self.df[tok] = self.df.get(tok, 0) + 1
                self.postings.setdefault(tok, []).append(i)

        n = max(1, len(self.chunks))
        self.avg_len = sum(self.len) / n
        self.idf: Dict[str, float] = {
            tok: math.log(1 + (n - df + 0.5) / (df + 0.5)) for tok, df in self.df.items()
        }

    def idf_mass(self, tokens: Sequence[str]) -> Tuple[float, float]:
        """返回 (查询信息量总和, 其中语料里一个词都不存在的部分)。

        "不存在于语料" 的词很关键：用户问了一个语料完全没有的概念（如"退相干"），
        它是一个强拒答信号——信息论上，缺失的正是最该关注的那部分。
        """
        total = 0.0
        oov = 0.0
        for t in set(tokens):
            if t in self.idf:
                total += self.idf[t]
            else:
                oov += 2.0  # 按"高价值但缺失"计，量级与常见词的 IDF 相当
                total += 2.0
        return total, oov

    def weighted_coverage(self, query_tokens: Sequence[str], doc_index: int) -> float:
        """IDF 加权的查询覆盖率 ∈ [0,1]。

        朴素覆盖率（命中token数/总token数）会把"什么/时候/多少"这类高频虚词算成有效证据，
        导致"没料也硬答"。用 IDF 加权后，能提供信息量的实词才计入分母，
        拒答判断因此变得可靠——这是本项目最关键的一处工程修正。
        """
        return self.weighted_coverage_union(query_tokens, [doc_index])

    def weighted_coverage_union(self, query_tokens: Sequence[str], doc_indexes: Sequence[int]) -> float:
        """多块联合的 IDF 加权覆盖率 ∈ [0,1]。

        **分母包含语料里不存在的查询词**（它们按 2.0 的虚拟信息量计入）。
        这一点看似"不公平"，却是必要的：它把"提问用词与文档用词不匹配"这个信号
        留在分数里（如"水温""麻辣香锅"这类语料从未出现的概念），
        由 engine 侧的 oov 惩罚统一处理，而不是在各处偷偷抵销。

        取 Top-N 的**并集**而非只看 Top1：答案常常跨块（"材料"在一节、"计分"在另一节），
        只看 Top1 会把这种"多跳可答"的问题误判为无证据 → 过度拒答。
        """
        q = set(query_tokens)
        if not q:
            return 0.0
        total, _ = self.idf_mass(q)
        if total <= 0:
            return 0.0
        covered: Dict[str, bool] = {}
        for idx in doc_indexes:
            for tok, count in self.tf[idx].items():
                if count and tok in q:
                    covered[tok] = True
        hit = sum(self.idf.get(t, 2.0) for t in covered)
        return min(1.0, hit / total)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: Dict[int, float] = {}
        for tok in q_tokens:
            posting = self.postings.get(tok)
            if not posting:
                continue
            idf = self.idf.get(tok, 0.0)
            for i in posting:
                f = self.tf[i][tok]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg_len)
                scores[i] = scores.get(i, 0.0) + idf * f * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


class TfidfVector:
    """字符 n-gram 的 TF-IDF + 余弦相似度（稀疏向量，标准库实现）。"""

    def __init__(self, chunks: Sequence[Chunk], n: int = 3) -> None:
        self.chunks = list(chunks)
        self.n = n
        self.vectors: List[Dict[str, float]] = []
        self.df: Dict[str, int] = {}
        docs = [char_ngrams(ch.text + " " + ch.heading, n) for ch in self.chunks]
        for grams in docs:
            for g in set(grams):
                self.df[g] = self.df.get(g, 0) + 1
        n_docs = max(1, len(self.chunks))
        for grams in docs:
            tf: Dict[str, int] = {}
            for g in grams:
                tf[g] = tf.get(g, 0) + 1
            vec: Dict[str, float] = {}
            for g, c in tf.items():
                idf = math.log((n_docs + 1) / (self.df.get(g, 0) + 1)) + 1.0
                vec[g] = (1 + math.log(c)) * idf
            self.vectors.append(_l2_normalize(vec))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_tf: Dict[str, int] = {}
        for g in char_ngrams(query, self.n):
            q_tf[g] = q_tf.get(g, 0) + 1
        if not q_tf:
            return []
        n_docs = max(1, len(self.chunks))
        q_vec: Dict[str, float] = {}
        for g, c in q_tf.items():
            idf = math.log((n_docs + 1) / (self.df.get(g, 0) + 1)) + 1.0
            q_vec[g] = (1 + math.log(c)) * idf
        q_vec = _l2_normalize(q_vec)
        scores: List[Tuple[int, float]] = []
        for i, vec in enumerate(self.vectors):
            # 稀疏点积：遍历较短的一侧
            small, big = (q_vec, vec) if len(q_vec) <= len(vec) else (vec, q_vec)
            dot = sum(w * big.get(g, 0.0) for g, w in small.items())
            if dot > 0:
                scores.append((i, dot))
        scores.sort(key=lambda kv: kv[1], reverse=True)
        return scores[:top_k]


def _l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in vec.items()}


def rrf_fuse(
    rankings: Sequence[Sequence[Tuple[int, float]]], k: int = 60, weights: Optional[Sequence[float]] = None
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion：score = Σ w / (k + rank)。只用排名，免去分数量纲问题。"""
    weights = list(weights or [1.0] * len(rankings))
    fused: Dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, (idx, _score) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def _best_snippet(text: str, query: str, width: int = 170) -> str:
    """抽出与查询词重叠最高的窗口，作为引用展示片段。"""
    q = set(tokenize(query))
    if not q:
        return text[:width]
    best, best_hit = 0, -1
    step = max(20, width // 3)
    for start in range(0, max(1, len(text) - width + 1), step):
        window = text[start : start + width]
        hit = len(q & set(tokenize(window)))
        if hit > best_hit:
            best, best_hit = start, hit
    snip = text[best : best + width]
    return ("…" if best > 0 else "") + snip.strip() + ("…" if best + width < len(text) else "")


class HybridRetriever:
    """融合检索器：BM25（含同义词扩召回）+ 字符 TF-IDF 向量，RRF 融合。"""

    def __init__(self, chunks: Sequence[Chunk], embedder: Optional[Any] = None) -> None:
        self.chunks = list(chunks)
        self.bm25 = BM25(self.chunks)
        self.vector = TfidfVector(self.chunks)
        self.embedder = embedder  # 可选：OpenAI 兼容 embedding 通道
        self.embeddings: Optional[List[List[float]]] = None
        if embedder is not None:
            try:
                self.embeddings = embedder.embed_documents([c.text + " " + c.heading for c in self.chunks])
            except Exception:
                self.embeddings = None  # 失败即静默降级为纯词法，保证 demo 不崩

    @property
    def semantic_enabled(self) -> bool:
        return bool(self.embeddings) and self.embedder is not None

    def _semantic_scores(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self.semantic_enabled:
            return []
        try:
            qv = self.embedder.embed_query(query)  # type: ignore[union-attr]
        except Exception:
            return []
        scored = [
            (i, _cosine(qv, dv)) for i, dv in enumerate(self.embeddings or [])
        ]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def search(self, query: str, top_k: int = 5, candidate_k: int = 24) -> List[ScoredChunk]:
        self.last_query_tokens: Tuple[str, ...] = tuple(tokenize(query))
        lex = self.bm25.search(query, top_k=candidate_k)
        vec = self.vector.search(query, top_k=candidate_k)
        rankings = [lex, vec]
        weights = [1.0, 1.0]

        semantic = self._semantic_scores(query, candidate_k)
        if semantic:
            rankings.append(semantic)
            weights.append(1.2)

        # 同义词扩展：作为额外的词法排名参与融合，提升召回不伤害主排序
        for extra_q in expand_query(query):
            extra = self.bm25.search(extra_q, top_k=candidate_k // 2)
            if extra:
                rankings.append(extra)
                weights.append(0.4)

        fused = rrf_fuse(rankings, weights=weights)
        lex_map = dict(lex)
        vec_map = dict(vec)
        sem_map = dict(semantic)
        q_tokens = set(tokenize(query))

        out: List[ScoredChunk] = []
        for idx, score in fused[: top_k * 2]:
            ch = self.chunks[idx]
            hay = set(tokenize(ch.text + " " + ch.heading + " " + ch.title))
            matched = tuple(sorted(q_tokens & hay))
            out.append(
                ScoredChunk(
                    chunk=ch,
                    score=score,
                    lexical=lex_map.get(idx, 0.0),
                    vector=vec_map.get(idx, 0.0),
                    semantic=sem_map.get(idx, 0.0),
                    matched=matched[:12],
                    snippet=_best_snippet(ch.text, query),
                )
            )
        # 查询覆盖率加成：避免"长文档靠长度取胜"，让真正回答问题的块浮上来
        out.sort(key=lambda s: (s.score * (1.0 + 0.25 * _coverage(s.matched, q_tokens))), reverse=True)
        return out[:top_k]


def _coverage(matched: Sequence[str], q_tokens: set) -> float:
    if not q_tokens:
        return 0.0
    return len(set(matched) & q_tokens) / len(q_tokens)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def reference_score(query: str, chunk: Chunk) -> float:
    """离线评测用：查询 token 在（标题+正文）中的加权覆盖率，范围 [0,1]。"""
    q = set(tokenize(query))
    if not q:
        return 0.0
    body = set(tokenize(chunk.text))
    head = set(tokenize(chunk.heading + " " + chunk.title))
    hit = len(q & body) + 1.5 * len(q & head)
    return min(1.0, hit / (len(q) * 1.5))
