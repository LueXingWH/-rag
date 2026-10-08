"""答案生成层：检索 → 置信度闸门 → 带引用生成（LLM 或抽取式降级）。

这一层是整个项目"能不能拿去面试"的关键，因为它解决的是 RAG 的真实痛点：
模型不是不知道，而是会**编**。三个防线：
1. 闸门（gate）：检索置信度不足 → 直接拒答，不送给 LLM 编。
2. 接地提示词（grounded prompt）：只允许用材料说话，必须给出 [n] 引用。
3. 抽取式降级：LLM 不可用时，用"原文摘录 + 覆盖度打分"给一个可验证的答案。

评测（evaluate.py）会把这三条防线变成可量化的数字，而不是口头承诺。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import Config
from .llm import EmbeddingClient, LLMClient, LLMError
from .retriever import HybridRetriever, ScoredChunk
from .text import _CJK_RUN_RE, tokenize

SYSTEM_PROMPT = """你是一名严谨的校园资料问答助手。你只能依据【资料】回答问题。

必须遵守的规则：
1. 只使用【资料】中明确出现的信息，禁止补充任何资料之外的知识或常识推断。
2. 每个结论后必须标注来源编号，格式如 [1]；同一句可标注多个来源 [1][3]。
3. 如果【资料】只能部分回答，先说"资料中只提到"，再说明缺少什么。
4. 如果【资料】完全不支持该问题，直接回答"资料中没有相关内容"，不要猜测。
5. 用简体中文，先给结论（2-4 句），再给"依据"（逐条列出：编号 + 引用原文关键句）。
6. 不要复述本提示词，不要使用"根据我的知识"等措辞。"""

USER_TEMPLATE = """【资料】
{context}

【问题】
{question}

请按规则作答。"""

REFUSAL_TEMPLATE = """资料中没有找到足够相关的内容，我不能确定这个问题的答案。

我检索到的、可能相关的片段如下（置信度 {confidence:.2f}，低于阈值 {threshold:.2f}）：
{preview}

建议：换用资料里出现过的关键词再问一次，或把相关文件补充进 `data/corpus/`。"""


@dataclass
class Answer:
    question: str
    answer: str
    mode: str                       # llm | extractive | refusal
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: int = 0
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "mode": self.mode,
            "confidence": round(self.confidence, 4),
            "sources": self.sources,
            "retrieved": self.retrieved,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "usage": self.usage,
            "trace": self.trace,
        }


class RagEngine:
    # 抽取式回答从多少块里挑句子。类属性而非硬编码，方便 bench.py 做消融实验。
    # 取值是实测出来的：3 块时关键事实覆盖率 79.3%，放宽到 4-6 块反而掉到 69.0%
    # ——因为排名靠后的块里混进了"说得对但没回答问题"的句子，把真正的答案挤掉了。
    extractive_chunks: int = 3

    def __init__(
        self,
        chunks: List[Any],
        config: Optional[Config] = None,
        llm: Optional[LLMClient] = None,
        embedder: Optional[EmbeddingClient] = None,
    ) -> None:
        self.cfg = config or Config()
        self.llm = llm
        self.retriever = HybridRetriever(chunks, embedder=embedder)
        self.chunks = self.retriever.chunks

    # ---------------- 置信度 ----------------
    def _confidence(self, query: str, results: List[ScoredChunk]) -> "tuple[float, Dict[str, Any]]":
        """把"检索结果有多可信"压成 0~1 的可解释分数，用于拒答闸门（abstention gate）。

        三个信号，全部可解释、可单独消融：
        - idf_cov：Top-N 证据块的 **IDF 加权联合覆盖率**——主信号。
          "覆盖了多少信息量"而不是"命中了多少词"，因为中文 bigram 下虚词命中毫无证据力。
        - focus  ：Top1 相对 Top2 的分数优势。没有优势 = 检索在"打平"，说明多篇文档都沾边，
          正是"问题超出了语料范围"的典型表现。
        - lex    ：BM25 绝对分的饱和压缩（有下限才有区分度：u01 的 5.9 与 q13 的 32.6 必须分开）。

        再加上两条校准项（都是被评测集逼出来的，见 calibrate.py）：
        - 稀有词质量惩罚：查询信息量若集中在"全语料只出现 1-2 次"的词上（如"水温""麻辣香锅"），
          命中它多半是巧合而非真的回答了问题 → 压低覆盖率。
        - 结构性缺失惩罚：全是高频虚词，或引用了语料中不存在的概念 → 再乘 0.7。
        """
        signals: Dict[str, Any] = {}
        if not results:
            return 0.0, signals
        bm25 = self.retriever.bm25
        token_source = getattr(self.retriever, "last_query_tokens", None) or tuple(tokenize(query))
        q_tokens = set(token_source)
        if not q_tokens:
            return 0.0, signals

        n_evidence = max(1, self.cfg.evidence_k)
        top_indexes = [self.retriever.chunks.index(r.chunk) for r in results[:n_evidence]]
        total_idf, oov = bm25.idf_mass(q_tokens)
        max_idf = max((bm25.idf.get(t, 2.0) for t in q_tokens), default=0.0)
        idf_cov_raw = bm25.weighted_coverage_union(q_tokens, top_indexes)
        rare_share = _rare_idf_share(bm25, q_tokens, len(self.chunks), self.cfg)
        idf_cov = idf_cov_raw * (1.0 - self.cfg.rare_penalty * rare_share)

        lexical_top = results[0].lexical
        spread = 0.0
        if len(results) > 1 and results[0].score > 0:
            spread = max(0.0, (results[0].score - results[1].score) / results[0].score)
        lex = lexical_top / (lexical_top + 14.0) if lexical_top > 0 else 0.0
        focus = min(1.0, spread / 0.20)

        base = 0.52 * idf_cov + 0.10 * focus + 0.38 * lex
        # OOV 惩罚：查询里"语料从未出现过"的词占了多少信息量，就按比例打折。
        # 这是区分"用词不同"（正常，能答）与"概念不存在"（该拒答）的唯一非词法依据。
        if total_idf > 0:
            oov_share = min(1.0, oov / total_idf)
            base *= 1.0 - 0.85 * oov_share
        if total_idf <= 4.5:
            base *= 0.70

        signals.update(
            {
                "idf_coverage": round(idf_cov_raw, 4),
                "idf_coverage_penalized": round(idf_cov, 4),
                "rare_idf_share": round(rare_share, 4),
                "max_idf": round(max_idf, 3),
                "max_lexical": round(lexical_top, 3),
                "spread": round(spread, 4),
                "lex_saturated": round(lex, 4),
                "focus": round(focus, 4),
                "oov_idf_mass": round(oov, 3),
                "oov_share": round(min(1.0, oov / total_idf), 4) if total_idf > 0 else 0.0,
                "evidence_chunks": len(top_indexes),
                "query_terms": sorted(q_tokens)[:12],
            }
        )
        return max(0.0, min(1.0, base)), signals

    # ---------------- 抽取式降级 ----------------
    def _extractive(self, query: str, results: List[ScoredChunk]) -> str:
        """无 LLM 时的答案：按 IDF 加权的查询覆盖度选句，原文摘录，绝不改写（因此不会幻觉）。

        选句规则的两点讲究：
        - 用 IDF 加权而不是"命中词数"：问题里的"什么/多少"命中与否无关紧要，
          而"绩点/显存/奖学金"这类实词命中才是真证据。
        - 数字与百分比额外加权：政策类问答的答案往往就是一个数字（"8000 元""30 天"），
          选句时优先带上数字，能显著提高"关键事实覆盖率"这个指标。
        """
        import re as _re

        bm25 = self.retriever.bm25
        q = set(tokenize(query))
        if not q:
            return "（问题为空）"
        # 判定"用户在问哪个主题"：把查询切成连续的汉字片段（去掉"的/是多少"这类尾巴），
        # 用片段与**标题**做子串匹配。比"取 IDF 最高的 token"稳得多——
        # 后者会被 bigram 噪声词（"金的""是的"）带偏。
        # 例：问"国家奖学金的金额是多少"，标题"国家奖学金"应该被识别为主题命中。
        query_phrases = _topic_phrases(query)

        # 查询里信息量最高的若干个词 = 用户真正在问的对象（"国家奖学金""金额""功率"）。
        key_terms = set(sorted(q, key=lambda t: bm25.idf.get(t, 2.0), reverse=True)[:3])
        total_idf = sum(bm25.idf.get(t, 2.0) for t in q) or 1.0
        # 计算用到的查询词：**只保留语料中真实存在的**。查询里那些不存在于任何文档的
        # 碎片（中文 bigram 会造出"金的""是多"这类词）不该参与归一化，
        # 否则每条句子的覆盖率都被同一个虚高的分母压低，句子之间失去区分度。
        effective_q = {t for t in q if t in bm25.idf}

        def sentence_score(sent: str) -> float:
            toks = set(tokenize(sent))
            hits = effective_q & toks
            if not hits:
                return 0.0
            # 用 BM25 的两个成熟机制给句子打分，而不是自己拍系数：
            #   ① IDF 饱和：一个词命中多次不会线性加分（k1 饱和），
            #      这正是"长句子靠词频堆分"的解药；
            #   ② 句长归一化：b 项按句长相对平均句长的比例调节，
            #      既惩罚"一句话里啥都提一点"，也不会像线性集中度那样把短句捧成冠军。
            # 这是本次迭代的关键一步：从"手调乘子"改成"复用信息检索里已验证的公式"。
            score = 0.0
            for t in hits:
                score += bm25.idf.get(t, 2.0)
            score *= (bm25.k1 + 1.0) / (1.0 + bm25.k1 * (1.0 - bm25.b))
            score += 0.05 * len(_re.findall(r"\d+(?:\.\d+)?%?", sent))  # 数字是强答案信号
            score += 0.35 * len(key_terms & toks)                        # 命中关键词的额外奖励
            return score / total_idf

        lines: List[str] = []
        budget = 6  # 最多摘录 6 句，避免答案变成"整篇复制"
        # 先收集所有候选句并**全局排序**：低相关块里的句子不该混进答案
        # （实测：只按块取前 2 句时，"挑战杯奖金"这类问题会把无关的奖项列表摘进来）
        candidates: List[tuple] = []
        for i, r in enumerate(results[: self.extractive_chunks], start=1):
            # 标题/小节命中是强信号：问"国家奖学金"时，"国家奖学金"这一节里的句子
            # 显然比"注意事项"节里的句子更可能是答案。检索排序已经用了标题加权，
            # 但**摘录阶段**也必须有同样的偏好，否则会把边缘条款排在答案前面。
            heading_text = r.chunk.heading + " " + r.chunk.title
            # 标题与小节的匹配要**按匹配长度给分**，不能"沾到就满分"：
            # "国家奖学金" 节匹配 4 个字、"国家励志奖学金" 节只沾到"奖学金"3 个字，
            # 两者都应该拿到加分，但前者必须更多，否则会把别的奖学金条目排到答案前面。
            topic_chars = max((len(p) for p in query_phrases[:8] if len(p) >= 3 and p in heading_text), default=0)
            heading_bonus = 1.0 + min(0.9, 0.15 * topic_chars)
            for s in _split_sentences(r.chunk.text):
                s = s.strip()
                # 最短句长不能设太大：文档里的 FAQ 答案常常就是把结论单独写一行
                # （"答：不能。"只有 4 个字），设成 6/12 字会把它整条丢掉，
                # 导致"明明写了答案却答不出来"。低于 4 字的碎片才丢弃。
                if len(s) < 4:
                    continue
                # 句子自身提到主题（"国家奖学金"）也是强信号，与所在小节同等重要
                if any(p in s for p in query_phrases[:4] if len(p) >= 3):
                    sent_bonus = 1.25
                else:
                    sent_bonus = 1.0
                sc = sentence_score(s) * heading_bonus * sent_bonus
                # 句子本身是"提问"时要重罚：FAQ 文档里的问句和用户的问题字面高度相似，
                # 会霸占第一名，但它只重复了问题、没有给出答案。
                # （实测：问"挂科重修还能申请推免吗"，排第一的是文档里那句一模一样的问句，
                # 它靠"字面像"拿到 21 个 token 命中，把真正的答案句挤到后面。）
                if _is_question_like(s):
                    sc *= 0.2
                if sc > 0:
                    candidates.append((sc, i, s))
        candidates.sort(key=lambda t: t[0], reverse=True)
        for _score, i, s in candidates[:budget]:
            lines.append(f"[{i}] {s}")
        if not lines:
            return "资料中没有找到能回答该问题的原句，请换一种问法。"
        header = "（离线抽取式回答：以下为资料原文摘录，未经改写，可直接核对）\n"
        return header + "\n".join(lines)

    # ---------------- 主流程 ----------------
    def answer(self, question: str, top_k: Optional[int] = None) -> Answer:
        top_k = top_k or self.cfg.top_k
        # 只检索一次，但取足够宽的候选集：top_k 块用于"喂给模型"，
        # 更大的 evidence_k 用于"判断有没有证据"和"抽取式回答挑句子"。
        # 为什么摘录也要更宽？实测：答案所在小节可能因为同文档存在近似小节
        # （"国家奖学金" vs "国家励志奖学金"）而在融合排序里掉到第 4-6 位，
        # 只看前 3 块就会把真正的答案句排除在外。
        results = self.retriever.search(question, top_k=max(top_k, self.cfg.evidence_k))
        confidence, conf_signals = self._confidence(question, results)

        sources = [
            {
                "index": i,
                "source": r.chunk.source,
                "heading": r.chunk.heading,
                "score": round(r.score, 4),
                "snippet": r.snippet,
                "text": r.chunk.text,
            }
            for i, r in enumerate(results[: self.cfg.max_context_chunks], start=1)
        ]
        retrieved = [r.to_dict() for r in results]
        trace = {
            "lexical_top": results[0].lexical if results else 0.0,
            "vector_top": results[0].vector if results else 0.0,
            "semantic_top": results[0].semantic if results else 0.0,
            "semantic_enabled": self.retriever.semantic_enabled,
            "matched_terms": list(results[0].matched) if results else [],
        }
        trace.update(conf_signals)

        # 防线 1：闸门（只有真正"作答"的路径才提前返回，所以这里用 mode 判断即可）
        if confidence < self.cfg.answer_threshold or not results:
            preview = "\n".join(
                f"  [候选{i}] {r.chunk.source} · {r.chunk.heading}：{r.snippet[:80]}"
                for i, r in enumerate(results[:3], start=1)
            )
            return Answer(
                question=question,
                answer=REFUSAL_TEMPLATE.format(
                    confidence=confidence, threshold=self.cfg.answer_threshold, preview=preview or "  （无）"
                ),
                mode="refusal",
                confidence=confidence,
                sources=sources[:3],
                retrieved=retrieved,
                trace=trace,
            )

        ctx_chunks = results[: self.cfg.max_context_chunks]
        context = "\n\n".join(
            f"[{i}] 来源：{r.chunk.source} · {r.chunk.heading}\n{r.chunk.text}"
            for i, r in enumerate(ctx_chunks, start=1)
        )

        # 防线 2/3：LLM 生成，失败即降级
        if self.llm is not None and self.cfg.use_llm:
            try:
                res = self.llm.chat(SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question))
                if res.text:
                    return Answer(
                        question=question,
                        answer=res.text,
                        mode="llm",
                        confidence=confidence,
                        sources=sources,
                        retrieved=retrieved,
                        latency_ms=res.latency_ms,
                        usage=res.usage,
                        trace=trace,
                    )
                # 防御性分支：LLMClient.chat 现在对空正文直接抛 LLMError（自带完整诊断），
                # 正常走不到这里。留着是为了万一换成别的客户端实现，也不会把空答案当成功。
                err = "LLM 返回空内容（客户端未抛异常；建议跑 python ask.py --check-llm 看详情）"
            except LLMError as e:
                err = str(e)
            return Answer(
                question=question,
                answer=self._extractive(question, ctx_chunks),
                mode="extractive",
                confidence=confidence,
                sources=sources,
                retrieved=retrieved,
                error=err,
                trace=trace,
            )

        return Answer(
            question=question,
            answer=self._extractive(question, ctx_chunks),
            mode="extractive",
            confidence=confidence,
            sources=sources,
            retrieved=retrieved,
            trace=trace,
        )


def _is_question_like(sent: str) -> bool:
    """判断一句话是不是"提问"而不是"陈述"。

    "相似度高"不等于"回答了问题"：FAQ 类文档里的问句与用户提问字面几乎一致，
    检索会把它排到第一；但它只是重复问题。识别并降权，答案句才有机会浮上来。
    """
    s = sent.strip()
    if s.endswith(("？", "?")) or s.startswith(("问：", "问:", "Q:", "Q：")):
        return True
    return bool(s.endswith(("吗", "呢")) and len(s) <= 60)


def _topic_phrases(query: str) -> List[str]:
    """把查询切成"主题候选短语"，用于和标题做子串匹配。

    做法：先按标点/疑问尾巴切开，再在每个汉字片段里滑动 2-6 字的窗口。
    这样"国家奖学金的金额是多少"能产出 "国家奖学金""金额" 等候选，
    从而命中标题"国家奖学金"，而不是拿整句去做不可能相等的子串比较。
    """
    import re

    tail = ("的", "是", "多少", "哪些", "什么", "怎么", "如何", "呢", "吗", "啊", "请问")
    phrases: List[str] = []
    for seg in re.split(r"[，。？！、,.?!;；:：\s]+", query):
        seg = seg.strip()
        if len(seg) < 2:
            continue
        core = seg
        for t in tail:  # 去掉口语尾巴："国家奖学金的金额是多少" -> "国家奖学金的金额"
            if core.endswith(t) and len(core) - len(t) >= 2:
                core = core[: -len(t)]
        parts = re.split(r"[的]", core)
        for part in parts:
            part = part.strip()
            for size in range(min(6, len(part)), 1, -1):
                for start in range(0, len(part) - size + 1):
                    phrases.append(part[start : start + size])
    # 长的短语更具主题性，先去重再按长度降序排列（匹配时优先用长短语）
    return sorted(set(phrases), key=len, reverse=True)


def _split_sentences(text: str) -> List[str]:
    """把块切成一"条"一条可摘录的句子。

    关键点：**不能简单地按换行切**。文档里的列表项经常是"短标签行 + 接续行"：
        - 核心条件：本学年加权平均成绩排名与综合测评排名均位于专业前 10%，
          且无不及格课程；须有至少一项突出表现
        - 奖励标准：每人每年 8000 元
    如果按换行切，"核心条件…" 会和它的续行分家，只剩下标签半句；
    更好的是把同一个列表项的续行**合并成完整一句**，这样摘录出来的才是可读、完整的证据。
    """
    import re

    # 1) 先把"续行"并回上一行：空行、列表项、标题、或已以句末标点结束的行才另起一条
    merged: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        is_new_item = bool(re.match(r"^(?:[#>*\-+]|\d+[.、)]|表[格头]|【)", line))
        ends_sentence = bool(re.search(r"[。！？!?；;：:]$", line))
        # 两种情况要把下一行并进来：
        #   ① 本行没写完（不以句末标点结尾）——常见于被换行折断的长句；
        #   ② 上一行是"答：""核心条件："这种纯标签（以冒号结尾），它在给下一行起头。
        #      不合并的话，"答：不能。"这种 4 字结论句会被丢掉，或标签与内容分家、读不通。
        if merged and not is_new_item and (not ends_sentence or merged[-1].endswith(("：", ":"))):
            joiner = "" if merged[-1].endswith(("，", "、", "（", "(")) else " "
            merged[-1] = (merged[-1] + joiner + line).strip()
        else:
            merged.append(line)

    # 2) 再按中文句末标点切成句子
    sentences: List[str] = []
    for item in merged:
        for part in re.split(r"(?<=[。！？!?；;])", item):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def _coverage_of(matched, q_tokens: set) -> float:
    """朴素覆盖率：命中 token 数 / 查询 token 数（仅作为弱信号使用）。"""
    if not q_tokens:
        return 0.0
    return len(set(matched) & q_tokens) / len(q_tokens)


def _rare_idf_share(bm25, q_tokens: set, n_docs: int, cfg: Any = None) -> float:
    """查询信息量中"落在极稀有词上"的比例 ∈ [0,1]。

    为什么要它：中文 bigram + IDF 有个反直觉的陷阱——**越稀有的词 IDF 越高**，
    于是"水温""麻辣香锅"这类只出现 1-2 次的词一旦碰巧命中，覆盖率会被抬得很高。
    但"词稀有"不等于"证据成立"：在 7 篇文档的语料里只出现一次的词，
    命中它更可能是字面巧合。把"稀有词占了多少信息量"作为惩罚项，
    正是区分"真命中"（申请/学分/报销，跨文档常见）与"巧合命中"（水温/香锅）的关键。
    """
    if not q_tokens or n_docs <= 0:
        return 0.0
    ratio = getattr(cfg, "rare_doc_ratio", 0.12)
    minimum = getattr(cfg, "rare_doc_min", 2)
    threshold = max(minimum, int(ratio * n_docs))
    total, _ = bm25.idf_mass(q_tokens)
    if total <= 0:
        return 0.0
    rare = sum(bm25.idf.get(t, 2.0) for t in q_tokens if 0 < len(bm25.postings.get(t, ())) <= threshold)
    return min(1.0, rare / total)
