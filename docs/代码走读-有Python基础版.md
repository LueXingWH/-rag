# 代码走读（Python 基础向）

> **写给谁**：能看懂函数、`if/for/return`、列表/字典，但没读过上千行工程代码的人。
> **前提**：不再解释 Python 语法。只讲**数据怎么在模块间流动**、**每个判断为什么存在**。
> **怎么读**：第 1 章 15 分钟拿到地图 → 第 2 章跟着一次问答走完全程（最重要）→ 第 3 章开始按模块查。

整个项目 **15 个 Python 文件、约 2600 行**（含注释和空行）。核心逻辑只有 **7 个文件、约 1100 行**：

```
ask.py ──→ engine.py ──→ retriever.py ──→ text.py
                │              │            data.py
                │              └─ BM25 / TfidfVector
                └─ llm.py（可选）
```

其余文件是 **工具**：`web.py` 界面、`evaluate.py` 评测、`qa_logger.py` 日志、
`config.py` 参数、`sweep/calibrate/bench/demo_check/tests` 五个验证脚本。

---

## 1 · 15 分钟：先建立四个认知

### 1.1 系统对外只有一个动作

```python
answer = engine.answer("南海校区宿舍晚上几点关门？")
```

**所有入口（命令行、网页、评测）最后都收敛到这一个方法**：`engine.py` [L254](campus_rag/engine.py#L254)。
读代码时先认清这一点，剩下的都是它的上下游。

```
上游：把语料变成 engine        （一次性，启动时做）
       load_corpus → 63 个 Chunk → RagEngine(chunks)

下游：answer() 返回一个 Answer 对象
       Answer 里装着：答案文本、置信度、来源、检索轨迹
```

### 1.2 数据只在五个类型之间流动

这是全文最重要的一张表。**看代码时只要盯住当前变量是哪个类型，就不会迷路。**

| 类型 | 定义位置 | 装什么 | 由谁产生 → 由谁消费 |
| --- | --- | --- | --- |
| `Config` | [config.py L31](campus_rag/config.py#L31) | 所有参数（阈值、块大小…） | `Config.from_env()` → 全项目 |
| `Chunk` | [data.py L25](campus_rag/data.py#L25) | 一个**文本块** + 它的来源/标题路径 | `chunk_markdown()` → retriever / engine / ui |
| `ScoredChunk` | [retriever.py L83](campus_rag/retriever.py#L83) | `Chunk` + 分数 + 命中的词 + 预览片段 | `retriever.search()` → `engine._confidence` / `_extractive` |
| `Answer` | [engine.py L48](campus_rag/engine.py#L48) | 答案 + 置信度 + `mode` + 来源列表 | `engine.answer()` → CLI / web / eval / 日志 |
| `LLMResult` | [llm.py L52](campus_rag/llm.py#L52) | 模型返回的文本 + 耗时 + token 用量 | `LLMClient.chat()` → engine |

**关键**：`Chunk` 是**检索的最小单位**，也是**引用的最小单位**。
UI 上看到的 `[2]` 指的不是"08 号文件"，而是"08 号文件的第 14 块"。

### 1.3 只有一个"开关"决定系统行为

```python
if confidence < self.cfg.answer_threshold or not results:
    return Answer(mode="refusal", ...)      # 拒答，不调用 LLM
```

**`Answer.mode` 只有三个取值，它是整个系统的状态标签**：

| mode | 什么时候出现 | 特点 |
| --- | --- | --- |
| `"refusal"` | 置信度 < 0.42 | 不编、不猜，属于**正确行为** |
| `"llm"` | 过了闸门 + 有 key + 调用成功 | 通顺，靠提示词约束不许乱说 |
| `"extractive"` | 过了闸门 + 没 key / 调用失败 | 原文摘录，绝不可能幻觉 |

评测、UI 配色、"该答的答了吗"的判断，全部基于这三个值——**没有第四个**。

### 1.4 全项目只有一条不可违反的规则

> **宁可拒答，绝不硬答。**（资料里没有却给出言之凿凿的回答 = 事故）

这条规则在代码里的三处硬体现：

```python
# ① 排序时"硬答率"排第一顺位（sweep.py）
rows.sort(key=lambda r: (r[2], -r[3], -r[5]))     # ↑硬答率 优先于 准确率、覆盖率

# ② 默认阈值 0.42 是"硬答率第一次归零"的最小值（config.py）
answer_threshold: float = 0.42

# ③ 评测的判断标准
#    硬答率 > 0 视为失败；在这个前提下再比别的指标
```

**后面每个"为什么这么写"的答案，最后都会回到这一条。**

---

## 2 · 跟着一次真实的问答走完全程

### 2.1 起点：`ask.py` 把零件装成 engine

```python
def build_engine(args):                                     # ask.py L58
    cfg = Config.from_env(corpus_dir=args.corpus, ...)      # ① 参数
    chunks = load_corpus(cfg.corpus_dir, ...)               # ② 数据
    llm = LLMClient(...) if cfg.use_llm else None           # ③ 可选客户端
    engine = RagEngine(chunks, config=cfg, llm=llm)          # ④ 组装
    return engine
```

**这就是依赖注入**，注意第 ③ 步：**没有 key 就是 `None`，而不是抛异常**。
`engine` 拿到 `None` 后自己决定走降级路径——把"可选"做成"值为 None"，
比散落各处的 `if 有没有key` 干净得多。

第 ② 步做了什么（`load_corpus` → `chunk_markdown`）：

```
data/corpus/南海校区/08-….md  (291 行 Markdown, 38 块)
data/corpus/南海校区/09-….md  (167 行 Markdown, 25 块)
                    ↓  ① 遇到标题就切块 ② 块>480字按句子切(留80字重叠)
                    ↓  ③ 表格把表头拼进每行（否则"可借册数"这类词根本不在数据行里）
63 个 Chunk，每个带 heading 如 "手册 > 宿舍管理与生活秩序 > 门禁与作息"
```

### 2.2 `RagEngine.__init__`：建两套索引

```python
def __init__(self, chunks, config=None, llm=None, embedder=None):    # engine.py L81
    self.retriever = HybridRetriever(chunks, embedder=embedder)      # ★ 建索引在这里
    self.chunks = self.retriever.chunks
```

`HybridRetriever.__init__`（[retriever.py L292](campus_rag/retriever.py#L292)）里两行是关键：

```python
self.bm25   = BM25(self.chunks)         # 倒排索引：词 → 出现在哪些块
self.vector = TfidfVector(self.chunks)  # 稀疏向量：每个块一个 {字符3gram: 权重}
```

**这两行做完，全部语料就在内存里索引好了**（实测 23 ms）。之后每次提问只做查表 + 算分。

### 2.3 `answer()` 逐行：一次问答的全部动作

```python
def answer(self, question, top_k=None):                          # engine.py L254
    top_k = top_k or self.cfg.top_k                              # 4

    # ① 检索：注意取 max(top_k, evidence_k) = max(4,5) = 5 个
    results = self.retriever.search(question, top_k=max(top_k, self.cfg.evidence_k))

    # ② 算置信度（下面 2.4 展开）
    confidence, conf_signals = self._confidence(question, results)

    # ③ 整理引用来源：只给前 4 个（max_context_chunks）
    sources = [{ "index": i, "source": r.chunk.source, "heading": r.chunk.heading,
                 "score": round(r.score, 4), "snippet": r.snippet, "text": r.chunk.text }
               for i, r in enumerate(results[:self.cfg.max_context_chunks], start=1)]

    # ④ 检索轨迹：把内部分数塞进 trace，给 UI 和调试用
    trace = { "lexical_top": ..., "vector_top": ..., "semantic_enabled": ... }
    trace.update(conf_signals)

    # ⑤ 闸门
    if confidence < self.cfg.answer_threshold or not results:
        return Answer(mode="refusal", ...)

    # ⑥ 拼上下文：把块编号成 [1][2][3][4]，让 LLM 能引用
    context = "\n\n".join(f"[{i}] 来源：{r.chunk.source} · {r.chunk.heading}\n{r.chunk.text}"
                          for i, r in enumerate(ctx_chunks, start=1))

    # ⑦ 有 LLM 就生成，失败即降级（★ 这一个 try/except 就是"永不白屏"）
    if self.llm is not None and self.cfg.use_llm:
        try:
            res = self.llm.chat(SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question))
            if res.text:
                return Answer(mode="llm", ...)
            err = "LLM 返回空内容"
        except LLMError as e:
            err = str(e)
        return Answer(answer=self._extractive(question, ctx_chunks), mode="extractive", error=err, ...)

    # ⑧ 压根没有 LLM
    return Answer(answer=self._extractive(question, ctx_chunks), mode="extractive", ...)
```

**为什么 `top_k`(4) 和 `evidence_k`(5) 要分开？**
`top_k` 是"送给模型几块"（要精），`evidence_k` 是"判断有没有证据看几块"（要宽）。
多跳问题的证据常散在第 4~6 名，用窄视野判断会**误判为无证据而拒答**。
实测：`evidence_k=1` 漏答率 62.5%，`=5` 降到 31.2%。

### 2.4 `_confidence()`：全项目最难的一段

**它要回答的问题只有一个：这堆检索结果，够不够格让我开口？**

```python
def _confidence(self, query, results):                            # engine.py L94
    bm25 = self.retriever.bm25
    q_tokens = set(self.retriever.last_query_tokens or tokenize(query))
    if not q_tokens: return 0.0, {}

    n_evidence = max(1, self.cfg.evidence_k)                      # 5
    top_indexes = [self.retriever.chunks.index(r.chunk) for r in results[:n_evidence]]
    total_idf, oov = bm25.idf_mass(q_tokens)                       # 查询信息量 / 缺失部分

    idf_cov_raw = bm25.weighted_coverage_union(q_tokens, top_indexes)   # 主信号
    rare_share  = _rare_idf_share(bm25, q_tokens, len(self.chunks), self.cfg)
    idf_cov = idf_cov_raw * (1.0 - self.cfg.rare_penalty * rare_share)  # 稀有词打折

    lexical_top = results[0].lexical
    spread = max(0.0, (results[0].score - results[1].score) / results[0].score) if len(results) > 1 else 0.0
    lex   = lexical_top / (lexical_top + 14.0) if lexical_top > 0 else 0.0   # 压到 0~1
    focus = min(1.0, spread / 0.20)

    base = 0.52 * idf_cov + 0.10 * focus + 0.38 * lex              # ★ 三个权重和为 1
    if total_idf > 0:
        base *= 1.0 - 0.85 * min(1.0, oov / total_idf)             # OOV 惩罚
    if total_idf <= 4.5:
        base *= 0.70                                               # 全是虚词 → 再打折
    return max(0.0, min(1.0, base)), signals
```

**五个信号各防一种失败**（这是面试最爱追问的地方）：

| 信号 | 权重 | 防的失败 | 一句话解释 |
| --- | --- | --- | --- |
| `idf_cov` | 0.52 | **虚词当证据** | 朴素覆盖率把"什么/多少"算成证据 → 没料也硬答。按 IDF 加权后只有实词算数 |
| `oov` 惩罚 | 乘 0.85 | **问了个语料没有的概念** | "游泳池"占三成信息量 → 打掉 25.5% 信心 |
| `rare_share` 惩罚 | 乘 0.30 | **冷门词巧合命中** | IDF 特性是越稀有越值钱，于是只出现 1 次的"水温"碰巧命中会虚高 |
| `focus` | 0.10 | **多篇文档都沾边** | 第 1 名和第 2 名打平 = 检索在打平，常意味着问题超出语料 |
| `lex` | 0.38 | **量纲不可比** | BM25 原始分无上界（见过 5~65），用 `x/(x+14)` 压进 0~1 |

**实跑对照（同样的公式，两种结局）**：

| 问题 | idf_cov | rare | oov | max_lex | **conf** | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| 宿舍几点关门？ | 0.810 | 0.584 | 0.190 | 65.88 | **0.561** | 作答 |
| 旷课多少节被警告？ | 0.954 | 0.657 | 0.046 | 47.33 | **0.680** | 作答 |
| 游泳池开放时间和收费？ | 0.468 | 0.467 | **0.352** | 17.08 | **0.296** | 拒答 |

第三行的 `oov=0.352` 就是"游泳池"被抓出来的地方。注意第二行 `rare=0.657` 最高却仍答对
——**惩罚是"打折"不是"一票否决"**。

### 2.5 `_extractive()`：没有 LLM 时怎么答

**核心承诺：绝不改写，只从原文挑句子。** 因为一个字都不改，所以它不可能幻觉。

```python
def _extractive(self, query, results):                            # engine.py L161
    query_phrases = _topic_phrases(query)          # 把问题切成主题短语，如 ["国家奖学金","金额"]
    key_terms = 信息量最高的 3 个查询词

    def sentence_score(sent):
        hits = effective_q & set(tokenize(sent))   # 只算"语料里真实存在"的词
        if not hits: return 0.0
        score = sum(bm25.idf[t] for t in hits)                          # ① IDF 加权
        score *= (bm25.k1 + 1.0) / (1.0 + bm25.k1 * (1.0 - bm25.b))     # ② 复用 BM25 饱和/长度项
        score += 0.05 * len(数字)                                        # ③ 数字是强答案信号
        score += 0.35 * len(key_terms & toks)                            # ④ 命中关键术语加分
        return score / total_idf

    for r in results[: self.extractive_chunks]:                        # 只看前 3 块
        heading_bonus = 1.0 + min(0.9, 0.15 * 标题匹配字数)              # 标题命中越多越强
        for s in _split_sentences(r.chunk.text):
            if len(s) < 4: continue
            sc = sentence_score(s) * heading_bonus * sent_bonus
            if _is_question_like(s): sc *= 0.2                          # ★ 问句降权
            candidates.append((sc, i, s))

    candidates.sort(reverse=True)          # 全局排序，不是每块各取几句
    return header + 前 6 句（带 [块号] 标注）
```

三个设计点，每个都是被真实 bug 逼出来的：

| 设计 | 原因 |
| --- | --- |
| **问句降权 ×0.2** | FAQ 文档里的问句和用户问题**字面几乎一样**，token 命中数碾压真正答案句。实测：问"挂科重修还能申请推免吗"，排第一的是文档里那句一模一样的问句 |
| **只从 3 块里挑** | 实测 1块46.9% → 2块53.1% → **3块62.5%** → 4/5/6块持平。排名 4 以后混进"说得对但没回答这个问题"的句子 |
| **标题按匹配长度给分** | "国家奖学金"节匹配 4 字、"国家励志奖学金"节只沾到"奖学金"3 字，前者必须得分更多，否则别的奖学金条目会排到答案前面 |

**它的代价**（必须诚实说）：不会归纳，所以答案里会混噪声。接上 LLM 后这类问题消失——
**`--llm` 是产品形态，抽取式是保证永不白屏的兜底。**

---

## 3 · 模块走读（按需跳读）

### 3.1 `config.py`（100 行）—— 所有旋钮在一处

```python
@dataclass
class Config:
    chunk_size: int = 480        # 一个块最多多少字
    top_k: int = 4               # 送进生成几块
    evidence_k: int = 5          # 判断该不该拒答看几块
    answer_threshold: float = 0.42   # ★ 拒答及格线
    ...
    @classmethod
    def from_env(cls, **overrides):     # L64
        # 环境变量 → 值；函数参数覆盖环境变量
        if not cfg.llm_api_key: cfg.use_llm = False        # ★ 没 key 就根本不去调接口
        if not cfg.embed_api_key: cfg.use_embeddings = False
```

**为什么要集中？** 因为做实验要"只改一个参数、其它不变"（控制变量）。
如果阈值写死在 `engine.py` 里，`sweep.py` 就没法一次跑 55 种组合。

**注意 `from_env` 最后两行**：这 4 行是"永不报错"的第一层实现——
没配 key 就不发请求，用户永远看不到 401 报错。

### 3.2 `text.py`（133 行）—— 中文分词

不用 `jieba`（要 `pip install`），改用 **相邻两字切分**：

```
"南海校区宿舍晚上几点关门？"
  ↓ bigram（主）                          ↓ unigram（辅）
南海 海校 校区 区宿 宿舍 舍晚 晚上 上几 几点 点关 关门   +   南 海 校 区 宿 舍 …
```

| 关键函数 | 作用 |
| --- | --- |
| `cjk_tokens(run, unigrams=True)` [L62](campus_rag/text.py#L62) | 汉字串 → bigram + 单字 |
| `tokenize(text)` [L86](campus_rag/text.py#L86) | 混合文本 → token 列表（含英文词干化、去停用词） |
| `char_ngrams(text, n=3)` [L112](campus_rag/text.py#L112) | 字符 3-gram，**给向量通道用**（另一套切法） |

**为什么要单字（unigram）？** 语料写"奖励标准：8000 元"，用户问"金额是多少"——
两者**没有任何共同 bigram**，纯 bigram 直接漏检。单字 IDF 天然极低，只兜底、不污染排序。

**为什么两套切法？** `tokenize` 给 BM25（有"词"的概念，IDF 有效）；
`char_ngrams` 给向量通道（无视词边界）。**切法不同才有融合价值**。

### 3.3 `data.py`（202 行）—— 切块

```python
def chunk_markdown(text, source, max_len=480, overlap=80, min_len=40):   # L127
    for raw in text.splitlines():
        m = _HEADING_RE.match(raw.strip())
        if m:                                  # 遇到标题
            flush()                            # 把攒的正文存成一块
            heading_stack = heading_stack[:level-1] + [name]    # 维护标题路径
        else:
            buf.append(raw)
```

三个动作：① 按标题切（标题是天然主题边界）② 块太大按句子切、尾部留 80 字重叠
③ 太短的碎片并回上一块。

**最值钱的一处修正是表格处理** `_coalesce_tables` [L65](campus_rag/data.py#L65)：

```
原始：  | 情形 | 处分 |
        |---|---|
        | 替考、组织作弊… | 开除学籍 |
        ↓ 数据行里根本没有"情形""处分"这两个词（它们只在表头）
回填后：表格表头：情形 | 处分
        表格行：情形：替考、组织作弊…，处分：开除学籍     ← 每行自带语义
```

不做这一步，问"哪种作弊会被开除"就有很大概率漏检。README 记录：
**"这一处改动直接把 Hit@1 推到了 100%"**（当时的语料版本）。

### 3.4 `retriever.py`（393 行）—— 两路检索 + 融合

**先看主流程** `HybridRetriever.search()` [L321](campus_rag/retriever.py#L321)：

```python
def search(self, query, top_k=5, candidate_k=24):
    self.last_query_tokens = tuple(tokenize(query))        # ① 分词（engine 稍后要用）
    lex = self.bm25.search(query, top_k=candidate_k)       # ② 通道A：BM25
    vec = self.vector.search(query, top_k=candidate_k)     # ③ 通道B：TF-IDF 余弦
    rankings, weights = [lex, vec], [1.0, 1.0]

    if semantic: rankings.append(semantic); weights.append(1.2)      # ④ 通道C：embedding（可选）
    for extra_q in expand_query(query):                             # ⑤ 通道D：同义词扩展
        rankings.append(self.bm25.search(extra_q, top_k=candidate_k // 2)); weights.append(0.4)

    fused = rrf_fuse(rankings, weights=weights)            # ⑥ 融合
    out.sort(key=lambda s: s.score * (1.0 + 0.25 * _coverage(s.matched, q_tokens)), reverse=True)
    return out[:top_k]                                     # ⑦ 覆盖率加成后截断
```

**两套打分算法各擅长什么**：

| 通道 | 类 | 擅长 | 脆弱点 |
| --- | --- | --- | --- |
| A 词法 | `BM25` [L108](campus_rag/retriever.py#L108) | 精确术语、数字（"12 学时""FM75""15 公里"） | 同义改写 |
| B 向量 | `TfidfVector` [L208](campus_rag/retriever.py#L208) | 模糊改写（字面重叠统计） | 精确数字 |

BM25 的三个机制（对应三个问题）：

```python
idf = log(1 + (N - df + 0.5) / (df + 0.5))              # 词越常见越不值钱
denom = f + k1 * (1 - b + b * len/avg_len)              # 词频饱和 + 长度归一化
scores[i] += idf * f * (k1 + 1) / denom                 # k1=1.2, b=0.75（Lucene 默认值）
```

**标题加权只用了一行**：`counts[标题里的词] += 2 * c`（把标题词词频人为放大 3 倍）。
不另算一套公式，代码只多 3 行。

**RRF 为什么不用分数？** [L261](campus_rag/retriever.py#L261)

```python
score = Σ w / (k + rank)        # k=60，只用排名
# 排第1 = 1/61 = 0.0164 ； 排第2 = 1/62 = 0.0161
```

BM25 的分数是 65.88、余弦是 0.0889，**量纲完全不同，加权求和毫无意义**。
RRF 只用排名就免去了归一化。代价是：**分数没有绝对意义**（第1名 0.0328 vs 第2名 0.0323），
所以**不能拿它比阈值**——这就是 `engine.py` 必须另算一套置信度的原因。

### 3.5 `engine.py`（457 行）—— 心脏

已在第 2 章逐行讲过。阅读顺序建议：

```
answer()        L254   ← 先读这个，拿到全局
_confidence()   L94    ← 最难，五个信号
_extractive()   L161   ← 最长，但逻辑直白（挑句子）
_is_question_like() L350  ┐
_topic_phrases()   L362  ├ 三个辅助函数，被 _extractive 调用
_split_sentences() L391  ┘
```

### 3.6 `llm.py`（152 行）—— 允许失败的薄层

```python
def _post_json(url, payload, api_key, timeout):     # L22，纯 urllib，零依赖
    ...
    except urllib.error.HTTPError as e:
        hint = {401: "API key 无效", 402: "余额不足", 429: "触发限流"}.get(e.code, "请求失败")
        raise LLMError(f"HTTP {e.code}（{hint}）: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"网络不可用: {e.reason}") from e
```

**这一层的设计要点是"薄"**：151 行，不重试、不并发、不缓存。
因为不确定性已经被隔离在这里了——上层只要一个 `try/except LLMError`，系统就获得"永不崩溃"。
错误分类是为了**让人知道该做什么**（401 去查 key、429 等一会、URLError 无所谓）。

### 3.7 `web.py`（149 行）+ `ui.html`（191 行）

```python
def serve(engine, host="127.0.0.1", port=8000, open_browser=True):     # L111
    for candidate in range(port, port + 20):        # 8000~8019 依次尝试，端口占用自动避让
        try: httpd = ThreadingHTTPServer((host, candidate), handler); port = candidate; break
        except OSError: continue
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()          # 延迟开浏览器
    httpd.serve_forever()
```

三个接口：`GET /`（返回 ui.html）、`GET /api/info`（页面顶部徽章）、`POST /api/ask`（提问）。

```python
def _build_payload(engine, question):              # L25
    with _LOCK:                                    # ★ 串行化：纯 Python 检索是 CPU 密集
        ans = engine.answer(question)
    payload["stats"] = { "chunks": len(engine.chunks),
                         "docs": len({c.source for c in engine.chunks}), ... }   # 集合去重
```

前端唯一需要留意的安全点：`esc()` 转义。资料是外部输入，不转义就是 XSS 漏洞。

### 3.8 `evaluate.py`（246 行）—— 把"好不好"变成数字

```python
def run_eval(engine, cases, k=3, progress=None):   # L90
    for case in cases:
        ans = engine.answer(case["question"])
        rank = 正确文档在第几名            # 用于 Hit@k / MRR
        coverage, missing = _term_coverage(ans.answer, case["must_include"])
```

六个指标各自回答一个问题：

| 指标 | 本次实测 | 回答什么 |
| --- | --- | --- |
| Hit@3 | 100% | 答案所在文档被召回了？→ **检索是 RAG 的上限** |
| Hit@1 | 81.2% | 正确文档排第一的比例 |
| MRR | 0.896 | 平均倒数排名（排第 2 也拿 0.5 分，比 Hit@1 细腻） |
| 拒答准确率 | 77.3% | 该拒/该答判断对的比例 |
| **硬答率** | **0.0%** | ★ 资料里没有却硬答的比例 |
| 关键事实覆盖率 | **62.5%** | 答案含期望关键词的比例（**拒答一律计 0**） |

**最后一行有个坑必须知道**：覆盖率原来报 81.2%，是**假的**——
拒答正文里会引用候选片段，而覆盖率在整段回答上匹配关键词，
于是"该答却拒答"的题反而记了 100%。修法是：

```python
coverage = 0.0 if ans.mode == "refusal" else _term_coverage(...)[0]
```

### 3.9 `qa_logger.py`（65 行）—— 旁路，绝不影响主流程

```python
def log_answer(payload, log_dir=..., extra=None):     # L23
    try:
        with _LOCK:                                    # 多线程写文件必须加锁
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return path
    except Exception:
        return None                                    # ★ 静默失败
```

用 JSONL（每行一个 JSON）的理由：追加写 O(1)、写到一半崩了只丢半行、pandas/jq 直接读。
`except: return None` 体现的是**"主路径"与"旁路"的区分**——问答是主路径，日志是旁路。

---

## 4 · 四个验证脚本（它们才是这个项目最值钱的部分）

| 脚本 | 行数 | 回答什么问题 |
| --- | --- | --- |
| `sweep.py` | 68 | 55 种 (阈值 × evidence_k) 哪个最好？按"硬答率优先"排序 |
| `calibrate.py` | 137 | 两组分布的中间在哪？0.42 是"0 硬答"的最小阈值 |
| `bench.py` | 129 | 这个改动真的有用吗？严格 / 端到端**双口径** |
| `demo_check.py` | 83 | 上台要演示的 10 个问题还都对吗？（退出码非 0 = 失败） |
| `tests/test_regressions.py` | 175 | 9 条回归测试：修过的 bug 不许回来 |

**`sweep.py` 的排序键是全项目价值观的代码化**：

```python
rows.sort(key=lambda r: (r[2], -r[3], -r[5]))
#                     ↑硬答率  ↑拒答准确率(负号=越大越前)  ↑覆盖率
```

**`bench.py` 的双口径设计**：

```python
"""strict ：只统计真正作答了的题（拒答不算）→ "答了的题答得准不准"
   end2end：以所有可答题为分母（拒答算没覆盖）→ "端到端解决了多少问题"
   只看前者会高估系统——拒答更多时它会虚假变好。"""
```

极端例子：系统把所有题都拒答 → `strict` 的分母是 0（或"完美"），而 `end2end` 是 0%。
**只报 strict 的系统，可以通过"多拒答"刷分。**

---

## 5 · 你可以马上做的四个实验

```powershell
# ① 看一次问答的全部内部信号（最有用的一条）
python ask.py "南海校区宿舍晚上几点关门？" --json

# ② 看两个分布能不能被一个阈值分开
python calibrate.py

# ③ 验证"抽取式候选块数=3"这个选择
python bench.py

# ④ 改一个参数，看指标怎么动（改了记得跑 ① ）
python ask.py --threshold 0.30 --eval --quiet
```

**⑤ 最推荐的练习：写 5 行探针脚本**（比盯代码快 10 倍）

```python
# tmp_probe.py
import sys, json
sys.path.insert(0, r"C:\Users\26586\Desktop\大学\my__ai")
from campus_rag.config import Config
from campus_rag.data import load_corpus
from campus_rag.engine import RagEngine
from campus_rag.text import tokenize

print(tokenize("保研 GPA 要求是多少？"))          # 看分词
cfg = Config.from_env(use_llm=False)
engine = RagEngine(load_corpus(cfg.corpus_dir), config=cfg)
ans = engine.answer("南海校区宿舍晚上几点关门？")
print(json.dumps(ans.trace, ensure_ascii=False, indent=2))   # 看置信度分项
```

**改代码后的回归清单（五条，缺一不可）**：

```powershell
python -m unittest discover -s tests   # ⓪ 修过的 bug 不许回来
python demo_check.py                   # ① 10 条演示用例
python ask.py --eval                   # ② 硬答率必须为 0
python bench.py                        # ③ 消融：改动到底有没有用
python sweep.py                        # ④ 改了分词/检索 → 阈值要重标定
```

---

## 6 · 调试索引：症状 → 该看哪一行

| 症状 | 第一站 | 看什么 |
| --- | --- | --- |
| 正确文档没出现在引用来源里 | `retriever.search` [L321](campus_rag/retriever.py#L321) | 分词、切块、是否真进了 `fused` |
| 出现在前几名却被拒答 | `_confidence` [L94](campus_rag/engine.py#L94) | `idf_cov` / `oov_share` / `rare_share` 哪个把分数打低了 |
| 作答了但答非所问 | `_extractive` [L161](campus_rag/engine.py#L161) | 是问句抢了第一（看 `_is_question_like`），还是数字没被优先 |
| 覆盖率异常高/低 | `evaluate.run_eval` [L90](campus_rag/evaluate.py#L90) | 拒答是否被正确计 0 |
| 网页打不开 | `web.serve` [L111](campus_rag/web.py#L111) | 端口避让范围、ui.html 是否存在 |
| 中文显示成 `????` | `ask._use_utf8_console` [L23](ask.py#L23) | 控制台编码 |
| `ModuleNotFoundError: campus_rag` | `ask.py` 头部 | 少了 `sys.path.insert(...)` |

---

## 7 · 术语 → 代码位置速查

| 术语 | 代码 |
| --- | --- |
| 块 Chunk（检索与引用的同一粒度） | [`data.Chunk`](campus_rag/data.py#L25) |
| 标题路径 Heading Path | `Chunk.heading` |
| 词法通道 | [`retriever.BM25`](campus_rag/retriever.py#L108) |
| 向量通道 | [`retriever.TfidfVector`](campus_rag/retriever.py#L208) |
| RRF 融合（只用排名） | [`retriever.rrf_fuse`](campus_rag/retriever.py#L261) |
| 同义词扩展（不参与置信度） | [`retriever.expand_query`](campus_rag/retriever.py#L54) |
| 置信度 | [`engine._confidence`](campus_rag/engine.py#L94) |
| 拒答闸门 | [`engine.answer`](campus_rag/engine.py#L254) 的 `if confidence <` |
| 拒答 / 漏答 / 硬答 | `mode="refusal"` / `over_refusal_rate` / `false_answer_rate` |
| 接地提示词 | [`engine.SYSTEM_PROMPT`](campus_rag/engine.py#L21) |
| 引用编号 | `ui.html` 的 `withCites` |
| 抽取式降级 | [`engine._extractive`](campus_rag/engine.py#L161) |
| 关键事实覆盖率（拒答计 0） | `evaluate._term_coverage` + `run_eval` |

---

> **两份配套文档**：
> [`项目全书-图解.md`](项目全书-图解.md) 用流程图讲同一件事（30 分钟）；
> [`项目全书-逐行讲解.md`](项目全书-逐行讲解.md) 是更啰嗦的逐行版（2.1 万字），需要时查。
>
> **所有数字都可用附录里的命令复现**：`ask.py --info / --eval`、`demo_check.py`、
> `bench.py`、`calibrate.py`、`sweep.py`、`python -m unittest discover -s tests`。
