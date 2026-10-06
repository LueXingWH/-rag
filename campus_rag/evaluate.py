"""评测层：把"效果好不好"变成可复现的数字。

三类指标，各自回答一个具体问题：
- 检索指标（Hit@k / MRR）：检索有没有把答案所在的文档捞出来？——RAG 的上限由它决定。
- 拒答指标（Refusal Accuracy / Grounded Rate）：该拒的拒了吗？该答的答了吗？
  RAG 系统最容易翻车的地方不是"答错"，而是"没料也硬答"。
- 生成指标（Term Coverage）：答案有没有真的包含关键事实（如具体数字）。
  这是对 Faithfulness（忠实度）的廉价代理指标，不依赖 LLM 判官；生产环境应补 RAGAS。

注意：这里刻意不引入 LLM 作为评审（虽然更准），因为评测必须能在断网环境复现。
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .engine import RagEngine


@dataclass
class CaseResult:
    id: str
    question: str
    category: str
    answerable: bool
    gold_source: Optional[str]
    mode: str
    confidence: float
    latency_ms: int
    hit_at_1: bool = False
    hit_at_k: bool = False
    reciprocal_rank: float = 0.0
    refusal_correct: bool = False
    grounded: bool = False
    coverage: float = 0.0
    missing_terms: List[str] = field(default_factory=list)
    retrieved_sources: List[str] = field(default_factory=list)
    answer: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class EvalReport:
    total: int
    k: int
    metrics: Dict[str, float]
    by_category: Dict[str, Dict[str, float]]
    results: List[CaseResult]
    config: Dict[str, Any]
    elapsed_ms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "k": self.k,
            "metrics": self.metrics,
            "by_category": self.by_category,
            "results": [r.to_dict() for r in self.results],
            "config": self.config,
            "elapsed_ms": self.elapsed_ms,
        }


def load_eval_set(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("cases", data if isinstance(data, list) else [])


def _term_coverage(answer: str, must_include: List[str]) -> tuple[float, List[str]]:
    """词面覆盖率：答案里出现了多少条期望关键事实。

    注意它**只适用于"真正作答了"的文本**。拒答文本里带着候选片段（见 run_eval），
    直接拿它来算覆盖率，等于让拒绝回答的题靠"引用了包含答案的片段"拿满分。
    """
    if not must_include:
        return 1.0, []
    normalized = answer.replace(" ", "").lower()
    missing = [t for t in must_include if t.replace(" ", "").lower() not in normalized]
    return (len(must_include) - len(missing)) / len(must_include), missing


def run_eval(
    engine: RagEngine,
    cases: List[Dict[str, Any]],
    k: int = 3,
    progress: Optional[Callable[[int, int, CaseResult], None]] = None,
) -> EvalReport:
    started = time.time()
    results: List[CaseResult] = []

    for i, case in enumerate(cases, start=1):
        gold = case.get("gold_source")
        answerable = bool(case.get("answerable", True))
        t0 = time.time()
        ans = engine.answer(case["question"])
        latency = int((time.time() - t0) * 1000)

        sources = [r["source"] for r in ans.retrieved]
        rank = 0
        if gold:
            for idx, src in enumerate(sources, start=1):
                if src == gold:
                    rank = idx
                    break

        refusal = ans.mode == "refusal"
        must_include = case.get("must_include") or []
        if refusal:
            # 拒答不算覆盖：拒答正文里会原样引用检索到的候选片段
            # （"...我检索到的、可能相关的片段如下：[候选1] ...8 千瓦时..."），
            # 若直接在拒答文本上做词面匹配，反而会给"该答却拒答"的题记满分——
            # 实测 n09/n11/n14 三条该答却拒答的题覆盖率全是 100%，
            # 把关键事实覆盖率从真实值抬到 81.2%。拒答就是没答，计 0。
            coverage, missing = 0.0, list(must_include)
        else:
            coverage, missing = _term_coverage(ans.answer, must_include)

        cr = CaseResult(
            id=case.get("id", f"case{i}"),
            question=case["question"],
            category=case.get("category", "unknown"),
            answerable=answerable,
            gold_source=gold,
            mode=ans.mode,
            confidence=round(ans.confidence, 4),
            latency_ms=latency,
            hit_at_1=(rank == 1),
            hit_at_k=(0 < rank <= k),
            reciprocal_rank=(1.0 / rank) if rank else 0.0,
            refusal_correct=(refusal == (not answerable)),
            grounded=(not refusal),
            coverage=round(coverage, 4),
            missing_terms=missing,
            retrieved_sources=sources[:k],
            answer=ans.answer,
            error=ans.error,
        )
        results.append(cr)
        if progress:
            progress(i, len(cases), cr)

    answerable_results = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    metrics = {
        f"hit@{k}": rate(sum(1 for r in answerable_results if r.hit_at_k), len(answerable_results)),
        "hit@1": rate(sum(1 for r in answerable_results if r.hit_at_1), len(answerable_results)),
        "mrr": round(
            statistics.fmean([r.reciprocal_rank for r in answerable_results]) if answerable_results else 0.0, 4
        ),
        "refusal_accuracy": rate(sum(1 for r in results if r.refusal_correct), len(results)),
        "over_refusal_rate": rate(sum(1 for r in answerable_results if r.mode == "refusal"), len(answerable_results)),
        "false_answer_rate": rate(
            sum(1 for r in unanswerable if r.mode != "refusal"), len(unanswerable)
        ),
        "answer_coverage": round(
            statistics.fmean([r.coverage for r in answerable_results]) if answerable_results else 0.0, 4
        ),
        "unanswerable_refusal_rate": rate(
            sum(1 for r in unanswerable if r.mode == "refusal"), len(unanswerable)
        ),
        "latency_p50_ms": round(
            statistics.median([r.latency_ms for r in results]) if results else 0.0, 1
        ),
        "latency_mean_ms": round(
            statistics.fmean([r.latency_ms for r in results]) if results else 0.0, 1
        ),
    }

    by_category: Dict[str, Dict[str, float]] = {}
    for cat in sorted({r.category for r in results}):
        group = [r for r in results if r.category == cat]
        by_category[cat] = {
            "n": len(group),
            f"hit@{k}": rate(sum(1 for r in group if r.hit_at_k), len(group)),
            "refusal_accuracy": rate(sum(1 for r in group if r.refusal_correct), len(group)),
            "answer_coverage": round(statistics.fmean([r.coverage for r in group]), 4),
        }

    return EvalReport(
        total=len(results),
        k=k,
        metrics=metrics,
        by_category=by_category,
        results=results,
        config=engine.cfg.to_dict(),
        elapsed_ms=int((time.time() - started) * 1000),
    )


def format_report(report: EvalReport, show_cases: bool = True) -> str:
    m = report.metrics
    lines: List[str] = []
    lines.append("=" * 68)
    lines.append(f"  校园资料问答 RAG · 离线评测报告   （{report.total} 题 / 耗时 {report.elapsed_ms} ms）")
    lines.append("=" * 68)
    lines.append("")
    lines.append("【检索质量】—— 决定 RAG 效果的上限")
    lines.append(f"  Hit@{report.k}（答案所在文档被召回的比例） : {m[f'hit@{report.k}']:.1%}")
    lines.append(f"  Hit@1（排在第一位的比例）               : {m['hit@1']:.1%}")
    lines.append(f"  MRR（平均倒数排名，越接近 1 越好）      : {m['mrr']:.3f}")
    lines.append("")
    lines.append("【拒答能力】—— 没料时敢不敢说不知道")
    lines.append(f"  拒答准确率（该拒/不该拒判断对的比例）  : {m['refusal_accuracy']:.1%}")
    lines.append(f"  该答却拒答率（过度保守，越低越好）      : {m['over_refusal_rate']:.1%}")
    lines.append(f"  没料却硬答率（幻觉风险，越低越好）      : {m['false_answer_rate']:.1%}")
    lines.append("")
    lines.append("【答案质量】")
    lines.append(f"  关键事实覆盖率（数字/术语命中）        : {m['answer_coverage']:.1%}")
    lines.append(f"  延迟 p50 / 均值                        : {m['latency_p50_ms']:.0f} ms / {m['latency_mean_ms']:.0f} ms")
    lines.append("")
    lines.append("【分类别表现】")
    for cat, v in report.by_category.items():
        lines.append(
            f"  {cat:<14} n={int(v['n']):<3} Hit@{report.k}={v[f'hit@{report.k}']:.0%}  "
            f"拒答准={v['refusal_accuracy']:.0%}  覆盖率={v['answer_coverage']:.0%}"
        )

    if show_cases:
        lines.append("")
        lines.append("【逐题明细】([OK]=该答的答了/该拒的拒了；[X]=没做对)")
        for r in report.results:
            flag = "[OK]" if (r.refusal_correct and (not r.answerable or r.hit_at_k)) else "[X] "
            mark = "" if r.answerable else "(应拒答)"
            lines.append(
                f"  {flag} {r.id:<4} {mark:<6} mode={r.mode:<10} conf={r.confidence:.2f} "
                f"cov={r.coverage:.0%} {r.question[:34]}"
            )
            if r.missing_terms:
                lines.append(f"        缺关键事实: {', '.join(r.missing_terms)}")
            if r.error:
                lines.append(f"        降级原因: {r.error[:90]}")
    lines.append("")
    return "\n".join(lines)
