"""消融实验：每项改动到底有没有用？用指标说话。

对每个变体跑同一套评测集，报告：关键事实覆盖率 / 追加事实覆盖率 / 拒答准确率 / 硬答率。
"追加覆盖率"= 回答里额外包含了期望事实的比例（宽松口径，衡量"有没有多说到点子上"）。

    python bench.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import Config  # noqa: E402
from campus_rag.data import load_corpus  # noqa: E402
from campus_rag.engine import RagEngine  # noqa: E402
from campus_rag.evaluate import load_eval_set  # noqa: E402

ROOT = Path(__file__).resolve().parent


def fact_stats(engine: RagEngine, cases) -> dict:
    """统计关键事实命中情况。

    两个口径，缺一不可：
    - strict：**分母只含真正作答了的题**，衡量"答了的题答得准不准"
    - end2end：**分母是全部可答题**（该答却拒答 = 该题的关键事实一条都没覆盖），
      衡量"端到端到底解决了多少问题"。
    只看 strict 会高估系统——拒答越多它反而越好看，所以两个都必须报。

    判定"拒答"用 `ans.mode`，而不是"置信度 < 某个阈值"：阈值是引擎内部的，
    在外面再比一次等于把同一件事实现两遍，两处一旦不同步，口径就悄悄错了。
    """
    answered_terms = answered_hits = loose_hits = 0
    all_terms = all_hits = 0
    answerable = refused = 0
    for c in cases:
        if not c.get("answerable"):
            continue
        answerable += 1
        terms = c.get("must_include") or []
        ans = engine.answer(c["question"])
        if ans.mode == "refusal":
            refused += 1
            all_terms += len(terms)  # 拒答：分母照算，分子不加——这就是"没覆盖"
            continue
        answer = ans.answer.replace(" ", "").lower()
        answered_terms += len(terms)
        all_terms += len(terms)
        for term in terms:
            t = term.replace(" ", "").lower()
            if t in answer:
                answered_hits += 1
                all_hits += 1
            elif any(part in answer for part in {t[:3], t[-3:]} if len(part) >= 2):
                loose_hits += 1  # 部分匹配（如期望 "9月30日" 而答出 "9 月 30 日"）
    return {
        "answerable": answerable,
        "refused": refused,
        "must_total": answered_terms,
        "strict": answered_hits / answered_terms if answered_terms else 0.0,
        "loose": (answered_hits + loose_hits) / answered_terms if answered_terms else 0.0,
        "end2end": all_hits / all_terms if all_terms else 0.0,
    }


def main() -> int:
    chunks = load_corpus(str(ROOT / "data" / "corpus"))
    cases = load_eval_set(ROOT / "data" / "logs" / "eval_set.json")

    print("【实验一】抽取式候选块数（阈值 0.42 / evidence_k=5，与 config 默认一致）")
    print("  严格=只算真正作答的题；端到端=分母含该答却拒答的题（拒答算没覆盖）")
    base = Config.from_env(answer_threshold=0.42, evidence_k=5, use_llm=False)
    print(f"{'候选块数':<14}{'严格覆盖率':>12}{'宽松覆盖率':>12}{'端到端覆盖率':>14}")
    print("-" * 52)
    results = []
    for n in (2, 3, 4, 5, 6):
        RagEngine.extractive_chunks = n
        engine = RagEngine(chunks, config=base)
        stats = fact_stats(engine, cases)
        results.append((n, stats))
        print(
            f"{n:<14}{stats['strict']:>12.1%}{stats['loose']:>12.1%}{stats['end2end']:>14.1%}"
        )
    # 选块数用"端到端"排序：候选块数放宽会同时改变"答不答"和"答得准不准"，
    # 只看严格口径会奖励"多拒答"（拒答题不进分母），把参数带偏。
    best_n = max(results, key=lambda r: (r[1]["end2end"], r[1]["strict"]))[0]
    print(f"-> 最佳候选块数 = {best_n}（按端到端覆盖率）\n")

    RagEngine.extractive_chunks = best_n
    print("【实验二】拒答阈值：端到端关键事实覆盖率（分母=所有可答题，拒答算没覆盖）")
    print(f"{'阈值':<8}{'端到端覆盖率':>14}{'严格覆盖率':>12}{'硬答率':>10}")
    print("-" * 44)
    from campus_rag.evaluate import run_eval

    for th in (0.35, 0.38, 0.42, 0.46, 0.50):
        cfg = Config.from_env(answer_threshold=th, evidence_k=5, use_llm=False)
        engine = RagEngine(chunks, config=cfg)
        stats = fact_stats(engine, cases)
        report = run_eval(engine, cases, k=3)
        print(
            f"{th:<8}{stats['end2end']:>14.1%}{stats['strict']:>12.1%}"
            f"{report.metrics['false_answer_rate']:>10.1%}"
        )

    print("\n【实验三】0.42 阈值下每题关键事实命中明细（用于定位到底哪些题没覆盖）")
    cfg = Config.from_env(answer_threshold=0.42, evidence_k=5, use_llm=False)
    engine = RagEngine(chunks, config=cfg)
    for c in cases:
        if not c.get("answerable"):
            continue
        terms = c.get("must_include") or []
        ans = engine.answer(c["question"])
        answer = ans.answer.replace(" ", "").lower()
        if ans.mode == "refusal":
            # 拒答文本里含候选片段，不能在它上面算"命中了什么"（见 evaluate.run_eval）
            status, missed = "拒答", list(terms)
        else:
            missed = [t for t in terms if t.replace(" ", "").lower() not in answer]
            status = "全中" if not missed else "缺"
        print(f"  {c['id']:<4} {status:<4} conf={ans.confidence:.2f} 缺={missed} {c['question'][:28]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
