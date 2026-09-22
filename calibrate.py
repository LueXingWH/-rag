"""置信度诊断脚本：把"拒答阈值"从拍脑袋变成可标定。

它回答一个具体问题：**同一套置信度公式下，可答题与不可答题的分布能不能被一个阈值分开？**
输出每题的分项信号，并画出可答题/不可答题两个分布的直方图，最后给出推荐阈值与后果。

    python calibrate.py            # 离线（词法+向量）标定
    python calibrate.py --llm      # 同时看开启大模型后的表现（阈值不受影响，闸门在前）
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import Config  # noqa: E402
from campus_rag.data import load_corpus  # noqa: E402
from campus_rag.engine import RagEngine  # noqa: E402
from campus_rag.evaluate import load_eval_set  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="拒答阈值标定")
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--eval-set", default=str(ROOT / "data" / "logs" / "eval_set.json"))
    ap.add_argument("--llm", action="store_true", help="只是用于确认闸门在 LLM 之前生效")
    args = ap.parse_args()

    cfg = Config.from_env(corpus_dir=args.corpus, use_llm=args.llm)
    chunks = load_corpus(cfg.corpus_dir, max_len=cfg.chunk_size, overlap=cfg.chunk_overlap)
    engine = RagEngine(chunks, config=cfg)
    cases = load_eval_set(args.eval_set)

    rows = []
    for c in cases:
        ans = engine.answer(c["question"])
        rows.append(
            {
                "id": c["id"],
                "answerable": bool(c.get("answerable", True)),
                "conf": ans.confidence,
                "signals": ans.trace,
                "q": c["question"],
            }
        )

    print("=" * 108)
    print("逐题置信度分项信号（idf_cov=查询信息量落在证据块里的比例；oov=语料中不存在的查询信息量占比）")
    print("=" * 108)
    print(f"{'id':<5}{'类别':<8}{'conf':>7}{'idf_cov':>9}{'rare':>7}{'oov':>7}{'max_lex':>9}{'spread':>8}{'lex':>7}  问题")
    for r in sorted(rows, key=lambda x: x["conf"]):
        s = r["signals"]
        tag = "可答" if r["answerable"] else "应拒答"
        print(
            f"{r['id']:<5}{tag:<8}{r['conf']:>7.3f}{s.get('idf_coverage', 0):>9.3f}"
            f"{s.get('rare_idf_share', 0):>7.3f}{s.get('oov_share', 0):>7.3f}"
            f"{s.get('max_lexical', 0):>9.2f}{s.get('spread', 0):>8.3f}"
            f"{s.get('lex_saturated', 0):>7.2f}  {r['q'][:32]}"
        )

    ans_confs = [r["conf"] for r in rows if r["answerable"]]
    unans_confs = [r["conf"] for r in rows if not r["answerable"]]
    print("\n" + "=" * 100)
    print("分布对比")
    print("=" * 100)
    if ans_confs:
        print(f"  可答题    n={len(ans_confs):<3} min={min(ans_confs):.3f} "
              f"p25={statistics.quantiles(ans_confs, n=4)[0] if len(ans_confs) > 3 else min(ans_confs):.3f} "
              f"中位={statistics.median(ans_confs):.3f} max={max(ans_confs):.3f}")
    if unans_confs:
        print(f"  应拒答题  n={len(unans_confs):<3} min={min(unans_confs):.3f} "
              f"中位={statistics.median(unans_confs):.3f} max={max(unans_confs):.3f}")

    print("\n阈值扫描（阈值 = 低于该值就拒答）")
    print(f"{'阈值':>6}{'拒答准确率':>12}{'漏答率(过度保守)':>18}{'硬答率(幻觉风险)':>18}")
    best = None
    thresholds = [round(0.18 + 0.01 * i, 2) for i in range(51)]
    for t in thresholds:
        refused = {r["id"] for r in rows if r["conf"] < t}
        correct = sum(1 for r in rows if (r["id"] in refused) == (not r["answerable"]))
        over = sum(1 for r in rows if r["answerable"] and r["id"] in refused)
        false_ans = sum(1 for r in rows if not r["answerable"] and r["id"] not in refused)
        acc = correct / len(rows)
        over_rate = over / max(1, len(ans_confs))
        fa_rate = false_ans / max(1, len(unans_confs))
        # 目标：先消灭"硬答"（幻觉风险），再最小化漏答；最后取阈值区间的中点，避免卡在悬崖边
        key = (fa_rate == 0.0, acc, -over_rate)
        if best is None or key > best[1]:
            best = (t, key)
    # 找出所有达到最优硬答率的阈值，从中取能让 acc - over_rate 最大的那个
    feasible = []
    for t in thresholds:
        refused = {r["id"] for r in rows if r["conf"] < t}
        false_ans = sum(1 for r in rows if not r["answerable"] and r["id"] not in refused)
        over = sum(1 for r in rows if r["answerable"] and r["id"] in refused)
        correct = sum(1 for r in rows if (r["id"] in refused) == (not r["answerable"]))
        if false_ans == 0:
            feasible.append((correct - over, correct, -over, t))
    if feasible:
        feasible.sort(reverse=True)
        chosen = feasible[0][3]
        # 在同等最优表现里取区间中点，让阈值不贴着边界
        same = [f[3] for f in feasible if f[:3] == feasible[0][:3]]
        chosen = round(sum(same) / len(same), 2)
    else:
        chosen = best[0]
    t = chosen
    for t_show in thresholds:
        if int(t_show * 100) % 3 == 0:
            refused = {r["id"] for r in rows if r["conf"] < t_show}
            correct = sum(1 for r in rows if (r["id"] in refused) == (not r["answerable"]))
            over = sum(1 for r in rows if r["answerable"] and r["id"] in refused)
            false_ans = sum(1 for r in rows if not r["answerable"] and r["id"] not in refused)
            print(
                f"{t_show:>6.2f}{correct / len(rows):>12.1%}"
                f"{over / max(1, len(ans_confs)):>18.1%}{false_ans / max(1, len(unans_confs)):>18.1%}"
            )
    refused = {r["id"] for r in rows if r["conf"] < t}
    worst = max(unans_confs) if unans_confs else 0.0
    print(f"\n推荐阈值：{t:.2f}")
    print(f"  → 拒答准确率 {(sum(1 for r in rows if (r['id'] in refused) == (not r['answerable'])) / len(rows)):.1%}"
          f"  漏答 {sum(1 for r in rows if r['answerable'] and r['id'] in refused)}/{len(ans_confs)}"
          f"  硬答 {sum(1 for r in rows if not r['answerable'] and r['id'] not in refused)}/{len(unans_confs)}")
    print(f"  两个分布的分界线：应拒答最大置信度 {worst:.3f} vs 可答最小置信度 "
          f"{min(ans_confs) if ans_confs else 0:.3f} → "
          f"{'分离良好 [OK]' if (min(ans_confs) if ans_confs else 0) > worst else '存在重叠（漏答不可避免，属安全侧失败）'}")
    print(f"  使用方式：python ask.py --threshold {t:.2f} --web")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
