"""参数扫描：用评测指标挑参数，而不是凭感觉调。

扫描 (answer_threshold, evidence_k) 网格，输出"硬答率 / 拒答准确率 / 漏答率 / 关键事实覆盖率"，
并按业务优先级排序：**先保证硬答率为 0，再最大化拒答准确率，最后看覆盖率**。

    python sweep.py                 # 默认网格
    python sweep.py --coarse        # 更快的粗网格
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import Config  # noqa: E402
from campus_rag.data import load_corpus  # noqa: E402
from campus_rag.engine import RagEngine  # noqa: E402
from campus_rag.evaluate import load_eval_set, run_eval  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(ROOT / "data" / "corpus"))
    ap.add_argument("--eval-set", default=str(ROOT / "data" / "eval_set.json"))
    ap.add_argument("--coarse", action="store_true")
    args = ap.parse_args()

    chunks = load_corpus(args.corpus)
    cases = load_eval_set(args.eval_set)
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40] if args.coarse else [
        round(0.30 + 0.02 * i, 2) for i in range(11)
    ]
    evidence_ks = [3, 4, 6] if args.coarse else [3, 4, 5, 6, 8]

    rows = []
    print(f"{'阈值':>6}{'evidence_k':>12}{'硬答率':>9}{'拒答准':>9}{'漏答率':>9}{'覆盖率':>9}")
    for ek in evidence_ks:
        for th in thresholds:
            cfg = Config.from_env(corpus_dir=args.corpus, answer_threshold=th, evidence_k=ek, use_llm=False)
            engine = RagEngine(chunks, config=cfg)
            report = run_eval(engine, cases, k=3)
            m = report.metrics
            rows.append((th, ek, m["false_answer_rate"], m["refusal_accuracy"], m["over_refusal_rate"], m["answer_coverage"]))
            print(
                f"{th:>6.2f}{ek:>12}{m['false_answer_rate']:>9.1%}{m['refusal_accuracy']:>9.1%}"
                f"{m['over_refusal_rate']:>9.1%}{m['answer_coverage']:>9.1%}"
            )

    # 排序：硬答率最低 > 拒答准确率最高 > 覆盖率最高
    rows.sort(key=lambda r: (r[2], -r[3], -r[5]))
    print("\n推荐参数（硬答率优先，其次拒答准确率，再看关键事实覆盖率）：")
    for r in rows[:5]:
        print(
            f"  threshold={r[0]:.2f} evidence_k={r[1]}  "
            f"硬答率={r[2]:.1%} 拒答准={r[3]:.1%} 漏答率={r[4]:.1%} 覆盖率={r[5]:.1%}"
        )
    best = rows[0]
    print(f"\n改为默认值可用：修改 campus_rag/config.py 中 answer_threshold={best[0]:.2f}, evidence_k={best[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
