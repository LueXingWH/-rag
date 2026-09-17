"""演示脚本自检：把面试时要演示的问题跑一遍，确认每条都拿到预期结果。

为什么需要它：我改了检索/抽取逻辑后，某些问题会悄悄退化
（实测"挑战杯奖金"从正确摘录退化成引到"竞赛分级"那一节）。
靠眼睛看一眼是发现不了的，必须自动化。

    python demo_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from campus_rag.config import Config  # noqa: E402
from campus_rag.data import load_corpus  # noqa: E402
from campus_rag.engine import RagEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent

# (问题, 期望回答里必须出现的关键词, 期望来源文档关键词)
#
# 关于"期望关键词"的口径：抽取式降级是**原文摘录**，不会改写措辞，
# 所以期望值要贴近文档原文，而不是贴近"人话"。
# 例：问"还能不能申请推免"，文档的答案是"答：不能。"（4 个字，太短会被丢弃），
# 真正会被摘出来的是"……因此不符合申请条件"。因此这里检查语义等价的说法，
# 而不是强行要求出现"不能"两个字——那是 LLM 生成模式才能稳定做到的。
DEMO_CASES = [
    ("挂科之后重修通过了，还能不能申请推免？", ["不符合申请条件"], "推免"),
    ("推免的综合成绩由哪几部分构成？各占多少比例？", ["三部分构成"], "推免"),
    ("国家奖学金的奖励标准是多少？", ["8000"], "奖学金"),
    ("实验室 GPU 工作站单任务最长能运行多久？", ["12"], "实验室"),
    ("本科生可借册数和借期分别是多少？", ["15", "30"], "图书馆"),
    ("参加挑战杯国家级一等奖能拿多少奖金？", ["20000"], "竞赛"),
    ("期末不及格补考考了 85 分，成绩按多少分记载？", ["60"], "选课"),
    ("宿舍单个宿舍总功率上限是多少？", ["2000W"], "生活服务"),
    # 下面是"必须拒答"的用例 —— 现场演示的重头戏
    ("学校食堂的麻辣香锅一份多少钱？", [], None),
    ("2026 年秋季学期什么时候开学？", [], None),
]


def main() -> int:
    cfg = Config.from_env(use_llm=False)
    chunks = load_corpus(cfg.corpus_dir, max_len=cfg.chunk_size, overlap=cfg.chunk_overlap)
    engine = RagEngine(chunks, config=cfg)

    print(f"演示脚本自检（阈值 {cfg.answer_threshold} / evidence_k {cfg.evidence_k}）")
    print("=" * 88)
    failures = []
    for question, must, source in DEMO_CASES:
        ans = engine.answer(question)
        refusals = [s for s in ans.sources]
        top_source = refusals[0]["source"] if refusals else ""
        answer_norm = ans.answer.replace(" ", "")

        if must:  # 期望作答
            hit = all(m.replace(" ", "") in answer_norm for m in must)
            src_ok = source in top_source if source else True
            ok = hit and src_ok and ans.mode != "refusal"
            detail = f"mode={ans.mode:<10} conf={ans.confidence:.2f} 缺={[m for m in must if m.replace(' ','') not in answer_norm]}"
        else:  # 期望拒答
            ok = ans.mode == "refusal"
            detail = f"mode={ans.mode:<10} conf={ans.confidence:.2f}"

        print(f"{'[OK]' if ok else '[FAIL]'} {question}")
        print(f"       {detail}  来源={top_source[:26]}")
        if not ok:
            failures.append(question)

    print("=" * 88)
    if failures:
        print(f"有 {len(failures)} 条演示用例未通过，请换问法或修检索：")
        for q in failures:
            print(f"  - {q}")
        return 1
    print(f"全部 {len(DEMO_CASES)} 条演示用例通过，可以放心上台演示。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
