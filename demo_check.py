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
# 语料范围：data/corpus/南海校区/（08-校规校纪问答、09-通知公告问答）。
DEMO_CASES = [
    ("南海校区宿舍晚上几点关门？周五周六和法定节假日会延后吗？", ["23:00", "23:30"], "08-"),
    ("旷课多少节会被警告处分？", ["12 学时"], "08-"),
    ("哪种考试作弊会被直接开除？", ["开除学籍", "替考"], "08-"),
    ("对处分决定不服，多久之内可以提出申诉？", ["10 日"], "08-"),
    ("南海校区电动自行车在校园内行驶有什么要求？", ["通行证", "15 公里"], "08-"),
    ("南海校区图书馆闭馆前提前多久清场？", ["10 分钟"], "08-"),
    ("南海校区医保门诊报销安排在什么时间？在哪里办理？", ["第四周", "113"], "09-"),
    ("四六级考试南海校区的听力广播是多少兆赫？", ["FM75"], "09-"),
    # 下面是"必须拒答"的用例 —— 现场演示的重头戏
    ("南海校区游泳池的开放时间和收费标准是什么？", [], None),
    ("2026 级新生军训从哪天开始？", [], None),
]


def main() -> int:
    cfg = Config.from_env(use_llm=False)
    # 语料只看南海校区（08、09），不加载"无用"目录中的旧语料
    chunks = load_corpus(ROOT / "data" / "corpus" / "南海校区", max_len=cfg.chunk_size, overlap=cfg.chunk_overlap)
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
