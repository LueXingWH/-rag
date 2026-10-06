"""回归测试：把"修过的 bug"钉死，防止它们悄悄回来。

为什么这个文件必须存在：这个项目已经吃过一次亏——README 声称的指标
一整轮都无法复现，而没有任何自动化在跑它（见 docs/known-issues.md）。
下面每条测试都对应一个**真实发生过的缺陷**，注释里写清"错了会怎样"。

零依赖：只用标准库 unittest，和项目其它部分保持同一个约束。

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ask import parse_chat_command  # noqa: E402
from bench import fact_stats  # noqa: E402
from campus_rag.config import Config  # noqa: E402
from campus_rag.engine import Answer  # noqa: E402
from campus_rag.evaluate import run_eval  # noqa: E402
from campus_rag.retriever import SYNONYMS, expand_query  # noqa: E402
from campus_rag.text import tokenize  # noqa: E402


class _FakeEngine:
    """最小可用的引擎替身：只实现评测层真正用到的那两个接口。

    用替身而不是真语料，是为了让这些测试**只测被修的那段逻辑**：
    真语料一变，断言就跟着抖，测试会失去定位能力。
    """

    def __init__(self, answers: dict) -> None:
        self._answers = answers
        self.cfg = Config()

    def answer(self, question: str) -> Answer:
        return self._answers[question]


def _answer(question: str, text: str, mode: str, confidence: float = 0.9) -> Answer:
    return Answer(
        question=question,
        answer=text,
        mode=mode,
        confidence=confidence,
        retrieved=[{"source": "doc.md"}],
    )


class SynonymExpansionTest(unittest.TestCase):
    """bug：同义词表里 3 字以上的词条全部是死代码。

    原因：expand_query 拿 tokenize() 的输出当 key，而分词只产出 CJK bigram/unigram，
    "多少钱""收费标准""申请条件"这类词条永远不可能等于某个 token。
    后果：README 里"同义词扩召回"这条设计决策实际上没生效，口语问法的漏答率没有改善。
    """

    def test_every_synonym_key_is_reachable(self) -> None:
        """表里每一条都必须真的能触发——否则它就是在骗后来读代码的人。"""
        unreachable = [k for k in SYNONYMS if not expand_query(f"{k}是什么")]
        self.assertEqual(unreachable, [], f"这些同义词词条永远不会命中：{unreachable}")

    def test_multi_char_key_fires_on_substring(self) -> None:
        """口语问法（3 字词条）必须能扩出文档用词，这是这张表存在的全部理由。"""
        self.assertIn("多少钱", SYNONYMS)
        extra = expand_query("四六级报名要交多少钱？")
        self.assertTrue(extra, "多字词条没被匹配到，同义词扩展又退化成死代码了")
        self.assertIn("标准", extra[0])
        # 顺带钉住 bug 的机理：这句话里根本不存在 "多少钱" 这个 token。
        self.assertNotIn("多少钱", tokenize("四六级报名要交多少钱？", drop_stopwords=False))

    def test_expansion_is_deterministic_and_deduped(self) -> None:
        q = "国家奖学金的金额是多少？收费标准是什么？"
        self.assertEqual(expand_query(q), expand_query(q))
        self.assertEqual(len(expand_query(q)[0].split()), len(set(expand_query(q)[0].split())))

    def test_empty_query_is_safe(self) -> None:
        for q in ("", "   ", None):
            self.assertEqual(expand_query(q or ""), [])


class CoverageMetricTest(unittest.TestCase):
    """bug：拒答正文里的候选片段把"关键事实覆盖率"刷成了满分。

    拒答模板会原样引用检索到的片段，于是"该答却拒答"的题（n09/n11/n14）
    在拒答文本里照样能找到期望数字/术语，覆盖率记 100%，
    把真实值（65.4%）抬到 README 里那个 81.2%。
    更糟的是 sweep.py 用这个指标挑阈值——等于奖励"拒答还带出处"。
    """

    def _case(self, cid: str) -> dict:
        return {
            "id": cid,
            "question": cid,
            "category": "direct",
            "answerable": True,
            "gold_source": "doc.md",
            "must_include": ["36 元"],
        }

    def test_refusal_scores_zero_even_when_terms_leak_into_preview(self) -> None:
        leaked = "资料中没有找到足够相关的内容。\n  [候选1] …报名费 36 元…"
        engine = _FakeEngine({"c1": _answer("c1", leaked, "refusal", confidence=0.3)})
        report = run_eval(engine, [self._case("c1")], k=3)
        self.assertEqual(report.results[0].coverage, 0.0)
        self.assertEqual(report.metrics["answer_coverage"], 0.0)
        self.assertEqual(report.results[0].missing_terms, ["36 元"])

    def test_real_answer_still_scores_coverage(self) -> None:
        """反向断言：别把"拒答计 0"修成"覆盖率永远为 0"。"""
        engine = _FakeEngine({"c1": _answer("c1", "报名费 36 元。", "extractive")})
        report = run_eval(engine, [self._case("c1")], k=3)
        self.assertEqual(report.results[0].coverage, 1.0)
        self.assertEqual(report.metrics["answer_coverage"], 1.0)


class BenchMetricTest(unittest.TestCase):
    """bug：bench.py 的 strict / end2end 两个口径算出来是同一个数。

    ① fact_stats 把拒答题也算进 strict 的分母（与它自己的文档注释相反），
    ② 实验二打印时又两次都输出 stats['strict']，两列永远相等。
    后果：消融实验看不出"多拒答"带来的虚假提升，选参数会被带偏。
    """

    def _cases(self) -> list:
        base = {"category": "direct", "answerable": True, "gold_source": "doc.md"}
        return [
            dict(base, id="hit", question="hit", must_include=["A"]),
            dict(base, id="refused", question="refused", must_include=["B"]),
        ]

    def test_strict_excludes_refusals_while_end2end_counts_them(self) -> None:
        engine = _FakeEngine(
            {
                "hit": _answer("hit", "答案是 A。", "extractive"),
                # 拒答文本里故意带上关键词：它绝不能被算作"覆盖了 B"
                "refused": _answer("refused", "资料中没有相关内容…[候选1] B…", "refusal", 0.3),
            }
        )
        stats = fact_stats(engine, self._cases())
        self.assertEqual(stats["refused"], 1)
        self.assertEqual(stats["answerable"], 2)
        self.assertEqual(stats["strict"], 1.0)      # 答了的那题答对了
        self.assertEqual(stats["end2end"], 0.5)     # 端到端只解决了一半
        self.assertNotEqual(stats["strict"], stats["end2end"])


class ChatCommandTest(unittest.TestCase):
    """bug：`q.startswith(":s") or True` 恒为真。

    后果：帮助里写的 ":s 查看检索详情"根本不存在（`:s xxx` 会被整串丢去检索），
    而检索详情又永远开着。命令解析必须是可测的纯函数。
    """

    def test_commands_are_recognized(self) -> None:
        self.assertEqual(parse_chat_command(":q"), "quit")
        self.assertEqual(parse_chat_command("quit"), "quit")
        self.assertEqual(parse_chat_command(":s"), "toggle_trace")
        self.assertEqual(parse_chat_command("  :s  "), "toggle_trace")
        self.assertEqual(parse_chat_command(":h"), "help")
        self.assertEqual(parse_chat_command(""), "empty")

    def test_questions_are_not_swallowed_by_commands(self) -> None:
        """`:s` 是命令，但以它开头的**问题**不能被吃掉。"""
        for q in (":s 是什么意思", "宿舍 :s 条件", "怎么退出推免申请"):
            self.assertEqual(parse_chat_command(q), "ask", q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
