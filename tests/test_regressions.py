"""回归测试：把"修过的 bug"钉死，防止它们悄悄回来。

为什么这个文件必须存在：这个项目已经吃过一次亏——README 声称的指标
一整轮都无法复现，而没有任何自动化在跑它（见 docs/known-issues.md）。
下面每条测试都对应一个**真实发生过的缺陷**，注释里写清"错了会怎样"。

零依赖：只用标准库 unittest，和项目其它部分保持同一个约束。

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ask  # noqa: E402
from ask import parse_chat_command  # noqa: E402
from bench import fact_stats  # noqa: E402
from campus_rag.config import Config, offline_requested  # noqa: E402
from campus_rag.engine import Answer  # noqa: E402
from campus_rag.evaluate import run_eval  # noqa: E402
from campus_rag.llm import LLMError  # noqa: E402
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


class _StubLLM:
    """替掉 ask.LLMClient：记录"有没有被构造"，并给出假的模型列表。"""

    constructed: list = []

    def __init__(self, *args, **kwargs) -> None:
        _StubLLM.constructed.append(args)

    def list_models(self):
        return ["deepseek-flash", "deepseek-v4-pro"]


class _BoomLLM(_StubLLM):
    def list_models(self):
        raise LLMError("HTTP 401（API key 无效）: invalid api key")


class InfoSelfCheckTest(unittest.TestCase):
    """bug：`--offline` / `CAMPUS_RAG_OFFLINE` 在自检路径上被绕过。

    `--offline` 的帮助写着"强制离线：不调用任何网络接口"，
    但 cmd_info 只看"有没有 key"，于是 `ask.py --info --offline` 照样去打 /models
    （实测会连到 api.deepseek.com）。一个"强制离线"的开关被绕过，
    在断网演示或严格离线环境里会直接打脸。

    同时修掉了另一句误导文案：没配 key 时提示"设置 DEEPSEEK_API_KEY 后自动启用"，
    而实测**恰好相反**——只设 DEEPSEEK_API_KEY 不加 --llm 是不会启用的
    （会自动启用的是 CAMPUS_RAG_API_KEY 这个别名）。
    """

    def setUp(self) -> None:
        _StubLLM.constructed = []
        self._real = ask.LLMClient

    def tearDown(self) -> None:
        ask.LLMClient = self._real

    def _engine(self, api_key: str = "sk-dummy", use_llm: bool = False):
        cfg = Config()
        cfg.llm_api_key = api_key
        cfg.embed_api_key = ""
        cfg.use_llm = use_llm
        cfg.use_embeddings = False
        cfg.corpus_dir = "data/corpus"
        return SimpleNamespace(cfg=cfg, chunks=[SimpleNamespace(source="08-x.md")])

    def _info(self, engine, **ns) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ask.cmd_info(engine, argparse.Namespace(**ns))
        return buf.getvalue()

    def test_offline_flag_skips_network_selfcheck(self) -> None:
        ask.LLMClient = _StubLLM
        out = self._info(self._engine(), offline=True)
        self.assertEqual(_StubLLM.constructed, [], "--offline 下不该构造任何 LLM 客户端")
        self.assertIn("已跳过", out)

    def test_campus_rag_offline_env_also_skips_network(self) -> None:
        """环境变量那一路必须和命令行同源，否则它就是个假开关。"""
        ask.LLMClient = _StubLLM
        with mock.patch.dict(os.environ, {"CAMPUS_RAG_OFFLINE": "1"}):
            self.assertTrue(offline_requested())
            out = self._info(self._engine(), offline=False)
        self.assertEqual(_StubLLM.constructed, [], "CAMPUS_RAG_OFFLINE=1 下仍去联网了")
        self.assertIn("已跳过", out)

    def test_selfcheck_still_runs_when_not_offline(self) -> None:
        """反向断言：别把"离线跳过"修成"自检永远不跑"。"""
        ask.LLMClient = _StubLLM
        out = self._info(self._engine(), offline=False)
        self.assertEqual(len(_StubLLM.constructed), 1)
        self.assertIn("API 自检: 可用", out)

    def test_selfcheck_failure_is_reported_not_raised(self) -> None:
        ask.LLMClient = _BoomLLM
        out = self._info(self._engine(), offline=False)
        self.assertIn("API 自检: 失败", out)

    def test_missing_key_hint_no_longer_lies(self) -> None:
        ask.LLMClient = _StubLLM
        out = self._info(self._engine(api_key=""), offline=False)
        self.assertIn("加 --llm 才启用", out)
        self.assertIn("CAMPUS_RAG_API_KEY", out)
        # 旧文案：设了 DEEPSEEK_API_KEY 就"自动启用"——说反了，不许回来。
        self.assertNotIn("后自动启用", out)

    def test_key_present_but_llm_off_says_so(self) -> None:
        """最容易踩的坑：key 设了却没加 --llm，必须明说原因。"""
        ask.LLMClient = _StubLLM
        out = self._info(self._engine(api_key="sk-dummy", use_llm=False), offline=False)
        self.assertIn("加 --llm 即用", out)


class OfflineConfigTest(unittest.TestCase):
    """CAMPUS_RAG_OFFLINE 必须真的关掉联网能力（不只是关掉提示）。"""

    def test_env_offline_disables_llm_and_embeddings(self) -> None:
        for value in ("1", "true", "TRUE", "yes", "on"):
            with mock.patch.dict(
                os.environ,
                {
                    "CAMPUS_RAG_OFFLINE": value,
                    "DEEPSEEK_API_KEY": "sk-dummy",
                    "SILICONFLOW_API_KEY": "sk-dummy",
                },
            ):
                cfg = Config.from_env()
                self.assertFalse(cfg.use_llm, value)
                self.assertFalse(cfg.use_embeddings, value)
                self.assertTrue(offline_requested(), value)

    def test_not_offline_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(offline_requested())

    def test_offline_beats_explicit_flag(self) -> None:
        """环境变量要求离线时，连 --llm 显式开启也不该联网（更保守的一侧胜出）。"""
        with mock.patch.dict(os.environ, {"CAMPUS_RAG_OFFLINE": "1"}):
            cfg = Config.from_env(use_llm=True, llm_api_key="sk-dummy")
            self.assertFalse(cfg.use_llm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
