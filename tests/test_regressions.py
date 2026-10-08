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
from campus_rag.config import Config, llm_disabled_reason, offline_requested  # noqa: E402
from campus_rag.engine import Answer, RagEngine  # noqa: E402
from campus_rag.evaluate import run_eval  # noqa: E402
from campus_rag.llm import LLMError, LLMResult  # noqa: E402
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


class LlmStatusTest(unittest.TestCase):
    """bug：Web 徽章把"配了 key"当成"大模型已启用"。

    现象（用户真实反馈）：右上角徽章亮着绿灯写"大模型生成 开"，
    下面每条回答却是"离线抽取式降级"——于是"我明明配了 API，怎么还是离线模式"。
    根因：ui.html 读的是 `config.llm_api_key`（有没有 key），而不是 `use_llm`（真开关）。
    修法：后端把结论 llm_enabled / llm_reason 算好给前端，前端不再自己猜。
    """

    def _cfg(self, api_key: str = "", use_llm: bool = False) -> Config:
        cfg = Config()
        cfg.llm_api_key = api_key
        cfg.use_llm = use_llm
        return cfg

    def test_reason_is_empty_when_enabled(self) -> None:
        self.assertEqual(llm_disabled_reason(self._cfg("sk-x", True)), "")

    def test_reason_distinguishes_the_three_causes(self) -> None:
        self.assertEqual(llm_disabled_reason(self._cfg("sk-x", False)), "未加 --llm")
        self.assertEqual(llm_disabled_reason(self._cfg("", False)), "未配置 API key")
        self.assertEqual(llm_disabled_reason(self._cfg("sk-x", False), offline=True), "强制离线")

    def test_info_payload_reports_the_real_switch(self) -> None:
        """关键断言：有 key 但没启用时，llm_enabled 必须是 False（旧逻辑会报 True）。"""
        import campus_rag.web as web

        engine = SimpleNamespace(
            cfg=self._cfg("sk-dummy", False),
            chunks=[SimpleNamespace(source="08-x.md")],
            retriever=SimpleNamespace(semantic_enabled=False),
        )
        payload = web.build_info_payload(engine)
        self.assertFalse(payload["llm_enabled"])
        self.assertEqual(payload["llm_reason"], "未加 --llm")
        # 脱敏后的 "***" 仍然"非空"——正是它能骗到旧徽章。整个字段摘掉，断掉这条路。
        self.assertNotIn("llm_api_key", payload["config"])
        self.assertNotIn("embed_api_key", payload["config"])

        engine.cfg.use_llm = True
        payload = web.build_info_payload(engine)
        self.assertTrue(payload["llm_enabled"])
        self.assertEqual(payload["llm_reason"], "")

    def test_ui_badge_does_not_guess_from_api_key(self) -> None:
        """前端不许再从 `llm_api_key` 推断开关——那正是这个 bug 的形状。"""
        ui = (ROOT / "campus_rag" / "ui.html").read_text(encoding="utf-8")
        self.assertIn("llm_enabled", ui)
        self.assertNotIn("config.llm_api_key", ui)

    def test_startup_hint_says_which_it_is(self) -> None:
        args = argparse.Namespace(llm=True, info=False, offline=False)
        self.assertIn("--llm", ask.llm_status_hint(args, self._cfg("", False), offline=False))
        # 要了 LLM 却拿不到 → 必须出声
        hint = ask.llm_status_hint(args, self._cfg("", False), offline=False)
        self.assertIn("没有启用", hint)
        # 没要、但配了 key → 一句提示就能省一次排查
        hint = ask.llm_status_hint(argparse.Namespace(llm=False), self._cfg("sk-x", False), offline=False)
        self.assertIn("没加 --llm", hint)
        # 启用了 → 告知用的是哪个模型
        hint = ask.llm_status_hint(argparse.Namespace(llm=True), self._cfg("sk-x", True), offline=False)
        self.assertIn("deepseek-flash", hint)
        # 什么都没配的纯离线演示 → 不啰嗦
        self.assertEqual(
            ask.llm_status_hint(argparse.Namespace(llm=False), self._cfg("", False), offline=False), ""
        )


class LlmEmptyContentTest(unittest.TestCase):
    """bug：`LLM 返回空内容` 六个字把所有人挡住了（用户实际报的故障）。

    DeepSeek V4 的 thinking **默认开启**且 effort=high，思维链走 reasoning_content、
    答案走 content；本项目 max_tokens=700 是留给"结论+依据"的，思维链一开就吃光，
    于是 content 为空 → 前端只显示一句"服务端错误：LLM 返回空内容"。
    更糟的是前端当时把整条**已经降级好的抽取式答案**丢掉了（见 UiDegradeTest）。

    这里把上游响应造出来，断言：默认不带 thinking、空正文会抛出**可照着修**的错误。
    """

    def _client(self, **kw):
        from campus_rag.llm import LLMClient

        kw.setdefault("thinking", "disabled")
        return LLMClient("https://api.deepseek.com", "sk-dummy", "deepseek-flash", **kw)

    def test_thinking_is_disabled_by_default(self) -> None:
        payload = self._client().build_payload("s", "u")
        self.assertEqual(payload["thinking"], {"type": "disabled"})

    def test_thinking_can_be_enabled_or_omitted(self) -> None:
        self.assertEqual(self._client(thinking="enabled").build_payload("s", "u")["thinking"],
                         {"type": "enabled"})
        # omit：不认厂商扩展字段的第三方网关走这条路
        self.assertNotIn("thinking", self._client(thinking="omit").build_payload("s", "u"))

    def test_empty_content_raises_actionable_error(self) -> None:
        """思维链吃光 max_tokens 的典型响应：content 为空、finish_reason=length。"""
        from campus_rag.llm import LLMError

        response = {
            "model": "deepseek-flash",
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "reasoning_content": "让我先想想这个问题的……" * 40},
            }],
            "usage": {"prompt_tokens": 1800, "completion_tokens": 700},
        }
        with mock.patch("campus_rag.llm._post_json", return_value=response):
            with self.assertRaises(LLMError) as ctx:
                self._client().chat("s", "u")
        msg = str(ctx.exception)
        self.assertIn("LLM 返回空内容", msg)
        self.assertIn("finish_reason=length", msg)   # 原因
        self.assertIn("思维链", msg)                  # 证据
        self.assertIn("700", msg)                     # 用量
        self.assertIn("max_tokens", msg)              # 能照着修的出路

    def test_plain_truncation_gets_different_advice(self) -> None:
        """没有思维链、单纯被截断：建议不该再提 thinking，否则就是误导。"""
        from campus_rag.llm import LLMError

        response = {"choices": [{"finish_reason": "length", "message": {"content": "  "}}],
                    "usage": {"completion_tokens": 42}}
        with mock.patch("campus_rag.llm._post_json", return_value=response):
            with self.assertRaises(LLMError) as ctx:
                self._client().chat("s", "u")
        self.assertIn("max_tokens", str(ctx.exception))
        self.assertNotIn("建议：关掉 thinking", str(ctx.exception))

    def test_normal_answer_reports_finish_reason_and_reasoning(self) -> None:
        response = {"model": "deepseek-flash", "choices": [{"finish_reason": "stop",
                   "message": {"content": "可用", "reasoning_content": "12345"}}],
                   "usage": {"completion_tokens": 3}}
        with mock.patch("campus_rag.llm._post_json", return_value=response):
            res = self._client().chat("s", "u")
        self.assertEqual(res.text, "可用")
        self.assertEqual(res.finish_reason, "stop")
        self.assertEqual(res.reasoning_chars, 5)

    def test_build_engine_passes_generation_params(self) -> None:
        """config 里的 temperature / max_tokens / thinking 必须真的传到客户端。

        原先 build_engine 只传了前四个位置参数，于是 config.temperature 和
        config.max_tokens 从来没生效——用户想靠调 max_tokens 解决截断，改了也没用。
        """
        captured = {}

        class _Capture:
            def __init__(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs

        cfg = Config()
        cfg.llm_api_key = "sk-dummy"
        cfg.use_llm = True
        cfg.temperature = 0.7
        cfg.max_tokens = 1234
        cfg.llm_thinking = "enabled"
        with mock.patch("ask.LLMClient", _Capture), \
             mock.patch("ask.Config") as cfg_cls, \
             mock.patch("ask.load_corpus", return_value=[SimpleNamespace(source="a.md")]), \
             mock.patch("ask.RagEngine"):
            cfg_cls.from_env.return_value = cfg
            cfg_cls.from_env.side_effect = None
            # build_engine 会顺手打印启动提示，测试里静音，别污染测试输出
            with contextlib.redirect_stdout(io.StringIO()):
                ask.build_engine(argparse.Namespace(
                    corpus="x", top_k=4, chunk_size=480, threshold=0.42, evidence_k=5,
                    llm=True, embeddings=False, offline=False, model="deepseek-flash",
                    extractive_chunks=None, info=False, check_llm=False,
                ))
        self.assertEqual(captured["kwargs"]["temperature"], 0.7)
        self.assertEqual(captured["kwargs"]["max_tokens"], 1234)
        self.assertEqual(captured["kwargs"]["thinking"], "enabled")


class CheckLlmCommandTest(unittest.TestCase):
    """`--check-llm`：把"能拿到答案吗"的原始证据打出来，而不是只证明 key 有效。"""

    def _engine(self, llm):
        cfg = Config()
        cfg.llm_api_key = "sk-dummy"
        cfg.use_llm = True
        return SimpleNamespace(cfg=cfg, llm=llm, chunks=[SimpleNamespace(source="a.md")])

    def _run(self, engine, **ns) -> tuple:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ask.cmd_check_llm(engine, argparse.Namespace(offline=False, **ns))
        return rc, buf.getvalue()

    def test_reports_full_diagnostics_on_success(self) -> None:
        from campus_rag.llm import LLMResult

        class _Ok:
            def list_models(self): return ["deepseek-flash"]
            def chat(self, s, u):
                return LLMResult(text="可用", mode="llm", model="deepseek-flash",
                                 latency_ms=7, finish_reason="stop", reasoning_chars=0,
                                 usage={"completion_tokens": 3})

        rc, out = self._run(self._engine(_Ok()))
        self.assertEqual(rc, 0)
        for token in ("finish_reason", "usage", "思维链长度", "可用"):
            self.assertIn(token, out)

    def test_surfaces_the_real_reason_when_it_fails(self) -> None:
        from campus_rag.llm import LLMError

        class _Boom:
            def list_models(self): return ["deepseek-flash"]
            def chat(self, s, u):
                raise LLMError("LLM 返回空内容（finish_reason=length｜思维链 1600 字）→ 建议：关掉 thinking")

        rc, out = self._run(self._engine(_Boom()))
        self.assertEqual(rc, 1)
        self.assertIn("finish_reason=length", out)   # 原因透传，不被吞掉
        self.assertIn("思维链", out)

    def test_skips_cleanly_when_llm_disabled(self) -> None:
        rc, out = self._run(SimpleNamespace(cfg=self._engine(None).cfg, llm=None, chunks=[]))
        self.assertEqual(rc, 1)
        self.assertIn("跳过", out)


class UiDegradeTest(unittest.TestCase):
    """bug：前端只要看到 error 字段，就把**已经降级好的答案**整条丢掉。

    服务端明明返回了完整的抽取式答案 + 引用来源（这正是"永不白屏"的设计），
    前端却只显示"服务端错误：LLM 返回空内容"——用户不但没看到原因，连答案都没了。
    修法：只有"真的没有 answer"才算服务端错误；降级走 renderCard，如实标注。
    """

    def test_ui_keeps_degraded_answer(self) -> None:
        ui = (ROOT / "campus_rag" / "ui.html").read_text(encoding="utf-8")
        self.assertIn("if(!d.answer)", ui)          # 判据改成"有没有答案"
        self.assertNotIn("if(d.error){ document", ui)  # 旧的短路口不许回来

    def test_engine_still_returns_answer_on_llm_failure(self) -> None:
        """行为断言：LLM 抛错时，Answer 必须同时带 answer 和 error。"""
        from campus_rag.llm import LLMError

        class _Boom:
            def chat(self, s, u): raise LLMError("LLM 返回空内容（finish_reason=length）")

        cfg = Config()
        cfg.use_llm = True
        cfg.answer_threshold = 0.0   # 保证不被闸门拦住，走到生成那一步
        engine = RagEngine([_chunk()], config=cfg, llm=_Boom())
        ans = engine.answer("宿舍几点关门")
        self.assertEqual(ans.mode, "extractive")
        self.assertTrue(ans.answer.strip())          # 答案在
        self.assertIn("LLM 返回空内容", ans.error)    # 原因也在


def _chunk():
    from campus_rag.data import Chunk

    return Chunk(
        chunk_id="c0", source="08-测试.md", title="测试", heading="测试 > 宿舍",
        text="宿舍大门每日 06:00 开；周日至周四 23:00 关门熄灯。迟归须登记。",
        position=0,
    )


class LlmToggleTest(unittest.TestCase):
    """功能：网页里的「是否使用大模型」开关（只在配了 key 时可用）。

    设计约束（都有测试盯住）：
    ① 开关是**按题**生效的（engine.answer(use_llm=...)），不改进程级配置——
       Web 是 ThreadingHTTPServer，改全局在并发下会互相串；
    ② 只要配了 key 就构造 LLMClient，否则没加 --llm 时开关是个摆设；
    ③ llm_available（能不能拨）与 llm_enabled（现在是开是关）是两件事；
    ④ 请求里没传 use_llm 时**必须用服务端默认**，不能被解析成 False 悄悄关掉。
    """

    def _engine_with(self, llm, use_llm_default: bool):
        cfg = Config()
        cfg.use_llm = use_llm_default
        cfg.answer_threshold = 0.0     # 别让闸门拦住，确保走到生成那一步
        return RagEngine([_chunk()], config=cfg, llm=llm)

    def test_switch_overrides_config_per_request(self) -> None:
        class _Spy:
            def __init__(self): self.calls = 0
            def chat(self, s, u):
                self.calls += 1
                return LLMResult(text="模型答案 [1]", mode="llm", model="m", latency_ms=1)

        spy = _Spy()
        # 服务端默认关，但这一题显式打开
        engine = self._engine_with(spy, use_llm_default=False)
        ans_on = engine.answer("宿舍几点关门", use_llm=True)
        self.assertEqual(ans_on.mode, "llm")
        self.assertEqual(spy.calls, 1)
        # 下一题不传 → 回到服务端默认（关），且**不能**因为上一题开过就被记住
        ans_default = engine.answer("宿舍几点关门")
        self.assertEqual(ans_default.mode, "extractive")
        self.assertEqual(spy.calls, 1)
        # 显式关掉时，即使默认是开也不该调用
        engine2 = self._engine_with(spy, use_llm_default=True)
        self.assertEqual(engine2.answer("宿舍几点关门", use_llm=False).mode, "extractive")
        self.assertEqual(spy.calls, 1)

    def test_trace_records_what_was_asked(self) -> None:
        class _Boom:
            def chat(self, s, u): raise LLMError("随便失败一下")

        engine = self._engine_with(_Boom(), use_llm_default=False)
        trace = engine.answer("宿舍几点关门", use_llm=True).trace
        self.assertTrue(trace["llm_requested"])      # 页面要了
        self.assertTrue(trace["llm_available"])      # 客户端在
        self.assertFalse(engine.answer("宿舍几点关门").trace["llm_requested"])

    def test_info_payload_separates_available_from_enabled(self) -> None:
        import campus_rag.web as web

        def payload(api_key: str, use_llm: bool):
            cfg = Config()
            cfg.llm_api_key = api_key
            cfg.use_llm = use_llm
            return web.build_info_payload(SimpleNamespace(
                cfg=cfg, chunks=[SimpleNamespace(source="a.md")],
                retriever=SimpleNamespace(semantic_enabled=False)))

        self.assertTrue(payload("sk-x", False)["llm_available"])   # 配了 key → 开关可拨
        self.assertFalse(payload("sk-x", False)["llm_enabled"])    # 但默认是关的
        self.assertFalse(payload("", True)["llm_available"])       # 没 key → 开关不可用
        self.assertFalse(payload("", True)["llm_enabled"])

    def test_offline_makes_switch_unavailable(self) -> None:
        import campus_rag.web as web

        cfg = Config()
        cfg.llm_api_key = "sk-x"
        cfg.use_llm = True
        engine = SimpleNamespace(cfg=cfg, chunks=[], retriever=SimpleNamespace(semantic_enabled=False))
        with mock.patch.dict(os.environ, {"CAMPUS_RAG_OFFLINE": "1"}):
            info = web.build_info_payload(engine)
        self.assertFalse(info["llm_available"])   # 强制离线时，开关必须点不动
        self.assertFalse(info["llm_enabled"])

    def test_request_flag_parsing(self) -> None:
        import campus_rag.web as web

        self.assertIs(web._as_bool(True), True)
        self.assertIs(web._as_bool(False), False)
        self.assertIs(web._as_bool("true"), True)
        self.assertIs(web._as_bool(" off "), False)
        self.assertIs(web._as_bool(1), True)
        # 关键：认不出来时返回 None（= 用服务端默认），不能猜成 False 把大模型关掉
        for junk in (None, "", "maybe", [], {}, "yes please"):
            self.assertIsNone(web._as_bool(junk), junk)

    def test_client_is_built_whenever_key_exists(self) -> None:
        """没加 --llm 时也要构造客户端，否则页面开关拨了没反应。"""
        captured = {}

        class _Capture:
            def __init__(self, *args, **kwargs):
                captured["built"] = True

        cfg = Config()
        cfg.llm_api_key = "sk-dummy"
        cfg.use_llm = False              # 注意：默认是关的
        cfg.llm_thinking = "disabled"
        with mock.patch("ask.LLMClient", _Capture), \
             mock.patch("ask.Config") as cfg_cls, \
             mock.patch("ask.load_corpus", return_value=[SimpleNamespace(source="a.md")]), \
             mock.patch("ask.RagEngine"):
            cfg_cls.from_env.return_value = cfg
            cfg_cls.from_env.side_effect = None
            with contextlib.redirect_stdout(io.StringIO()):
                ask.build_engine(argparse.Namespace(
                    corpus="x", top_k=4, chunk_size=480, threshold=0.42, evidence_k=5,
                    llm=False, embeddings=False, offline=False, model="deepseek-flash",
                    extractive_chunks=None, info=False, check_llm=False, web=True,
                ))
        self.assertTrue(captured.get("built"), "配了 key 就该构造客户端，否则开关是摆设")

    def test_ui_has_the_switch_and_sends_it(self) -> None:
        ui = (ROOT / "campus_rag" / "ui.html").read_text(encoding="utf-8")
        self.assertIn('id="llmtoggle"', ui)
        self.assertIn("llm_available", ui)                  # 开关可用性来自后端
        self.assertIn("use_llm: llmAvailable ? useLlm : false", ui)   # 每题带上开关
        self.assertIn("disabled", ui)                       # 不可用时禁用

    def test_switch_state_is_not_persisted(self) -> None:
        """开关的初始状态必须**只看服务端**，不许读本地记录。

        实测踩过：把选择存进 localStorage 后，存过的 "关" 会在下次带 --llm 启动时
        把服务端的"开"悄悄压掉——用户在 bat 里选了 [2]，页面开关却还是"关"，
        每题都发 use_llm:false，于是"用 bat 还是用不了大模型"。
        （注释里可以提 localStorage 解释原因，但代码不许读写。）
        """
        ui = (ROOT / "campus_rag" / "ui.html").read_text(encoding="utf-8")
        self.assertNotIn("localStorage.getItem", ui)
        self.assertNotIn("localStorage.setItem", ui)
        self.assertIn("useLlm = llmAvailable && llmDefault", ui)


class RunWebBatTest(unittest.TestCase):
    """`run_web.bat` 由 cmd.exe 执行，Python 测试跑不到它，只能盯它的**文件性质**。

    而这三个性质我真的都踩坏过（就在加"大模型开关"那一轮）：
    ① 纯 ASCII：cmd.exe 按**字节**偏移读批处理，UTF-8 中文会让它从词中间接着读，
       报一堆 `'indow:'` / `'EEK_API_KEY'` 不是内部或外部命令；
    ② CRLF 行尾：同理会让逐行解析错位；
    ③ `%key:"=%` 去引号之前必须先 `if not defined key` 兜住——
       空输入时 key 未定义，cmd 会把引号泄漏进下一行的 if，
       报 `="=="" goto offline was unexpected at this time`（实测复现过）。
    """

    def setUp(self) -> None:
        self.raw = (ROOT / "run_web.bat").read_bytes()

    def test_pure_ascii(self) -> None:
        """README 里那句"纯 ASCII，避免 bat 编码坑"要靠测试执行，不能靠记性。"""
        bad = sorted({b for b in self.raw if b > 127})
        self.assertEqual(bad, [], f"run_web.bat 里出现了非 ASCII 字节 {bad}；cmd 会解析错位")

    def test_crlf_line_endings(self) -> None:
        self.assertGreater(self.raw.count(b"\r\n"), 0)
        self.assertEqual(self.raw.replace(b"\r\n", b"").count(b"\n"), 0,
                         "存在裸 LF：cmd.exe 会错位解析")

    def test_no_quote_stripping_left_in_the_bat(self) -> None:
        """`%key:"=%` 在 key 未定义时会让 cmd 解析崩溃（实测过）。

        现在这段逻辑整体挪进了 Python（`ask.py --ask-key`：中文提示 + 读入 + 去引号
        + 校验），bat 里**不该再有**它。注释里提到这个坑是允许的，所以只看可执行行。
        """
        for line in self.raw.decode("ascii").splitlines():
            code = line.strip()
            if not code or code.lower().startswith("rem"):
                continue
            self.assertNotIn('%key:"=%', code, "去引号逻辑应留在 Python 侧，bat 不许再有它")
        # 没拿到 key 时应该交给 Python 那个中文引导流程，而不是自己 set /p
        bat = self.raw.decode("ascii")
        self.assertIn("ask.py --web --llm --ask-key", bat)
        self.assertNotIn("set /p key=", bat)


class DoctorTest(unittest.TestCase):
    """`doctor.py`：一条命令把"为什么用不了大模型"查到底。

    它存在的理由本身就是一条教训：前面几轮我全靠猜（key 没配上？没加 --llm？
    thinking 吃光 token？前端缓存？），来回好几轮没解决。体检脚本把每一环
    （环境变量 → 配置解析 → DNS/TCP → /models → /chat/completions → 旧服务端口）
    都变成可判定的输出，让人**看结论**而不是继续猜。
    """

    def test_mask_never_leaks_the_middle_of_a_key(self) -> None:
        """报告可能被贴到聊天窗口里——只许露前 6 位和末 4 位。"""
        import doctor

        key = "sk-abcdefghijklmnopqrstuvwxyz1234"
        masked = doctor._mask(key)
        self.assertTrue(masked.startswith("sk-abc"))
        self.assertTrue(masked.endswith("1234"))
        self.assertNotIn("hijklmnop", masked)          # 中间那段绝不能出现
        self.assertEqual(doctor._mask(""), "(空)")
        self.assertNotIn("secret", doctor._mask("secret"))  # 太短的整段打码

    def test_host_port_parsing(self) -> None:
        import doctor

        self.assertEqual(doctor._host_port("https://api.deepseek.com"), ("api.deepseek.com", 443))
        self.assertEqual(doctor._host_port("http://127.0.0.1:8201/v1"), ("127.0.0.1", 8201))
        self.assertEqual(doctor._host_port("api.deepseek.com"), ("api.deepseek.com", 443))

    def test_doctor_is_wired_into_the_bat(self) -> None:
        """bat 里必须能直接调起体检——否则用户还得自己找窗口、自己配环境。"""
        bat = (ROOT / "run_web.bat").read_text(encoding="ascii")
        self.assertIn("goto doctor", bat)
        self.assertIn("py -3 doctor.py", bat)


class ApiKeyPromptTest(unittest.TestCase):
    """`ask.py --ask-key`：中文引导用户粘贴 key（从 .bat 挪进 Python 的那段）。

    为什么要挪：`run_web.bat` 必须纯 ASCII（cmd.exe 按字节偏移解析，中文会让它
    从词中间读起），所以那里只能写英文——而"第一步教用户抄 key"恰恰最需要中文。
    挪进 Python 还顺带保证 key 落在**将要运行服务的那个进程**里，
    不必再靠 cmd 的 set 传递（"$env: 只对当前窗口有效"就是这么坑了好几轮）。
    """

    def setUp(self) -> None:
        self._saved = os.environ.pop("DEEPSEEK_API_KEY", None)
        self._saved2 = os.environ.pop("CAMPUS_RAG_API_KEY", None)

    def tearDown(self) -> None:
        os.environ.pop("DEEPSEEK_API_KEY", None)
        os.environ.pop("CAMPUS_RAG_API_KEY", None)
        if self._saved:
            os.environ["DEEPSEEK_API_KEY"] = self._saved
        if self._saved2:
            os.environ["CAMPUS_RAG_API_KEY"] = self._saved2

    def _run(self, answers) -> tuple:
        buf = io.StringIO()
        with mock.patch("builtins.input", side_effect=answers), contextlib.redirect_stdout(buf):
            ok = ask.prompt_for_api_key()
        return ok, buf.getvalue()

    def test_accepts_a_key_and_puts_it_in_this_process(self) -> None:
        ok, out = self._run(["sk-abcdefghijklmnop"])
        self.assertTrue(ok)
        # 关键：写进**当前进程**，随后的 Config.from_env 才读得到
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-abcdefghijklmnop")
        self.assertIn("sk-abc...mnop", out)

    def test_quotes_and_spaces_are_stripped(self) -> None:
        ok, _ = self._run(['  "sk-quotedkey123456"  '])
        self.assertTrue(ok)
        self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "sk-quotedkey123456")

    def test_never_prints_the_whole_key(self) -> None:
        """终端可能被截图或贴到聊天窗口里——只许出现掩码。"""
        ok, out = self._run(["sk-supersecretmiddle9999"])
        self.assertTrue(ok)
        self.assertNotIn("supersecretmiddle", out)
        self.assertIn("9999", out)          # 末 4 位用于让用户核对

    def test_empty_input_retries_then_falls_back(self) -> None:
        ok, out = self._run(["", "   ", ""])
        self.assertFalse(ok)
        self.assertNotIn("DEEPSEEK_API_KEY", os.environ)   # 不能把空串当成 key
        self.assertIn("离线模式", out)

    def test_eof_does_not_crash(self) -> None:
        ok, _ = self._run([EOFError])
        self.assertFalse(ok)
        self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_warns_on_a_non_sk_prefix_but_still_accepts(self) -> None:
        """别的 OpenAI 兼容网关前缀可能不是 sk-，所以只提醒不拦截。"""
        ok, out = self._run(["my-gateway-token-1234"])
        self.assertTrue(ok)
        self.assertIn("不是以", out)


class LlmProbeTest(unittest.TestCase):
    """开关能不能拨，取决于**启动时那次探测真的成功**，而不是"有没有配 key"。

    用户原话："没连 llm 时『使用大模型』按钮还能用这不合理"——
    只判断 key 存在的话，key 写错时开关照样能拨，然后每题静默降级。
    """

    def _engine(self, status, api_key="sk-dummy"):
        cfg = Config()
        cfg.llm_api_key = api_key
        cfg.use_llm = True
        return SimpleNamespace(cfg=cfg, chunks=[SimpleNamespace(source="a.md")],
                               retriever=SimpleNamespace(semantic_enabled=False),
                               llm_status=status)

    def test_probe_failure_disables_the_switch_with_the_real_reason(self) -> None:
        import campus_rag.web as web

        info = web.build_info_payload(self._engine(
            {"checked": True, "ok": False, "error": "HTTP 401（API key 无效）"}))
        self.assertTrue(info["llm_key_configured"])   # key 是配了的……
        self.assertFalse(info["llm_available"])       # ……但用不了，所以开关不可拨
        self.assertFalse(info["llm_enabled"])
        self.assertIn("401", info["llm_reason"])      # 原因要如实透出来

    def test_probe_success_keeps_the_switch_usable(self) -> None:
        import campus_rag.web as web

        info = web.build_info_payload(self._engine({"checked": True, "ok": True, "error": ""}))
        self.assertTrue(info["llm_available"])
        self.assertTrue(info["llm_enabled"])

    def test_without_probe_it_falls_back_to_key_presence(self) -> None:
        """非 Web 路径（比如直接构造引擎）没有探测结果时，保持原行为，别误锁。"""
        import campus_rag.web as web

        info = web.build_info_payload(self._engine({}))
        self.assertTrue(info["llm_available"])

    def test_no_key_is_unavailable_regardless_of_probe(self) -> None:
        import campus_rag.web as web

        info = web.build_info_payload(self._engine({"checked": True, "ok": True}, api_key=""))
        self.assertFalse(info["llm_available"])
        self.assertFalse(info["llm_key_configured"])

    def test_probe_reports_model_name_mismatch(self) -> None:
        """key 有效但模型名不对，同样是"用不了"，必须挡住。"""
        from campus_rag.llm import probe_llm

        cfg = Config()
        cfg.llm_api_key = "sk-dummy"
        cfg.llm_model = "deepseek-does-not-exist"
        with mock.patch("campus_rag.llm.LLMClient.list_models", return_value=["deepseek-flash"]):
            status = probe_llm(cfg)
        self.assertFalse(status["ok"])
        self.assertIn("不在账号可用列表", status["error"])

    def test_ui_offers_a_recheck(self) -> None:
        ui = (ROOT / "campus_rag" / "ui.html").read_text(encoding="utf-8")
        self.assertIn("/api/llm-check", ui)      # 瞬时失败不该逼用户重启服务
        self.assertIn("llm_key_configured", ui)  # 没配 key 时不显示"重新检测"


if __name__ == "__main__":
    unittest.main(verbosity=2)
