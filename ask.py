"""CLI 入口：ask（单问） / chat（交互） / eval（评测） / web（浏览器 Demo） / info（自检）。

用法示例：
    python ask.py "推免需要什么条件？"           # 离线也能跑
    python ask.py --web                          # 打开浏览器 Demo
    python ask.py --eval                         # 跑离线评测，出指标
    python ask.py --eval --llm                   # 评测时连上大模型（需 API key）
    python ask.py --info                         # 检查环境与可选依赖状态
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 允许 `python ask.py` 直接运行（把项目根加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _use_utf8_console() -> None:
    """把标准输出切到 UTF-8，避免 Windows 中文命令行（GBK）打印中文时崩溃。

    这是一个真实的坑：双击 .bat 运行时，控制台代码页常是 936(GBK)，
    Python 用 GBK 编码输出中文标点/emoji 会直接抛 UnicodeEncodeError，
    用户看到的是"闪一下就没了"。errors='replace' 保证即使编码失败也只是显示成 ?，
    绝不会中断程序。任何面向新手的工具都必须先处理这一步。
    """
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


_use_utf8_console()

from campus_rag.config import Config  # noqa: E402
from campus_rag.data import load_corpus  # noqa: E402
from campus_rag.engine import RagEngine  # noqa: E402
from campus_rag.evaluate import format_report, load_eval_set, run_eval  # noqa: E402
from campus_rag.llm import EmbeddingClient, LLMClient, LLMError  # noqa: E402
from campus_rag.qa_logger import log_answer  # noqa: E402

ROOT = Path(__file__).resolve().parent
BANNER = r"""
   ___   _   _  _  _ ___ ___    ___   _   ___
  / __| /_\ | \| || | _ \ _ \  | _ \ /_\ / __|  校园资料问答助手
 | (__ / _ \| .` || |  _/   /  |   // _ \ (_ |  Retrieval-Augmented Generation
  \___/_/ \_\_|\_||_|_| |_|_\  |_|_/_/ \_\___|  零依赖 · 可引用 · 会拒答
"""


def build_engine(args: argparse.Namespace) -> RagEngine:
    cfg = Config.from_env(
        corpus_dir=args.corpus,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        answer_threshold=args.threshold,
        evidence_k=args.evidence_k,
        use_llm=getattr(args, "llm", False) or bool(os.environ.get("CAMPUS_RAG_API_KEY")),
        use_embeddings=getattr(args, "embeddings", False),
        llm_model=args.model,
    )
    if getattr(args, "offline", False):
        cfg.use_llm = False
        cfg.use_embeddings = False
    if getattr(args, "extractive_chunks", None):
        RagEngine.extractive_chunks = args.extractive_chunks

    chunks = load_corpus(cfg.corpus_dir, max_len=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunks:
        raise SystemExit(f"语料为空：{cfg.corpus_dir}。请放入 .md/.txt 文件。")

    llm = LLMClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model, cfg.llm_timeout) if cfg.use_llm else None
    embedder = (
        EmbeddingClient(cfg.embed_base_url, cfg.embed_api_key, cfg.embed_model)
        if cfg.use_embeddings
        else None
    )

    t0 = time.time()
    engine = RagEngine(chunks, config=cfg, llm=llm, embedder=embedder)
    engine.build_ms = int((time.time() - t0) * 1000)  # type: ignore[attr-defined]
    return engine


def render_answer(ans, show_trace: bool = True) -> str:
    mode_label = {
        "llm": "大模型生成（严格接地）",
        "extractive": "离线抽取式降级",
        "refusal": "拒答",
    }.get(ans.mode, ans.mode)
    out = [
        "",
        f"[问] {ans.question}",
        f"── 回答模式：{mode_label} | 置信度：{ans.confidence:.2f} | 耗时：{ans.latency_ms} ms",
        "",
        ans.answer,
    ]
    if ans.error:
        out.append(f"\n[注意] 已降级（原因：{ans.error[:120]}）")
    if ans.sources and ans.mode != "refusal":
        out.append("\n[引用来源]")
        for s in ans.sources:
            out.append(f"  [{s['index']}] {s['source']} · {s['heading']}  (相关度 {s['score']})")
            out.append(f"      “{s['snippet']}”")
    if show_trace and ans.trace:
        t = ans.trace
        out.append(
            f"\n[检索轨迹] 词法={t.get('lexical_top', 0):.2f} 向量={t.get('vector_top', 0):.2f} "
            f"语义={'开' if t.get('semantic_enabled') else '关'} 命中词={'/'.join(t.get('matched_terms', [])[:8])}"
        )
    return "\n".join(out)


def cmd_ask(engine: RagEngine, args: argparse.Namespace) -> int:
    ans = engine.answer(args.question)
    log_answer(ans.to_dict(), extra={"channel": "cli"})
    if args.json:
        print(json.dumps(ans.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(render_answer(ans, show_trace=not args.no_trace))
    return 0


def cmd_chat(engine: RagEngine, args: argparse.Namespace) -> int:
    print(BANNER)
    print(f"已加载 {len(engine.chunks)} 个知识块，语料目录：{engine.cfg.corpus_dir}")
    print("输入问题开始问答；:q 退出，:s 查看检索详情。\n")
    while True:
        try:
            q = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if not q:
            continue
        if q in {":q", ":quit", "exit", "quit"}:
            print("再见！")
            return 0
        ans = engine.answer(q)
        log_answer(ans.to_dict(), extra={"channel": "cli"})
        print(render_answer(ans, show_trace=q.startswith(":s") or True))


def cmd_eval(engine: RagEngine, args: argparse.Namespace) -> int:
    cases = load_eval_set(args.eval_set)
    print(f"载入评测集 {len(cases)} 题（{args.eval_set}），开始评测…\n")

    def progress(i: int, total: int, r) -> None:
        filled = int(20 * i / total)
        bar = "#" * filled + "-" * (20 - filled)
        sys.stdout.write(f"\r  [{bar}] {i}/{total}  {r.id} mode={r.mode:<10}")
        sys.stdout.flush()

    report = run_eval(engine, cases, k=args.eval_k, progress=None if args.json else progress)
    if not args.json:
        sys.stdout.write("\r" + " " * 70 + "\r")
        print(format_report(report, show_cases=not args.quiet))
    if args.out:
        Path(args.out).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整报告已写入：{args.out}")
    return 0


def cmd_info(engine: RagEngine, args: argparse.Namespace) -> int:
    cfg = engine.cfg
    print(BANNER)
    print("【语料】")
    print(f"  目录：{cfg.corpus_dir}")
    print(f"  知识块数：{len(engine.chunks)}")
    sources = sorted({c.source for c in engine.chunks})
    print(f"  文档数：{len(sources)}")
    for s in sources:
        n = sum(1 for c in engine.chunks if c.source == s)
        print(f"    - {s}（{n} 块）")
    print(f"  索引构建耗时：{getattr(engine, 'build_ms', 0)} ms")
    print("\n【检索/生成配置】")
    for k, v in cfg.to_dict().items():
        if k in {"llm_api_key", "embed_api_key"}:
            continue
        print(f"  {k} = {v}")
    print("\n【可选能力状态】")
    print(f"  大模型生成：{'已启用 ' + cfg.llm_model if cfg.use_llm else '未启用（走离线抽取式降级）'}")
    print(f"  语义向量通道：{'已启用 ' + cfg.embed_model if cfg.use_embeddings else '未启用（走 BM25+TF-IDF 融合）'}")
    if cfg.llm_api_key:
        try:
            models = LLMClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model).list_models()
            print(f"  API 自检: 可用，模型 {models[:6]}")
        except LLMError as e:
            print(f"  API 自检: 失败 - {e}")
    else:
        print("  API 自检: 未配置 key（设置 DEEPSEEK_API_KEY 后自动启用）")
    print("\n【提示】")
    print("  离线演示： python ask.py --web            （无需网络，必不翻车）")
    print("  联网增强： $env:DEEPSEEK_API_KEY='sk-xxx'; python ask.py --web --llm")
    return 0


def cmd_web(engine: RagEngine, args: argparse.Namespace) -> int:
    from campus_rag.web import serve

    serve(engine, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="ask.py",
        description="校园资料问答助手（RAG）· 零依赖 · 可引用 · 会拒答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("question", nargs="?", default=None, help="要提问的问题；不填则进入交互模式")
    p.add_argument("--web", action="store_true", help="启动浏览器 Demo")
    p.add_argument("--eval", action="store_true", help="运行离线评测")
    p.add_argument("--info", action="store_true", help="打印语料与配置自检信息")
    p.add_argument("--ask", action="store_true", help="单次提问模式")
    p.add_argument("--chat", action="store_true", help="强制进入交互模式")
    p.add_argument("--llm", action="store_true", help="启用大模型生成（需 API key）")
    p.add_argument("--embeddings", action="store_true", help="启用语义向量检索（需 embedding key）")
    p.add_argument("--offline", action="store_true", help="强制离线：不调用任何网络接口")
    p.add_argument("--corpus", default=str(ROOT / "data" / "corpus"), help="语料目录")
    p.add_argument("--eval-set", default=str(ROOT / "data" / "logs" / "eval_set.json"), help="评测集路径")
    p.add_argument("--eval-k", type=int, default=3, help="评测的 K 值")
    p.add_argument("--top-k", type=int, default=4, help="召回块数")
    p.add_argument("--chunk-size", type=int, default=480, help="分块最大字数")
    p.add_argument("--evidence-k", type=int, default=5, help="评估证据时纳入的候选块数（影响拒答判定）")
    p.add_argument("--extractive-chunks", type=int, default=3, help="离线抽取式回答从几块里挑句子")
    p.add_argument("--threshold", type=float, default=0.42, help="拒答置信度阈值（由 sweep.py 标定）")
    p.add_argument("--model", default="deepseek-flash", help="LLM 模型名（如 deepseek-flash / deepseek-v4-pro）")
    p.add_argument("--json", action="store_true", help="以 JSON 输出（便于脚本调用）")
    p.add_argument("--out", default=None, help="评测报告输出路径")
    p.add_argument("--quiet", action="store_true", help="评测只输出汇总，不打逐题明细")
    p.add_argument("--no-trace", action="store_true", help="不打印检索轨迹")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = p.parse_args(argv)

    engine = build_engine(args)

    if args.info:
        return cmd_info(engine, args)
    if args.eval:
        return cmd_eval(engine, args)
    if args.web:
        return cmd_web(engine, args)
    if args.question:
        return cmd_ask(engine, args)
    return cmd_chat(engine, args)


if __name__ == "__main__":
    raise SystemExit(main())
