"""Web Demo：标准库 http.server + 单文件前端（无构建、无 npm、无 CDN 依赖）。

为什么不用 Streamlit / React？
面试现场最怕"再装一个依赖"。这里 `python ask.py --web` 就能开，断网也能开。
前端从 campus_rag/ui.html 读取（与 Python 分离，方便你改样式后直接刷新看效果）。
"""
from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .config import llm_disabled_reason, offline_requested
from .engine import RagEngine
from .qa_logger import log_answer

UI_PATH = Path(__file__).resolve().parent / "ui.html"
_LOCK = threading.Lock()


def build_info_payload(engine: RagEngine) -> Dict[str, Any]:
    """`/api/info` 的响应体。

    单独抽成函数是为了可测——这里踩过一个真实的坑：
    前端徽章原先读 `config.llm_api_key`（"有没有配 key"）来决定显示"大模型生成 开/关"，
    而真正的开关是 `use_llm`。于是"配了 key 但启动时没加 --llm"时，
    页面右上角亮着绿灯写"开"，下面每条回答却都是"离线抽取式降级"——
    用户看到的就是"我明明配了 API，为什么还是离线模式"。

    现在把**结论**（llm_enabled / llm_reason）由后端算好交给前端，前端不再自己猜；
    判断口径也只有 `config.llm_disabled_reason` 这一处（同一个教训见 bug 6）。
    """
    cfg = engine.cfg
    config = cfg.to_dict()
    # 把两个 key 字段整个摘掉：它们在 to_dict 里已被脱敏成 "***"，
    # 浏览器拿不到真值、却会因为"非空"而误判（旧徽章就是这么被骗的）。
    # 命令行 --info 早就不打印这两个字段了，这里与它保持一致。
    for secret in ("llm_api_key", "embed_api_key"):
        config.pop(secret, None)
    return {
        "chunks": len(engine.chunks),
        "docs": sorted({c.source for c in engine.chunks}),
        "config": config,
        "semantic_enabled": engine.retriever.semantic_enabled,
        "llm_enabled": bool(cfg.use_llm),
        "llm_model": cfg.llm_model,
        "llm_reason": llm_disabled_reason(cfg, offline=offline_requested()),
    }


def _build_payload(engine: RagEngine, question: str) -> Dict[str, Any]:
    started = time.time()
    with _LOCK:  # 纯 Python 检索是 CPU 密集，串行化避免互相抢 GIL 导致延迟抖动
        ans = engine.answer(question)
    payload = ans.to_dict()
    payload["server_ms"] = int((time.time() - started) * 1000)
    payload["stats"] = {
        "chunks": len(engine.chunks),
        "docs": len({c.source for c in engine.chunks}),
        "answer_threshold": engine.cfg.answer_threshold,
        "llm": engine.cfg.use_llm,
        "embedding": engine.retriever.semantic_enabled,
        "corpus_dir": engine.cfg.corpus_dir,
    }
    log_answer(payload, extra={"channel": "web", "server_ms": payload["server_ms"]})
    return payload


def make_handler(engine: RagEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "CampusRAG/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:  # 安静一点
            if self.path.startswith("/api/"):
                print(f"  [api] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                try:
                    html = UI_PATH.read_text(encoding="utf-8")
                except FileNotFoundError:
                    self._send(500, "ui.html 缺失".encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if self.path == "/api/info":
                self._send(
                    200,
                    json.dumps(build_info_payload(engine), ensure_ascii=False).encode("utf-8"),
                    "application/json",
                )
                return
            if self.path == "/health":
                self._send(200, b'{"ok":true}', "application/json")
                return
            self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/ask":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError:
                self._send(400, b'{"error":"bad json"}', "application/json")
                return
            question = (data.get("question") or "").strip()
            if not question:
                self._send(400, b'{"error":"empty question"}', "application/json")
                return
            try:
                payload = _build_payload(engine, question[:500])
            except Exception as e:  # noqa: BLE001 — 任何异常都以 JSON 返回，前端不会白屏
                self._send(
                    500,
                    json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode("utf-8"),
                    "application/json",
                )
                return
            self._send(200, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json")

    return Handler


def serve(engine: RagEngine, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True) -> None:
    handler = make_handler(engine)
    httpd = None
    for candidate in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, candidate), handler)
            port = candidate
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"端口 {port}-{port + 19} 都被占用，请用 --port 指定其它端口")

    url = f"http://{host}:{port}"
    mode = []
    mode.append("LLM 生成: " + ("开" if engine.cfg.use_llm else "关（离线抽取式）"))
    mode.append("语义向量: " + ("开" if engine.retriever.semantic_enabled else "关（BM25+TF-IDF）"))
    print("\n" + "=" * 62)
    print("  校园资料问答助手 - Web Demo 已启动")
    print("=" * 62)
    print(f"  请在浏览器打开这个地址: {url}")
    print(f"  知识块    : {len(engine.chunks)} 块 / {len({c.source for c in engine.chunks})} 篇文档")
    print(f"  运行模式  : {' | '.join(mode)}")
    print(f"  拒答阈值  : {engine.cfg.answer_threshold}")
    print("  停止服务  : 在这个黑窗口里按 Ctrl+C，或直接关闭窗口")
    print("=" * 62 + "\n")
    try:
        sys.stdout.flush()  # 立即刷出地址，否则输出被缓冲时用户看不到提示
    except Exception:
        pass
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务…")
    finally:
        httpd.server_close()
