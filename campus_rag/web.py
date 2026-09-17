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

from .engine import RagEngine
from .qa_logger import log_answer

UI_PATH = Path(__file__).resolve().parent / "ui.html"
_LOCK = threading.Lock()


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
                stats = {
                    "chunks": len(engine.chunks),
                    "docs": sorted({c.source for c in engine.chunks}),
                    "config": engine.cfg.to_dict(),
                    "semantic_enabled": engine.retriever.semantic_enabled,
                }
                self._send(200, json.dumps(stats, ensure_ascii=False).encode("utf-8"), "application/json")
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
