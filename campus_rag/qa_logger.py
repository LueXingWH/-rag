"""问答日志：每次问答追加一行 JSON 到 data/logs/qa_YYYY-MM-DD.jsonl。

为什么用 JSONL（每行一个 JSON）而不是一个大 JSON 数组？
- 追加写：不用每次读全量再写回，O(1) 追加；
- 抗崩溃：写到一半崩了，最多丢最后半行，前面的行依然合法；
- 好分析：pandas / jq / grep 都能直接读，按天切分也方便归档。

线程安全：web 是 ThreadingHTTPServer，多线程并发写同一文件必须加锁。
容错原则：日志失败绝不能影响主流程——问答是主路径，日志是旁路。
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "logs"


def log_answer(
    payload: Dict[str, Any],
    log_dir: Path | str = _DEFAULT_DIR,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """把一次问答结果追加到按天切分的 JSONL 日志。

    payload 通常就是 Answer.to_dict() 的返回值。
    extra 可附带 {"channel": "web" / "cli", "server_ms": ...} 之类的来源信息。
    返回写入的文件路径；失败返回 None（静默）。
    """
    try:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        record: Dict[str, Any] = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "question": payload.get("question"),
            "answer": payload.get("answer"),
            "mode": payload.get("mode"),              # llm | extractive | refusal
            "confidence": payload.get("confidence"),
            "latency_ms": payload.get("latency_ms"),
            "error": payload.get("error"),
            "sources": [
                {
                    "source": s.get("source"),
                    "heading": s.get("heading"),
                    "score": s.get("score"),
                }
                for s in (payload.get("sources") or [])
            ],
        }
        if extra:
            record.update(extra)
        day = datetime.now().strftime("%Y-%m-%d")
        path = log_dir / f"qa_{day}.jsonl"
        line = json.dumps(record, ensure_ascii=False)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return path
    except Exception:
        return None
