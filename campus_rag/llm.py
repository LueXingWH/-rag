"""LLM 与 Embedding 客户端（OpenAI 兼容协议，纯标准库 urllib 实现）。

关键工程决策：**LLM 是可选的、可失败的**。
它只负责"把检索结果组织成通顺的中文"，检索层完全不依赖它。
API 超时/401/429/没网 → 自动降级到抽取式回答，demo 永不白屏。
这是面试里很值得讲的一点：把不确定性隔离在一层薄薄的、可降级的边界内。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class LLMError(RuntimeError):
    pass


def _post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:400]
        hint = {401: "API key 无效", 402: "余额不足", 429: "触发限流"}.get(e.code, "请求失败")
        raise LLMError(f"HTTP {e.code}（{hint}）: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"网络不可用: {e.reason}") from e
    except TimeoutError as e:
        raise LLMError("请求超时") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise LLMError(f"返回不是合法 JSON: {body[:200]}") from e


@dataclass
class LLMResult:
    text: str
    mode: str          # "llm" | "extractive"
    model: str
    latency_ms: int
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    finish_reason: str = ""
    reasoning_chars: int = 0   # 思维链长度（thinking 模式下诊断"答案被吃掉"的关键证据）


def _empty_content_error(choice: Dict[str, Any], reasoning: str, usage: Any) -> str:
    """把"返回空内容"翻译成能直接照着修的一句话。

    这条信息原本是 `LLM 返回空内容` 六个字，用户只能干瞪眼。实际上响应里
    已经把原因写好了——finish_reason、思维链长度、token 用量——只是没人读。
    实测最典型的两种：
    - thinking 模式默认开启（effort=high），思维链把 max_tokens 吃光 → content 为空；
    - 单纯 max_tokens 太小，答案还没写完就被截断。
    """
    finish = (choice.get("finish_reason") or "未提供") if isinstance(choice, dict) else "未提供"
    used = ""
    if isinstance(usage, dict):
        in_tok, out_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
        if out_tok is not None:
            used = f"｜输出 tokens {out_tok}" + (f"（输入 {in_tok}）" if in_tok is not None else "")
    chain = f"｜思维链 {len(reasoning)} 字" if reasoning else "｜无思维链"
    if reasoning and finish == "length":
        advice = ("思维链占满了 max_tokens，答案还没开始写就被截断。"
                  "建议：关掉 thinking（CAMPUS_RAG_THINKING=disabled，或 config.llm_thinking=\"disabled\"，本项目默认已关），"
                  "或把 max_tokens 调大")
    elif finish == "length":
        advice = "答案被 max_tokens 截断。建议：把 max_tokens 调大"
    else:
        advice = "上游返回了空正文。建议：先跑 python ask.py --check-llm 看完整诊断，再检查网关是否有内容过滤"
    return f"LLM 返回空内容（finish_reason={finish}{chain}{used}）→ {advice}"


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 700,
        thinking: str = "disabled",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = (thinking or "").strip().lower()
        self._model_cache: Optional[str] = None

    # ---------------- chat ----------------
    def build_payload(self, system: str, user: str) -> Dict[str, Any]:
        """构造请求体。单独抽出来是为了能被测试直接断言（不必真的发请求）。"""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        # enabled / disabled 才发这个字段；omit（或空）表示"别带厂商扩展字段"，
        # 给严格校验请求体的第三方 OpenAI 兼容网关留一条活路。
        if self.thinking in {"enabled", "disabled"}:
            payload["thinking"] = {"type": self.thinking}
        return payload

    def chat(self, system: str, user: str) -> LLMResult:
        payload = self.build_payload(system, user)
        started = time.time()
        data = _post_json(f"{self.base_url}/chat/completions", payload, self.api_key, self.timeout)
        latency = int((time.time() - started) * 1000)
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"返回结构异常: {json.dumps(data, ensure_ascii=False)[:300]}") from e
        if not text.strip():
            # 绝不把"空内容"当成正常结果往下传：这里一次性把可诊断的信息全带上，
            # 否则调用方（engine → 前端）只能显示一句无从下手的六个字。
            raise LLMError(_empty_content_error(choice, reasoning, data.get("usage")))
        return LLMResult(
            text=text.strip(),
            mode="llm",
            model=data.get("model", self.model),
            latency_ms=latency,
            usage=data.get("usage"),
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning_chars=len(reasoning),
        )

    def list_models(self) -> List[str]:
        """GET /models，用于起步时自检 key 与可用模型名（避免模型名写错踩坑）。"""
        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001 — 自检失败不应中断主流程
            raise LLMError(f"列出模型失败: {e}") from e
        ids = [m.get("id", "") for m in data.get("data", []) if isinstance(m, dict)]
        self._model_cache = next((m for m in ids if m), None)
        return ids


class EmbeddingClient:
    """OpenAI 兼容 /embeddings。用于把词法检索升级为语义检索。"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0, batch: int = 16) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch = batch

    def _embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i : i + self.batch]
            data = _post_json(
                f"{self.base_url}/embeddings",
                {"model": self.model, "input": batch},
                self.api_key,
                self.timeout,
            )
            items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
            out.extend([item.get("embedding", []) for item in items])
        return out

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embed([text])[0]
