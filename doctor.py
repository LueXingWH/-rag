"""一条命令查清"为什么大模型用不了"。

为什么要有它：前面几轮我都是靠猜（key 没配上？没加 --llm？thinking 吃光 token？
前端缓存？），来回好几轮。这个脚本把**所有环节**按顺序查一遍，每步给出可判定的结论，
最后直接告诉你卡在哪一环、下一步做什么。

    python doctor.py

退出码：0 = 大模型这一层是通的；1 = 有问题（原因会打印出来）。
报告同时写到 data/logs/doctor_report.txt，方便直接发给人看。
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from campus_rag.config import Config, llm_disabled_reason, offline_requested  # noqa: E402
from campus_rag.llm import LLMClient, LLMError  # noqa: E402

REPORT = ROOT / "data" / "logs" / "doctor_report.txt"
_WEB_PORTS = range(8000, 8020)


def _mask(key: str) -> str:
    """只显示前缀和末 4 位——报告可能会被贴到聊天窗口里。"""
    if not key:
        return "(空)"
    if len(key) <= 12:
        return key[:3] + "***"
    return f"{key[:6]}...{key[-4:]}"


def _host_port(url: str) -> tuple:
    """从 base_url 解析出 (host, port)。

    没写 scheme 时按 **https** 处理、端口按 443——不能落到 80。
    这不是吹毛求疵：探错端口会报出假的"连不上"，让人去查网络，
    而真正的原因是解析假设错了（这个 bug 就是被测试当场抓出来的）。
    """
    parts = urllib.parse.urlparse(url if "//" in url else "https://" + url)
    scheme = parts.scheme or "https"
    return (parts.hostname or "api.deepseek.com", parts.port or (443 if scheme == "https" else 80))


def _listening_ports() -> list:
    """本机 8000-8019 上还有谁在监听——旧的 demo 进程会让浏览器看到旧页面。"""
    alive = []
    for port in _WEB_PORTS:
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                alive.append(port)
    return alive


def main() -> int:
    lines = []
    problems = []

    def say(text: str = "") -> None:
        print(text)
        lines.append(text)

    say("=" * 68)
    say("  校园资料问答 · 大模型接入体检")
    say("=" * 68)

    # ---- 1. 环境变量：这个进程到底看不看得见 key ----
    say("\n【1/6】环境变量（当前这个进程看得见什么）")
    env_keys = ["DEEPSEEK_API_KEY", "CAMPUS_RAG_API_KEY", "OPENAI_API_KEY"]
    seen = {k: os.environ.get(k, "") for k in env_keys}
    hit = [k for k, v in seen.items() if v]
    for k, v in seen.items():
        say(f"  {k:<22}{'已设置 ' + _mask(v) if v else '（未设置）'}")
    if not hit:
        problems.append("没有任何 API key 环境变量。若你在别的窗口用过 $env:，那个变量"
                        "只属于那个窗口；双击 .bat 会新开进程，看不到它。")
        say("  ⚠ 结论：这个进程读不到任何 key —— 后面几步必然失败。")
    else:
        say(f"  ✅ 生效的变量：{hit[0]}（优先级 CAMPUS_RAG_API_KEY > DEEPSEEK_API_KEY > OPENAI_API_KEY）")

    # ---- 2. 配置解析 ----
    say("\n【2/6】配置解析")
    cfg = Config.from_env(use_llm=True)
    offline = offline_requested()
    reason = llm_disabled_reason(cfg, offline=offline)
    say(f"  接口        {cfg.llm_base_url}")
    say(f"  模型        {cfg.llm_model}")
    say(f"  思维链      {cfg.llm_thinking}（本项目默认 disabled）")
    say(f"  生成参数    temperature={cfg.temperature} max_tokens={cfg.max_tokens} timeout={cfg.llm_timeout}s")
    say(f"  强制离线    {offline}")
    say(f"  读取到的 key {_mask(cfg.llm_api_key)}")
    say(f"  use_llm     {cfg.use_llm}" + (f"（未启用原因：{reason}）" if reason else ""))
    if offline:
        problems.append("处于强制离线状态（--offline 或 CAMPUS_RAG_OFFLINE=1），任何联网都不会发生。")
    elif not cfg.llm_api_key:
        problems.append("配置解析后 llm_api_key 为空 —— 环境变量没被读到。")

    # ---- 3. 网络可达性（纯 socket，不依赖任何库）----
    say("\n【3/6】网络可达性（DNS + TCP）")
    host, port = _host_port(cfg.llm_base_url)
    say(f"  目标        {host}:{port}")
    t0 = time.time()
    try:
        socket.create_connection((host, port), timeout=8).close()
        say(f"  ✅ 可连接（{int((time.time() - t0) * 1000)} ms）")
        net_ok = True
    except Exception as e:
        say(f"  ❌ 连不上：{type(e).__name__}: {e}")
        problems.append(f"TCP 连不上 {host}:{port} —— 网络/代理/防火墙问题，"
                        "不是 key 的问题。若本机需要代理，请设置 HTTPS_PROXY。")
        net_ok = False

    # ---- 4./5. 真的调两次接口 ----
    client = LLMClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model, timeout=min(cfg.llm_timeout, 30.0),
                       temperature=cfg.temperature, max_tokens=cfg.max_tokens, thinking=cfg.llm_thinking)
    models = []
    if not cfg.llm_api_key or offline or not net_ok:
        say("\n【4/6】GET /models —— 跳过（前面已经失败）")
        say("\n【5/6】POST /chat/completions —— 跳过")
    else:
        say("\n【4/6】GET /models（证明 key 有效）")
        try:
            models = client.list_models()
            say(f"  ✅ 可用模型：{models[:8]}")
            if cfg.llm_model not in models:
                problems.append(f"配置的模型 '{cfg.llm_model}' 不在账号可用列表里 → "
                                f"用 --model 或 CAMPUS_RAG_MODEL 改成上面列表中的一个。")
                say(f"  ⚠ 但配置的 '{cfg.llm_model}' 不在列表里！")
        except LLMError as e:
            say(f"  ❌ 失败：{e}")
            problems.append(f"GET /models 失败：{e}")

        say("\n【5/6】POST /chat/completions（证明能拿到答案）")
        try:
            res = client.chat("你是一个测试助手。", "只回复两个字：可用")
            say(f"  ✅ {res.latency_ms} ms，模型 {res.model}")
            say(f"  finish_reason={res.finish_reason or '未提供'} usage={res.usage}")
            say(f"  正文 {len(res.text)} 字｜思维链 {res.reasoning_chars} 字｜预览：{res.text[:60]}")
        except LLMError as e:
            say(f"  ❌ 失败：{e}")
            problems.append(f"对话调用失败：{e}")

    # ---- 6. 本机有没有旧的 demo 进程占着端口 ----
    say("\n【6/6】本机 demo 服务端口")
    alive = _listening_ports()
    if alive:
        say(f"  8000-8019 上正在监听：{alive}")
        say("  ⚠ 如果你浏览器里开的是旧端口，那是**旧进程**（旧代码），")
        say("     刷新页面也不会变。关掉所有黑窗口重新起一次最省事。")
    else:
        say("  8000-8019 上没有任何服务在跑（说明你现在没起服务，或已经关干净了）")

    # ---- 结论 ----
    say("\n" + "=" * 68)
    if problems:
        say("  ❌ 结论：大模型这一层**没通**。原因按顺序如下：")
        for i, p in enumerate(problems, 1):
            say(f"     {i}. {p}")
        say("\n  修完之后重跑一次本命令，直到这里变成 ✅。")
        rc = 1
    else:
        say("  ✅ 结论：大模型这一层是通的（key 有效、能拿到答案）。")
        say("     如果页面上仍显示「离线抽取式降级」，按顺序检查：")
        say("       1. 服务进程是不是旧的？改了代码必须重启 python ask.py --web")
        say("       2. 浏览器 Ctrl+F5 强刷（旧页面不会带 use_llm 参数，也不会显示开关）")
        say("       3. 输入框上方的「使用大模型」开关是不是「关」？点一下")
        rc = 0
    say("=" * 68)

    try:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n报告已写入：{REPORT}")
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
