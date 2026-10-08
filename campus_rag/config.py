"""全局配置：所有可调参数集中一处，便于在 eval 里做消融实验。"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus"
DEFAULT_CACHE = PROJECT_ROOT / ".cache"


def _env_flag(name: str) -> bool:
    """把 "1 / true / yes / on" 这类环境变量当作布尔开关（大小写不敏感）。"""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def offline_requested() -> bool:
    """环境变量这一路是否要求"不调用任何网络接口"（与命令行 --offline 同义）。

    单独抽出来是因为它必须和 `--offline` 被**同源判断**：
    原先 from_env 认 CAMPUS_RAG_OFFLINE、而 cmd_info 只认 `--offline`，
    于是 `CAMPUS_RAG_OFFLINE=1 python ask.py --info` 照样去调 /models——
    一个"强制离线"的开关，在自检那条路径上被绕过去了。
    """
    return _env_flag("CAMPUS_RAG_OFFLINE")


def llm_disabled_reason(cfg: "Config", offline: bool = False) -> str:
    """用一句人话解释"大模型为什么没启用"，没启用时返回空串。

    为什么要有这么个函数：这个坑被踩了不止一次——**"配了 key"≠"启用了"**。
    而且判断口径必须只有一处，否则又会像 bug 6 那样两处各说各话：
    CLI 的 --info、启动横幅、Web 页面的徽章，全都调它。

    注意推导链：use_llm=False 且 key 存在、又不在离线状态 ⟹ 就是没加 --llm
    （因为 build_engine 里 --llm 是 or 短路的第一项，加了就一定是 True）。
    """
    if cfg.use_llm:
        return ""
    if offline:
        return "强制离线"
    if not cfg.llm_api_key:
        return "未配置 API key"
    return "未加 --llm"


@dataclass
class Config:
    # --- 分块 ---
    chunk_size: int = 480
    chunk_overlap: int = 80
    min_chunk_len: int = 40
    # --- 检索 ---
    top_k: int = 4
    candidate_k: int = 24
    rrf_k: int = 60
    # --- 生成 ---
    max_context_chunks: int = 4
    max_tokens: int = 700
    temperature: float = 0.2
    # --- 拒答闸门 ---
    # 低于该置信度则拒答（"我不确定"比胡说八道更有价值）。
    # 0.42 / evidence_k=5 由 sweep.py 在 22 题评测集上网格搜索得到：
    #   硬答率 0.0%、拒答准确率 81.8%、漏答率 25.0%、关键事实覆盖率 82.3%
    # 取舍原则：**先保证硬答率为 0（幻觉是这类系统最不可接受的错误）**，
    # 再在满足该前提的参数里最大化拒答准确率与覆盖率。
    # 现场若想更保守，调高即可——宁可拒答也不编。
    answer_threshold: float = 0.42
    # 置信度评估时纳入的证据块数量：生成只用 top_k 块，但"是否该拒答"应看更宽的候选集，
    # 否则多跳问题会因为证据分散在 5-6 名而被误判为无证据。
    evidence_k: int = 5
    # 极稀有词惩罚强度与判定阈值（见 engine._rare_idf_share）
    rare_penalty: float = 0.30
    rare_doc_ratio: float = 0.12
    rare_doc_min: int = 2
    # --- LLM（可选，OpenAI 兼容接口） ---
    # 注意：DeepSeek 已下线 devseek-chat / deepseek-reasoner 这类旧模型名，
    # 现役为 deepseek-flash（便宜、快）与 deepseek-v4-pro（强）。模型名务必走配置，不要硬编码。
    llm_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-flash"
    llm_api_key: str = ""
    llm_timeout: float = 60.0
    use_llm: bool = True
    # 思维链（thinking）开关：enabled / disabled / omit。
    # **默认必须关**，这是被一个真实故障逼出来的：DeepSeek V4 的 thinking 默认**开启**
    # 且 effort 默认 high，思维链走 reasoning_content、答案走 content。
    # 本项目 max_tokens=700 是给"结论 + 依据"留的，思维链一开就会被它吃光，
    # 于是 content 返回空串 → 用户看到"LLM 返回空内容"。
    # 另外 thinking 模式下 temperature 是被**静默忽略**的（官方文档明说）——
    # 那就等于本项目刻意设的 temperature=0.2 从来没生效过。
    # omit：完全不发这个字段，给不认 DeepSeek 扩展字段的第三方网关用。
    llm_thinking: str = "disabled"
    # --- 语义向量通道（可选） ---
    embed_base_url: str = "https://api.siliconflow.cn/v1"
    embed_model: str = "BAAI/bge-m3"
    embed_api_key: str = ""
    use_embeddings: bool = True
    # --- 评测 ---
    eval_k: int = 3
    # --- 路径 ---
    corpus_dir: str = str(DEFAULT_CORPUS)
    cache_dir: str = str(DEFAULT_CACHE)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> "Config":
        """环境变量 > 默认值；函数参数 > 环境变量；**离线开关 > 以上全部**。

        离线是唯一的例外，必须最后生效：它是"不许联网"的安全闸，不是普通配置项。
        原先它写在 overrides 之前，于是 `CAMPUS_RAG_OFFLINE=1 ... --llm` 会被 --llm 翻掉
        ——而命令行 `--offline` 却能压住 `--llm`。同一个意图的两种写法结果不同，
        等于环境变量那一路是个假开关（断网/省流量的场景下会直接联网）。
        """
        env_map = {
            "llm_base_url": ("CAMPUS_RAG_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
            "llm_model": ("CAMPUS_RAG_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL"),
            "llm_api_key": ("CAMPUS_RAG_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
            # 思维链开关也走环境变量：报错信息里会建议这一条，就必须真的能被设置
            "llm_thinking": ("CAMPUS_RAG_THINKING", "DEEPSEEK_THINKING"),
            "embed_base_url": ("CAMPUS_RAG_EMBED_BASE_URL", "SILICONFLOW_BASE_URL"),
            "embed_model": ("CAMPUS_RAG_EMBED_MODEL",),
            "embed_api_key": ("CAMPUS_RAG_EMBED_API_KEY", "SILICONFLOW_API_KEY"),
            "corpus_dir": ("CAMPUS_RAG_CORPUS",),
        }
        values: Dict[str, Any] = {}
        for field_name, env_names in env_map.items():
            for name in env_names:
                if os.environ.get(name):
                    values[field_name] = os.environ[name]
                    break
        values.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**values)
        # 离线闸最后落：任何来源（命令行 --llm / 显式 override）都不许把它翻回去
        if offline_requested():
            cfg.use_llm = False
            cfg.use_embeddings = False
        # 没配 key 就别去调接口，直接走降级路径（面试现场少一个报错源）
        if not cfg.llm_api_key:
            cfg.use_llm = False
        if not cfg.embed_api_key:
            cfg.use_embeddings = False
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d.get("llm_api_key"):
            d["llm_api_key"] = "***"
        if d.get("embed_api_key"):
            d["embed_api_key"] = "***"
        return d
