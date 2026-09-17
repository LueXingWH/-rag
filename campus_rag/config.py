"""全局配置：所有可调参数集中一处，便于在 eval 里做消融实验。"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "corpus"
DEFAULT_CACHE = PROJECT_ROOT / ".cache"


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
        """环境变量 > 默认值；函数参数 > 环境变量。不用 dotenv，少一个依赖。"""
        env_map = {
            "llm_base_url": ("CAMPUS_RAG_BASE_URL", "DEEPSEEK_BASE_URL", "OPENAI_BASE_URL"),
            "llm_model": ("CAMPUS_RAG_MODEL", "DEEPSEEK_MODEL", "OPENAI_MODEL"),
            "llm_api_key": ("CAMPUS_RAG_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
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
        if os.environ.get("CAMPUS_RAG_OFFLINE", "").lower() in {"1", "true", "yes"}:
            values["use_llm"] = False
            values["use_embeddings"] = False
        values.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**values)
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
