"""文档加载与分块（chunking）。

分块策略（面试常被追问的点）：
1. 先按 Markdown 标题切"语义块"——标题是天然的主题边界，比定长切分语义更完整。
2. 块过大再按句子切，并保留 overlap，避免答案被切在两个块中间。
3. 每块都带 heading 路径（如 "推免 > 成绩要求"），既用于展示引用，
   也用于检索时的标题加成（标题命中 = 强信号）。

尺寸经验值：中文 300~600 字/块是常见甜点区。太小 → 上下文不足；
太大 → 向量/词频被稀释，召回反而下降（可在 eval 里调 chunk_size 验证）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# 中英文句子边界
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n{2,}")


@dataclass
class Chunk:
    chunk_id: str
    source: str          # 文件名
    title: str           # 文档一级标题
    heading: str         # 块所在的标题路径
    text: str
    position: int        # 块在文档中的序号
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "position": self.position,
        }
        d.update(self.meta)
        return d


def _is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.count("|") >= 2


def _split_table_separator_before(lines: List[str]) -> int:
    """返回"|-|-|"分隔行之前累计的表格行数（0 表示不是表格）。"""
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        if _is_table_row(s) and set(s) <= set("|-: "):
            return i
        if not _is_table_row(s):
            return 0
    return 0


def _coalesce_tables(lines: List[str]) -> List[str]:
    """把 Markdown 表格压成"表头 + 每个数据行"的文本行。

    这是被评测集逼出来的修正：原始表格行 "| 本科生 | 15 册 | 30 天 | 2 次 |"
    里没有"可借册数""借期"这些词——答案性关键词全在**表头**里。
    如果表头与数据行被切到不同的块，检索就会漏掉这类问题（实测 q08 先被误判为无证据）。
    修法：识别连续表格 → 把表头拼到数据行的文本里，让每一行自带语义。
    """
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        if not _is_table_row(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        block: List[str] = []
        while i < n and _is_table_row(lines[i]):
            block.append(lines[i])
            i += 1
        sep_at = _split_table_separator_before(block)
        if sep_at >= 1:
            header_cells = [c.strip() for c in block[0].strip().strip("|").split("|")]
            data_rows = block[sep_at + 1 :]
            out.append("表格表头：" + " | ".join(header_cells))
            for row in data_rows:
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                pairs = [
                    f"{h}：{c}" for h, c in zip(header_cells, cells) if h and c
                ]
                out.append("表格行：" + "，".join(pairs))
        else:
            out.extend(block)
    return out


def _split_long(text: str, max_len: int, overlap: int) -> List[str]:
    """按句子累积到 max_len，超长时带 overlap 滑窗。"""
    if len(text) <= max_len:
        return [text.strip()] if text.strip() else []
    sents = [s for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]
    if not sents:
        sents = [text]
    out: List[str] = []
    buf = ""
    for s in sents:
        if len(buf) + len(s) <= max_len:
            buf += s
            continue
        if buf.strip():
            out.append(buf.strip())
        # 用尾部 overlap 字符续接，保证跨块语义连续
        tail = buf[-overlap:] if overlap > 0 and buf else ""
        buf = (tail + s) if len(s) < max_len else ""
        if len(s) >= max_len:
            step = max(1, max_len - overlap)
            out.extend(s[i : i + max_len].strip() for i in range(0, len(s), step))
    if buf.strip():
        out.append(buf.strip())
    return [c for c in out if c]


def chunk_markdown(
    text: str, source: str, max_len: int = 480, overlap: int = 80, min_len: int = 40
) -> List[Chunk]:
    """把一篇 Markdown 文档切成带标题路径的块。"""
    doc_title = Path(source).stem
    heading_stack: List[str] = []
    blocks: List[tuple] = []  # (heading_path, content)
    buf: List[str] = []

    def flush() -> None:
        content = "\n".join(_coalesce_tables(buf)).strip()
        buf.clear()
        if content:
            blocks.append((" > ".join(heading_stack) or doc_title, content))

    for raw in text.splitlines():
        m = _HEADING_RE.match(raw.strip())
        if m:
            flush()
            level = len(m.group(1))
            name = m.group(2).strip()
            if level == 1 and not heading_stack:
                doc_title = name
            heading_stack = heading_stack[: level - 1]
            while len(heading_stack) < level - 1:
                heading_stack.append("")
            heading_stack.append(name)
        else:
            buf.append(raw)
    flush()

    chunks: List[Chunk] = []
    counter = 0
    for heading, content in blocks:
        for piece in _split_long(content, max_len, overlap):
            if len(piece) < min_len and chunks:
                # 太短的尾巴并入上一块，避免"碎片块"污染检索
                chunks[-1].text = (chunks[-1].text + "\n" + piece).strip()
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{source}#{counter}",
                    source=source,
                    title=doc_title,
                    heading=heading,
                    text=piece,
                    position=counter,
                )
            )
            counter += 1
    return chunks


def load_corpus(corpus_dir: str | Path, **kwargs: Any) -> List[Chunk]:
    """加载目录下所有 .md/.txt，返回块列表。"""
    root = Path(corpus_dir)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在: {root}")
    chunks: List[Chunk] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".md", ".markdown", ".txt"} or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="gbk", errors="ignore")
        chunks.extend(chunk_markdown(text, path.name, **kwargs))
    return chunks


def iter_documents(corpus_dir: str | Path) -> Iterable[Path]:
    root = Path(corpus_dir)
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".md", ".markdown", ".txt"} and path.is_file():
            yield path
