"""
paper_polish.py
================

一个基于 RAG (检索增强生成) 的英文学术段落润色工具。

工作流程:
1. build-index  : 扫描 Training_dataset 下两位教授 (Prof. Filip Biljecki,
                  Prof. Yunmi Park) 的 PDF 论文, 抽取段落, 用
                  sentence-transformers 生成向量, 保存为本地索引。
2. polish       : 读入用户的英文段落, 在索引中检索最相似的若干段教授原文,
                  把它们作为 "风格参考" 一并发送给 Claude API, 让 Claude
                  按这些教授的学术写作风格对段落进行润色。

依赖 (见 requirements.txt):
    pip install anthropic pypdf sentence-transformers numpy tqdm

环境变量:
    ANTHROPIC_API_KEY   你的 Anthropic API Key (https://console.anthropic.com/)

用法示例:
    # 1) 第一次先建索引 (默认会扫描脚本同目录下两个教授文件夹)
    python paper_polish.py build-index

    # 2) 润色一段文字 (从命令行直接传)
    python paper_polish.py polish --text "Your English paragraph here."

    # 3) 润色一段文字 (从文件读)
    python paper_polish.py polish --input my_paragraph.txt --output polished.txt

    # 4) 自定义参数
    python paper_polish.py polish --text "..." --top-k 8 --tone academic
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# 默认路径配置 (相对脚本所在目录, 可被命令行参数覆盖)
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_DIR = SCRIPT_DIR           # 默认: 脚本同目录
DEFAULT_PROFESSOR_DIRS = [
    "Prof. Filip Biljecki",
    "Prof. Yunmi Park",
]
DEFAULT_INDEX_PATH = SCRIPT_DIR / ".style_index.pkl"

# 默认使用的本地 embedding 模型 (轻量, ~80MB, 英文学术文本表现很好)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 默认使用的 Claude 模型
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """一段从 PDF 中抽取出来的文本块。"""
    text: str
    source_file: str       # PDF 文件名 (不含路径)
    professor: str         # 所属教授文件夹名
    chunk_id: int          # 在该文件中的序号


# --------------------------------------------------------------------------- #
# 1. PDF 解析与分块
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(pdf_path: Path) -> str:
    """用 pypdf 抽取一份 PDF 的全部文本。"""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SystemExit(
            "缺少 pypdf, 请先运行: pip install pypdf"
        ) from e

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ! 无法打开 {pdf_path.name}: {e}", file=sys.stderr)
        return ""

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # 个别页解析失败就跳过
            continue
    return "\n".join(pages_text)


_HEADER_PATTERNS = [
    r"^\s*\d+\s*$",                       # 单独的页码
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",  # "Page 3 of 12"
    r"^\s*https?://\S+\s*$",              # 单独一行的 URL
    r"^\s*doi[:\s]\S+\s*$",               # DOI 行
]
_HEADER_RE = re.compile("|".join(_HEADER_PATTERNS), re.IGNORECASE)


def clean_text(raw: str) -> str:
    """简单清洗: 合并换行、去掉页眉页脚、去掉参考文献后续段。"""
    # 删掉明显是页眉/页脚的整行
    lines = []
    for line in raw.splitlines():
        if _HEADER_RE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # 处理英文常见的"行尾连字符"换行: e.g. "develop-\nment" -> "development"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # 单换行 (段内换行) 改成空格
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 多空格折叠
    text = re.sub(r"[ \t]+", " ", text)

    # 截断参考文献部分 (通常在论文末尾, 不适合作为正文风格样本)
    cut = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if cut:
        text = text[: cut.start()]

    return text.strip()


def chunk_paragraphs(text: str, min_words: int = 40, max_words: int = 220) -> List[str]:
    """把整篇文本切成"语义段落"。

    策略: 先按两个换行切粗段, 再合并过短的段、拆分过长的段。
    目标是每块大约 40-220 词, 适合作为风格参考。
    """
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_words = 0

    def flush():
        nonlocal buf, buf_words
        if buf:
            chunks.append(" ".join(buf).strip())
            buf = []
            buf_words = 0

    for para in raw_paras:
        words = para.split()
        wc = len(words)

        # 过长 -> 按句号粗略切, 然后塞进 buf
        if wc > max_words:
            flush()
            sentences = re.split(r"(?<=[\.\?\!])\s+", para)
            sub_buf: List[str] = []
            sub_wc = 0
            for sent in sentences:
                sw = len(sent.split())
                if sub_wc + sw > max_words and sub_buf:
                    chunks.append(" ".join(sub_buf).strip())
                    sub_buf = [sent]
                    sub_wc = sw
                else:
                    sub_buf.append(sent)
                    sub_wc += sw
            if sub_buf:
                chunks.append(" ".join(sub_buf).strip())
            continue

        # 普通段落: 累积到 buf, 直到达到 min_words 才输出
        buf.append(para)
        buf_words += wc
        if buf_words >= min_words:
            flush()

    flush()
    # 过短的最后一块也保留 (反正只剩这些了), 但显式过滤完全没意义的
    chunks = [c for c in chunks if len(c.split()) >= 15]
    return chunks


# --------------------------------------------------------------------------- #
# 2. 向量化 & 索引存储
# --------------------------------------------------------------------------- #
def load_embedder(model_name: str):
    """惰性加载 sentence-transformers (第一次会下载模型)。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "缺少 sentence-transformers, 请先运行:\n"
            "    pip install sentence-transformers"
        ) from e

    print(f"[i] 加载 embedding 模型: {model_name} (首次会自动下载, 约80MB)")
    return SentenceTransformer(model_name)


def build_index(
    dataset_dir: Path,
    professor_dirs: List[str],
    index_path: Path,
    embed_model: str,
) -> None:
    """扫描所有 PDF -> 抽取 -> 切块 -> embedding -> 保存索引。"""
    import numpy as np
    from tqdm import tqdm

    all_chunks: List[Chunk] = []
    for prof in professor_dirs:
        folder = dataset_dir / prof
        if not folder.exists():
            print(f"[!] 找不到文件夹: {folder}, 跳过")
            continue
        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            print(f"[!] {folder} 下没有 PDF, 跳过")
            continue
        print(f"\n[i] 处理 {prof} ({len(pdfs)} 个 PDF)")
        for pdf in pdfs:
            print(f"    - {pdf.name}")
            raw = extract_text_from_pdf(pdf)
            if not raw.strip():
                print(f"      (空白或无法解析, 跳过)")
                continue
            cleaned = clean_text(raw)
            chunks = chunk_paragraphs(cleaned)
            for i, c in enumerate(chunks):
                all_chunks.append(
                    Chunk(text=c, source_file=pdf.name, professor=prof, chunk_id=i)
                )
            print(f"      切出 {len(chunks)} 块")

    if not all_chunks:
        raise SystemExit("[x] 没有抽到任何文本块, 索引未生成。")

    print(f"\n[i] 共得到 {len(all_chunks)} 个文本块, 开始向量化...")

    embedder = load_embedder(embed_model)
    texts = [c.text for c in all_chunks]
    # show_progress_bar 由 sentence-transformers 内部控制
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # 归一化, 余弦相似度 = 点积
    ).astype(np.float32)

    index = {
        "embed_model": embed_model,
        "chunks": [asdict(c) for c in all_chunks],
        "embeddings": embeddings,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump(index, f)
    size_mb = index_path.stat().st_size / 1024 / 1024
    print(
        f"\n[OK] 索引已保存: {index_path}  "
        f"({len(all_chunks)} 块, {size_mb:.1f} MB)"
    )


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit(
            f"[x] 找不到索引文件: {index_path}\n"
            f"    请先运行: python {Path(__file__).name} build-index"
        )
    with open(index_path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# 3. 检索
# --------------------------------------------------------------------------- #
def retrieve(
    query: str,
    index: dict,
    top_k: int = 5,
):
    """对 query 做 embedding, 在索引中找相似度最高的 top_k 块。"""
    import numpy as np

    embedder = load_embedder(index["embed_model"])
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]

    embs = index["embeddings"]  # (N, D), 已归一化
    sims = embs @ q             # 余弦相似度
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in top_idx:
        c = index["chunks"][int(i)]
        results.append({
            "score": float(sims[i]),
            "professor": c["professor"],
            "source_file": c["source_file"],
            "text": c["text"],
        })
    return results


# --------------------------------------------------------------------------- #
# 4. 调用 Claude 进行润色
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are an expert academic English editor specializing in \
urban analytics, GIScience, urban planning, and remote sensing — the fields of \
Prof. Filip Biljecki and Prof. Yunmi Park.

You will be given:
1. A paragraph the user wrote (their DRAFT).
2. A small set of reference excerpts taken from these two professors' published \
papers. Treat them ONLY as a STYLE REFERENCE — do not copy their content, do not \
introduce their findings or citations into the user's draft.

Your job:
- Rewrite the DRAFT in fluent, precise, formal academic English that matches the \
register, rhythm, vocabulary, and connective style of the reference excerpts.
- Preserve the user's original meaning, claims, and structure. Do NOT add new facts.
- Improve clarity, conciseness, grammar, word choice, and academic tone.
- Use British or American English consistently with the draft (default: American).
- Avoid overly flowery language, hype words ("very", "really", "huge"), and \
ChatGPT-style cliches ("delve", "in the realm of", "tapestry", etc.).
- Keep technical terms intact.

Output format (strict):
First, output the polished paragraph between <polished> ... </polished> tags.
Then, output a short bullet list of the key edits you made between \
<notes> ... </notes> tags (max 5 bullets, in Chinese is fine).
Output nothing else.
"""


def build_user_prompt(draft: str, references: list, tone: str) -> str:
    refs_block = []
    for i, r in enumerate(references, 1):
        refs_block.append(
            f"[Ref {i}] (from {r['professor']} — {r['source_file']}, "
            f"similarity={r['score']:.3f})\n{r['text']}"
        )
    refs_text = "\n\n".join(refs_block)

    return (
        f"### STYLE REFERENCES (do not copy content, only mimic style)\n\n"
        f"{refs_text}\n\n"
        f"### USER DRAFT (to be polished)\n\n{draft.strip()}\n\n"
        f"### INSTRUCTIONS\n"
        f"- Target tone: {tone}\n"
        f"- Match the academic register of the references.\n"
        f"- Return <polished>...</polished> and <notes>...</notes> only."
    )


def call_claude(
    draft: str,
    references: list,
    tone: str,
    model: str,
    max_tokens: int = 2000,
) -> str:
    try:
        from anthropic import Anthropic
    except ImportError as e:
        raise SystemExit(
            "缺少 anthropic, 请先运行: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "[x] 没找到环境变量 ANTHROPIC_API_KEY。\n"
            "    Windows PowerShell:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "    macOS/Linux bash:    export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = Anthropic(api_key=api_key)
    user_prompt = build_user_prompt(draft, references, tone)

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # 拼接所有 text block
    return "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )


# --------------------------------------------------------------------------- #
# 5. CLI
# --------------------------------------------------------------------------- #
def cmd_build_index(args: argparse.Namespace) -> None:
    build_index(
        dataset_dir=Path(args.dataset_dir),
        professor_dirs=args.professors,
        index_path=Path(args.index),
        embed_model=args.embed_model,
    )


def cmd_polish(args: argparse.Namespace) -> None:
    # 读取 draft
    if args.text:
        draft = args.text
    elif args.input:
        draft = Path(args.input).read_text(encoding="utf-8")
    else:
        print("[i] 没传 --text/--input, 从 stdin 读取段落 (Ctrl-D / Ctrl-Z 结束):")
        draft = sys.stdin.read()

    draft = draft.strip()
    if not draft:
        raise SystemExit("[x] 输入段落为空。")

    print("\n[i] 加载索引 ...")
    index = load_index(Path(args.index))
    print(f"[i] 索引共 {len(index['chunks'])} 块")

    print(f"[i] 检索 top-{args.top_k} 风格参考 ...")
    refs = retrieve(draft, index, top_k=args.top_k)
    for i, r in enumerate(refs, 1):
        print(
            f"    Ref{i}  sim={r['score']:.3f}  "
            f"{r['professor']} / {r['source_file']}"
        )

    print(f"\n[i] 调用 Claude ({args.model}) 进行润色 ...")
    output = call_claude(
        draft=draft,
        references=refs,
        tone=args.tone,
        model=args.model,
    )

    # 把检索元数据 + 模型输出一并写到文件
    if args.output:
        out_path = Path(args.output)
        meta_refs = [
            {
                "rank": i,
                "score": r["score"],
                "professor": r["professor"],
                "source_file": r["source_file"],
            }
            for i, r in enumerate(refs, 1)
        ]
        out_path.write_text(
            "=== Polished Output (Claude) ===\n\n"
            + output
            + "\n\n=== Retrieved Style References (metadata) ===\n"
            + json.dumps(meta_refs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\n[OK] 已写入: {out_path}")

    print("\n========== 润色结果 ==========\n")
    print(output)
    print("\n==============================\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_polish",
        description="基于教授论文的 RAG 学术英文润色工具",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-index
    p1 = sub.add_parser("build-index", help="扫描 PDF 建立向量索引")
    p1.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help=f"数据集根目录 (默认: 脚本同目录)",
    )
    p1.add_argument(
        "--professors",
        nargs="+",
        default=DEFAULT_PROFESSOR_DIRS,
        help="教授子文件夹名 (默认即 Prof. Filip Biljecki / Prof. Yunmi Park)",
    )
    p1.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help=f"索引输出路径 (默认: {DEFAULT_INDEX_PATH.name})",
    )
    p1.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"sentence-transformers 模型名 (默认: {DEFAULT_EMBED_MODEL})",
    )
    p1.set_defaults(func=cmd_build_index)

    # polish
    p2 = sub.add_parser("polish", help="对一段英文进行风格润色")
    src = p2.add_mutually_exclusive_group()
    src.add_argument("--text", help="直接传入要润色的英文段落")
    src.add_argument("--input", help="从文件读入英文段落 (UTF-8)")
    p2.add_argument("--output", help="把结果写到这个文件 (UTF-8)")
    p2.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help=f"索引路径 (默认: {DEFAULT_INDEX_PATH.name})",
    )
    p2.add_argument("--top-k", type=int, default=5, help="检索多少段参考 (默认 5)")
    p2.add_argument(
        "--tone",
        default="academic",
        help="润色语气, 例如 academic / formal / concise (默认 academic)",
    )
    p2.add_argument(
        "--model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude 模型 (默认: {DEFAULT_CLAUDE_MODEL})",
    )
    p2.set_defaults(func=cmd_polish)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
