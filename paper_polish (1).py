"""
paper_polish.py
================

A RAG (Retrieval-Augmented Generation) tool for polishing academic English paragraphs.

Workflow:
1. build-index  : Scans PDF papers from two professors (Prof. Filip Biljecki,
                  Prof. Yunmi Park) under the Training_dataset directory, extracts
                  paragraphs, generates embeddings with sentence-transformers,
                  and saves a local index.
2. polish       : Reads the user's English paragraph, retrieves the most similar
                  reference excerpts from the index, and sends them to the Claude API
                  as "style references" so Claude can polish the paragraph in the
                  academic writing style of these professors.

Dependencies (see requirements.txt):
    pip install anthropic pypdf sentence-transformers numpy tqdm

Environment variable:
    ANTHROPIC_API_KEY   Your Anthropic API key (https://console.anthropic.com/)

Usage examples:
    # 1) Build the index first (scans the two professor folders in the script directory by default)
    python paper_polish.py build-index

    # 2) Polish a paragraph passed directly on the command line
    python paper_polish.py polish --text "Your English paragraph here."

    # 3) Polish a paragraph from a file
    python paper_polish.py polish --input my_paragraph.txt --output polished.txt

    # 4) Custom parameters
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
# Default path configuration (relative to the script directory; overridable via CLI)
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_DIR = SCRIPT_DIR           # Default: same directory as script
DEFAULT_PROFESSOR_DIRS = [
    "Prof. Filip Biljecki",
    "Prof. Yunmi Park",
]
DEFAULT_INDEX_PATH = SCRIPT_DIR / ".style_index.pkl"

# Default local embedding model (lightweight, ~80 MB, good for academic English text)
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Default Claude model
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Chunk:
    """A text chunk extracted from a PDF."""
    text: str
    source_file: str       # PDF filename (without path)
    professor: str         # Name of the professor's folder
    chunk_id: int          # Sequential index within the file


# --------------------------------------------------------------------------- #
# 1. PDF parsing and chunking
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise SystemExit(
            "Missing pypdf. Please install it first: pip install pypdf"
        ) from e

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        print(f"  ! Cannot open {pdf_path.name}: {e}", file=sys.stderr)
        return ""

    pages_text = []
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            # Skip pages that fail to parse
            continue
    return "\n".join(pages_text)


_HEADER_PATTERNS = [
    r"^\s*\d+\s*$",                       # Standalone page numbers
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",  # "Page 3 of 12"
    r"^\s*https?://\S+\s*$",              # Standalone URLs on their own line
    r"^\s*doi[:\s]\S+\s*$",               # DOI lines
]
_HEADER_RE = re.compile("|".join(_HEADER_PATTERNS), re.IGNORECASE)


def clean_text(raw: str) -> str:
    """Basic cleaning: merge line breaks, remove headers/footers, truncate references."""
    # Remove lines that look like headers or footers
    lines = []
    for line in raw.splitlines():
        if _HEADER_RE.match(line):
            continue
        lines.append(line)
    text = "\n".join(lines)

    # Handle English end-of-line hyphenation: e.g. "develop-\nment" -> "development"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Convert single newlines (mid-paragraph line breaks) to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Truncate the references section (typically at the end of a paper;
    # not suitable as style reference material)
    cut = re.search(r"\n\s*(References|REFERENCES|Bibliography)\s*\n", text)
    if cut:
        text = text[: cut.start()]

    return text.strip()


def chunk_paragraphs(text: str, min_words: int = 40, max_words: int = 220) -> List[str]:
    """Split the full text into semantic paragraph chunks.

    Strategy: split on double newlines to get rough paragraphs, then merge
    short ones and split long ones. Target size: ~40–220 words per chunk,
    suitable for use as style references.
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

        # If a paragraph is too long, split it roughly at sentence boundaries
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

        # Normal paragraphs: accumulate in buffer until min_words is reached
        buf.append(para)
        buf_words += wc
        if buf_words >= min_words:
            flush()

    flush()
    # Retain the final short chunk if any; filter out chunks that are too short
    chunks = [c for c in chunks if len(c.split()) >= 15]
    return chunks


# --------------------------------------------------------------------------- #
# 2. Embedding & index storage
# --------------------------------------------------------------------------- #
def load_embedder(model_name: str):
    """Lazily load a sentence-transformers model (downloads on first use)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "Missing sentence-transformers. Please install it first:\n"
            "    pip install sentence-transformers"
        ) from e

    print(f"[i] Loading embedding model: {model_name} (will auto-download ~80 MB on first run)")
    return SentenceTransformer(model_name)


def build_index(
    dataset_dir: Path,
    professor_dirs: List[str],
    index_path: Path,
    embed_model: str,
) -> None:
    """Scan all PDFs -> extract text -> chunk -> embed -> save index."""
    import numpy as np
    from tqdm import tqdm

    all_chunks: List[Chunk] = []
    for prof in professor_dirs:
        folder = dataset_dir / prof
        if not folder.exists():
            print(f"[!] Folder not found: {folder}, skipping")
            continue
        pdfs = sorted(folder.glob("*.pdf"))
        if not pdfs:
            print(f"[!] No PDFs found in {folder}, skipping")
            continue
        print(f"\n[i] Processing {prof} ({len(pdfs)} PDF(s))")
        for pdf in pdfs:
            print(f"    - {pdf.name}")
            raw = extract_text_from_pdf(pdf)
            if not raw.strip():
                print(f"      (Empty or unreadable, skipping)")
                continue
            cleaned = clean_text(raw)
            chunks = chunk_paragraphs(cleaned)
            for i, c in enumerate(chunks):
                all_chunks.append(
                    Chunk(text=c, source_file=pdf.name, professor=prof, chunk_id=i)
                )
            print(f"      Extracted {len(chunks)} chunk(s)")

    if not all_chunks:
        raise SystemExit("[x] No text chunks extracted. Index was not created.")

    print(f"\n[i] Total chunks: {len(all_chunks)}. Starting vectorization...")

    embedder = load_embedder(embed_model)
    texts = [c.text for c in all_chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # Normalized for cosine similarity via dot product
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
        f"\n[OK] Index saved: {index_path}  "
        f"({len(all_chunks)} chunks, {size_mb:.1f} MB)"
    )


def load_index(index_path: Path) -> dict:
    if not index_path.exists():
        raise SystemExit(
            f"[x] Index file not found: {index_path}\n"
            f"    Please run first: python {Path(__file__).name} build-index"
        )
    with open(index_path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# 3. Retrieval
# --------------------------------------------------------------------------- #
def retrieve(
    query: str,
    index: dict,
    top_k: int = 5,
):
    """Embed the query and retrieve the top_k most similar chunks from the index."""
    import numpy as np

    embedder = load_embedder(index["embed_model"])
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)[0]

    embs = index["embeddings"]  # (N, D), already normalized
    sims = embs @ q             # Cosine similarity via dot product
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
# 4. Claude API call for polishing
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
<notes> ... </notes> tags (max 5 bullets).
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
            "Missing anthropic. Please install it first: pip install anthropic"
        ) from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "[x] Environment variable ANTHROPIC_API_KEY not found.\n"
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
    # Concatenate all text blocks
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
    # Read the draft
    if args.text:
        draft = args.text
    elif args.input:
        draft = Path(args.input).read_text(encoding="utf-8")
    else:
        print("[i] No --text or --input provided. Reading paragraph from stdin (Ctrl-D / Ctrl-Z to finish):")
        draft = sys.stdin.read()

    draft = draft.strip()
    if not draft:
        raise SystemExit("[x] Input paragraph is empty.")

    print("\n[i] Loading index ...")
    index = load_index(Path(args.index))
    print(f"[i] Index contains {len(index['chunks'])} chunks")

    print(f"[i] Retrieving top-{args.top_k} style references ...")
    refs = retrieve(draft, index, top_k=args.top_k)
    for i, r in enumerate(refs, 1):
        print(
            f"    Ref{i}  sim={r['score']:.3f}  "
            f"{r['professor']} / {r['source_file']}"
        )

    print(f"\n[i] Calling Claude ({args.model}) for polishing ...")
    output = call_claude(
        draft=draft,
        references=refs,
        tone=args.tone,
        model=args.model,
    )

    # Write retrieval metadata + model output to file if requested
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
        print(f"\n[OK] Output written to: {out_path}")

    print("\n========== Polished Result ==========\n")
    print(output)
    print("\n=====================================\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="paper_polish",
        description="RAG-based academic English polishing tool using professors' papers as style references",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-index
    p1 = sub.add_parser("build-index", help="Scan PDFs and build a vector index")
    p1.add_argument(
        "--dataset-dir",
        default=str(DEFAULT_DATASET_DIR),
        help="Root directory of the dataset (default: same directory as this script)",
    )
    p1.add_argument(
        "--professors",
        nargs="+",
        default=DEFAULT_PROFESSOR_DIRS,
        help="Professor subfolder names (default: Prof. Filip Biljecki / Prof. Yunmi Park)",
    )
    p1.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help=f"Output path for the index (default: {DEFAULT_INDEX_PATH.name})",
    )
    p1.add_argument(
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help=f"sentence-transformers model name (default: {DEFAULT_EMBED_MODEL})",
    )
    p1.set_defaults(func=cmd_build_index)

    # polish
    p2 = sub.add_parser("polish", help="Polish an English paragraph in the professors' style")
    src = p2.add_mutually_exclusive_group()
    src.add_argument("--text", help="English paragraph to polish, passed directly")
    src.add_argument("--input", help="Read the paragraph from a file (UTF-8)")
    p2.add_argument("--output", help="Write the result to this file (UTF-8)")
    p2.add_argument(
        "--index",
        default=str(DEFAULT_INDEX_PATH),
        help=f"Path to the index file (default: {DEFAULT_INDEX_PATH.name})",
    )
    p2.add_argument("--top-k", type=int, default=5, help="Number of reference chunks to retrieve (default: 5)")
    p2.add_argument(
        "--tone",
        default="academic",
        help="Target tone, e.g. academic / formal / concise (default: academic)",
    )
    p2.add_argument(
        "--model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Claude model to use (default: {DEFAULT_CLAUDE_MODEL})",
    )
    p2.set_defaults(func=cmd_polish)

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
