# paper_polish — Academic Paragraph Polisher Based on Professor Papers

A lightweight RAG-based tool: it extracts paragraphs from the PDF papers of Prof. Yunmi Park, builds a vector index, and retrieves the most similar passages from those papers as "style references" each time you polish a paragraph — sending them together to Claude so it can rewrite your text in the academic writing style of these two professors.

> Note: This is not genuine "fine-tuning." Real model training requires GPUs and tens of gigabytes of data, which would be neither cost-effective nor appropriate for just a handful of papers. RAG is a better trade-off between quality and cost.

## 1. Installation

```bash
pip install -r requirements.txt
```

The first time you run `build-index`, `sentence-transformers` will automatically download the local embedding model (`all-MiniLM-L6-v2`, ~80 MB).

## 2. Set Your API Key

Get a key at <https://console.anthropic.com/>, then set it in your terminal:

Windows PowerShell:
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-xxxxxxxx"
```

macOS / Linux bash:
```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

## 3. Step 1: Build the Index (once only; re-run when adding new PDFs)

```bash
python paper_polish.py build-index
```

This generates a `.style_index.pkl` file in the same directory as the script.

## 4. Step 2: Polish a Paragraph

Choose any one of three input methods:

```bash
# Option A: pass text directly
python paper_polish.py polish --text "Your English paragraph here."

# Option B: read from a file
python paper_polish.py polish --input my_draft.txt --output polished.txt

# Option C: read from stdin (paste your paragraph, then press Ctrl-Z / Ctrl-D to finish)
python paper_polish.py polish
```

Optional flags:
- `--top-k 5`  — how many professor passages to retrieve as style references (default: 5)
- `--tone academic`  — polishing tone; can be changed to `formal`, `concise`, etc.
- `--model claude-sonnet-4-5`  — specify the Claude model

## 5. Output Format

Claude returns two sections:
```
<polished>
(the polished English paragraph)
</polished>
<notes>
- A bullet-point list of the main changes made
</notes>
```

If `--output xxx.txt` is specified, the file will also include metadata for the retrieved reference passages (professor name, paper title, similarity score) for your reference.

## 6. Workflow Overview

```
PDF ─► pypdf text extraction ─► cleaning + chunking ─► sentence-transformers embedding ─► .style_index.pkl
                                                                                           │
Your paragraph ──► same model embedding ──► cosine similarity retrieval (top-k) ──────────┘
                                                              │
                                                              ▼
                                               Claude API (style references + your paragraph)
                                                              │
                                                              ▼
                                                   Polished English paragraph
```

## 7. FAQ

**Q: Error — "missing pypdf / sentence-transformers / anthropic"**
A: Run `pip install -r requirements.txt`.

**Q: No text extracted from a PDF?**
A: Some scanned PDFs are image-only and require OCR (e.g., `pytesseract`). The script will skip these automatically for now.

**Q: Want to add more professors' papers?**
A: Create a new folder under `Training_dataset/` (e.g., `Prof. XYZ`), place the PDFs inside, then run:
```bash
python paper_polish.py build-index --professors "Prof. Filip Biljecki" "Prof. Yunmi Park" "Prof. XYZ"
```
