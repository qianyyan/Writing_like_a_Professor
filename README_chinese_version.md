# paper_polish — 基于教授论文的英文学术段落润色

一个用 RAG 思路实现的小工具：从 Prof. Yunmi Park 的论文 PDF 中抽取段落、建立向量索引，每次润色时检索最相似的若干段教授原文作为「风格参考」一并发给 Claude，让 Claude 按这两位教授的学术写作风格改写你的段落。

> 注意：这不是真正的「fine-tune」。真正训练大模型需要 GPU、几十 GB 数据，对你这两位教授的几篇论文来说既不划算也容易过拟合。RAG 在效果和成本之间是更合适的选择。

## 1. 安装

```bash
pip install -r requirements.txt
```

第一次跑 `build-index` 时，`sentence-transformers` 会自动下载约 80 MB 的本地 embedding 模型（`all-MiniLM-L6-v2`）。

## 2. 设置 API Key

去 <https://console.anthropic.com/> 申请一个 key，然后在终端里设置：

Windows PowerShell：
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-xxxxxxxx"
```

macOS / Linux bash：
```bash
export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

## 3. 第一步：建索引（只做一次，加新 PDF 后重跑即可）

```bash
python paper_polish.py build-index
```

跑完会在脚本同目录生成一个 `.style_index.pkl`。

## 4. 第二步：润色一段英文

三种输入方式任选一种：

```bash
# 方式 A：直接传文本
python paper_polish.py polish --text "Your English paragraph here."

# 方式 B：从文件读
python paper_polish.py polish --input my_draft.txt --output polished.txt

# 方式 C：从 stdin 读（粘贴段落后按 Ctrl-Z / Ctrl-D 结束）
python paper_polish.py polish
```

可选参数：
- `--top-k 5`     检索多少段教授原文作为风格参考（默认 5）
- `--tone academic`  润色语气，可改成 `formal` / `concise` 等
- `--model claude-sonnet-4-5`  指定 Claude 模型

## 5. 输出格式

Claude 会输出两块内容：
```
<polished>
（润色后的英文段落）
</polished>
<notes>
- 中文列出做了哪些主要修改
</notes>
```

如果加了 `--output xxx.txt`，文件里还会附上本次检索到的参考段落元数据（来自哪位教授、哪篇论文、相似度多少），方便你回查。

## 6. 工作流程示意

```
PDF ─► pypdf 抽文本 ─► 清洗 + 分块 ─► sentence-transformers 向量化 ─► .style_index.pkl
                                                                        │
你的段落 ──► 同模型向量化 ──► 余弦相似度检索 top-k ─────────────────────┘
                                                  │
                                                  ▼
                                       Claude API（风格参考 + 你的段落）
                                                  │
                                                  ▼
                                          润色后的英文段落
```

## 7. 常见问题

**Q: 报错 "缺少 pypdf / sentence-transformers / anthropic"**
A: `pip install -r requirements.txt` 即可。

**Q: PDF 抽不出文字？**
A: 个别扫描版 PDF 是图片，需要 OCR（`pytesseract` 之类），目前脚本会自动跳过。

**Q: 想加更多教授的论文？**
A: 在 `Training_dataset/` 下新建一个文件夹（比如 `Prof. XYZ`），把 PDF 放进去，然后：
```bash
python paper_polish.py build-index --professors "Prof. Filip Biljecki" "Prof. Yunmi Park" "Prof. XYZ"
```
