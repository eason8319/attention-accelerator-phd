# 文献 PDF 本地库

> **PDF 不进 Git**。云端仅保留下载脚本；在本机运行脚本后，PDF 将保存到下方各子目录。  
> 文献索引见 [references.bib](../manuscript/references.bib)、[paper_matrix.md](../paper_matrix.md)。

## 目录结构（运行脚本后生成）

```
docs/survey/papers/
├── 00_baseline/          # Attention、FlashAttention、RoPE、RMSNorm
├── L1_flashattention/    # FSA、PLENA、FlatAttention 等
├── L2_mixed_precision/   # SAW-INT4、BitDecoding、QuaRot 等
├── L3_non_gemm/          # MIVE、SOLE、Softermax 等
├── L4_compiler/          # Timeloop、SCALE-Sim v3、TransInferSim 等
├── adjacent_sparse/      # Salca、Sanger 等
├── adjacent_memory/      # AMMA、LoL-PIM、NeuPIM 等
├── download_papers.py    # 批量下载脚本（45 篇，约 142 MB）
└── manifest.json         # 下载完成后自动生成
```

## 本机下载（Windows / macOS / Linux）

需安装 [Python 3](https://www.python.org/downloads/)（勾选 Add to PATH）。

```powershell
cd docs\survey\papers
python download_papers.py
```

或：

```bash
cd docs/survey/papers
python3 download_papers.py
```

脚本会跳过已存在的 PDF，仅下载缺失文件。下载完成后查看 `manifest.json` 确认状态。

## 两台电脑说明

- 电脑 A、B 各自 clone 仓库后**各运行一次**上述脚本即可在本地得到完整 PDF 库
- PDF 仅保存在本机，不参与 Git 同步，避免仓库体积过大
