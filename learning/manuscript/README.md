# P1–P5 Learning Synthesis Manuscript

> **Archived R0 write-up.** Does not define the doctoral mainline; see [`docs/research_plan.md`](../../docs/research_plan.md) (R0–R5) and [`research/`](../../research/).

English IEEE conference-style short paper synthesizing the completed learning projects.

| File | Role |
| --- | --- |
| [`attention_learning_pipeline.tex`](attention_learning_pipeline.tex) | Main paper |
| [`references.bib`](references.bib) | BibTeX source of truth (verified subset) |
| [`IEEEtran.cls`](IEEEtran.cls) | IEEE class (from `templates/latex/ieee/`) |
| [`figures/`](figures/) | Staged plots from P2/P3/P5 `outputs/` |

## Build

```bash
cd learning/manuscript
pdflatex attention_learning_pipeline.tex
pdflatex attention_learning_pipeline.tex
```

The current `.tex` embeds a manual `thebibliography` matched to `references.bib` (no BibTeX pass required).
To switch to BibTeX later: replace the `thebibliography` block with `\bibliographystyle{IEEEtran}` + `\bibliography{references}` and add `IEEEtran.bst`.

> **Note:** This environment lacked a working `pdflatex`/`tectonic` install (apt/docker/GitHub downloads blocked). Compile locally with TeX Live or MiKTeX using the commands above.

## Figure provenance

| Paper figure | Source |
| --- | --- |
| `figures/error_analysis.png` | `learning/p2_quantization/outputs/error_analysis.png` |
| `figures/roofline_points.png` | `learning/p3_arch_eval/outputs/roofline_points.png` |
| `figures/util_prefill_vs_decode.png` | `learning/p3_arch_eval/outputs/util_prefill_vs_decode.png` |
| `figures/traffic_energy_stack.png` | `learning/p3_arch_eval/outputs/traffic_energy_stack.png` |
| `figures/pareto_prefill.png` | `learning/p5_tile_sim/outputs/pareto_prefill_s4096.png` |
| `figures/pareto_decode.png` | `learning/p5_tile_sim/outputs/pareto_decode_s4096.png` (available; optional) |

Regenerate P5 figures if missing:

```bash
conda activate p5-tile-sim
cd learning/p5_tile_sim && python run_p5.py
```

## Citation notes

Unverified survey `.bib` entries were not copied. Timeloop is ISPASS’19; Softermax is DAC’21.
