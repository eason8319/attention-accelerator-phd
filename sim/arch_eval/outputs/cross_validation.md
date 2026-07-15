# P3 Cross-Validation Summary

## Inputs

- `roofline_table.csv`: analytical AI and compute/memory bound
- `scalesim_results.csv`: cycle / utilization / traffic
- `timeloop_energy.csv`: MAC / register / SRAM / DRAM energy

SCALE-Sim attained TOPS assumes a 1 GHz clock for unit conversion only.

## Prefill vs decode utilization

| dataflow | seq | gemm | prefill util | decode util | ratio |
|---|---:|---|---:|---:|---:|
| WS | 4096 | QK_T | 73.149% | 1.053% | 69.47× |
| WS | 4096 | PV | 73.149% | 1.053% | 69.47× |
| WS | 32768 | QK_T | 73.149% | 1.053% | 69.47× |
| WS | 32768 | PV | 73.149% | 1.053% | 69.47× |
| WS | 131072 | QK_T | 73.149% | 1.053% | 69.47× |
| WS | 131072 | PV | 73.149% | 1.053% | 69.47× |
| OS | 4096 | QK_T | 67.374% | 2.107% | 31.98× |
| OS | 4096 | PV | 80.511% | 2.518% | 31.98× |
| OS | 32768 | QK_T | 67.374% | 2.107% | 31.98× |
| OS | 32768 | PV | 80.511% | 2.518% | 31.98× |
| OS | 131072 | QK_T | 67.374% | 2.107% | 31.98× |
| OS | 131072 | PV | 80.511% | 2.518% | 31.98× |

WS `QK_T` utilization: prefill **73.15%** vs decode **1.05%** (69.5×).

## Timeloop energy shares

| mode | seq | MAC | registers | SRAM | DRAM |
|---|---:|---:|---:|---:|---:|
| decode | 4096 | 0.027% | 0.096% | 89.176% | 10.701% |
| decode | 32768 | 0.026% | 0.094% | 89.169% | 10.710% |
| decode | 131072 | 0.026% | 0.094% | 89.167% | 10.713% |
| prefill | 4096 | 0.058% | 0.208% | 99.403% | 0.330% |
| prefill | 32768 | 0.057% | 0.204% | 99.326% | 0.413% |
| prefill | 131072 | 0.057% | 0.203% | 99.302% | 0.437% |

## Figures

- `util_prefill_vs_decode.png`
- `traffic_energy_stack.png`
- `roofline_points.png`
- `cross_joined.csv`: per-GEMM join of the three tools

## Agreement (relative conclusions)

1. **Decode is memory-bound / under-utilized.** Roofline places decode `QK_T`/`PV` at AI≈50.9 ops/byte (below ridge 128), while prefill AI≈248. SCALE-Sim shows decode PE utilization one to two orders of magnitude below prefill for both WS and OS.
2. **Longer context grows traffic and energy.** Both SCALE-Sim DRAM/SRAM traffic and Timeloop total energy rise with seq_len; decode projection GEMMs stay skinny while `QK_T`/`PV` grow with S.
3. **Dataflow modulates absolute util, not the decode gap.** OS and WS differ in absolute percentages, but both preserve decode ≪ prefill.

## Discrepancy sources (expected)

| Tool | What it answers | Why absolutes diverge |
|---|---|---|
| Roofline | Ideal AI and bound latency | Perfect bandwidth overlap; no stalls, tiling waste, or PE mapping inefficiency |
| SCALE-Sim | Cycle-accurate systolic schedule + buffers | Utilization driven by array mapping and tile shape; tile repetition omits inter-tile reuse/overlap |
| Timeloop/Accelergy | Mapping search + energy/area model | Cycle semantics differ from SCALE-Sim; bundled PAT makes 16 MiB SRAM dominate energy, so DRAM share is model-dependent |

An extra scale gap appears on the roofline plot: the analytical peak is 128 TOPS, while a 32×32 array at 1 GHz peaks near 2.05 TOPS. Prefill points sit near that array roof times utilization (~0.65 TOPS), not the 128 TOPS system peak. Compare AI / bound class and util ratios, not absolute TOPS across tools.

Do **not** equate SCALE-Sim cycles with Timeloop cycles, or Roofline microseconds with either simulator. Accept the shared relative story: decode skinny GEMMs sit under the memory roof and leave the array poorly utilized.

## Caveats carried into `analysis.md`

- Fixed 256-bounded tiles: per-tile util is seq-independent.
- Timeloop energy shares need tech-node calibration before claiming DRAM energy dominance.
- Attained TOPS on the roofline plot inherit the 1 GHz assumption.
- Analytical 128 TOPS peak ≠ SCALE-Sim 32×32 microarchitecture peak.
