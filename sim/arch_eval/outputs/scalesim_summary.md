# SCALE-Sim 32×32 Attention Sweep

## Method

- SCALE-Sim version: 3.0.0
- Array: 32×32; dataflows: WS and OS
- SRAM: 6 MiB IFMAP + 6 MiB filter + 4 MiB OFMAP
- Workloads: prefill/decode × 4K/32K/128K × four GEMMs
- Exact fixed tiles bounded by 256 per M/N/K dimension are simulated once and multiplied by their repetition count.
- Aggregate cycles/traffic exclude inter-tile reuse and overlap; use them for trend comparison, not absolute end-to-end latency.

## Utilization

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

## Interpretation

Decode preserves only one query row per head, so both dataflows show severe PE under-utilization. WS and OS map different GEMM dimensions spatially/temporally, which explains their different absolute utilization.
Because all sequence lengths use the same fixed tile shape, per-tile utilization is sequence-length independent; sequence length changes tile count, aggregate cycles and traffic.

Raw reports are under `outputs/scalesim_raw/`; aggregated cycle, utilization and SRAM/DRAM traffic are in `scalesim_results.csv`.
