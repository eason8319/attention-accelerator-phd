# Roofline table (LLaMA-7B-scale attention layer)

## Assumption

Each GEMM reads A/B once from HBM and writes C once; ops count uses $2MNK$.

## Hardware

- Peak compute: **128 TOPS** INT8
- HBM bandwidth: **1 TB/s**
- SRAM (reference): **16 MB**
- Element size: **1 byte** (INT8)
- Ridge AI: **128 ops/byte**

## Per-GEMM

| mode | seq | gemm | M | N | K | AI | bound | t_bound (us) |
|---|---:|---|---:|---:|---:|---:|---|---:|
| prefill | 4096 | QKV_proj | 4096 | 12288 | 4096 | 3511 | compute | 3221.225 |
| prefill | 4096 | QK_T | 131072 | 4096 | 128 | 248 | compute | 1073.742 |
| prefill | 4096 | PV | 131072 | 128 | 4096 | 248 | compute | 1073.742 |
| prefill | 4096 | O_proj | 4096 | 4096 | 4096 | 2731 | compute | 1073.742 |
| prefill | 32768 | QKV_proj | 32768 | 12288 | 4096 | 5617 | compute | 25769.804 |
| prefill | 32768 | QK_T | 1048576 | 32768 | 128 | 255 | compute | 68719.477 |
| prefill | 32768 | PV | 1048576 | 128 | 32768 | 255 | compute | 68719.477 |
| prefill | 32768 | O_proj | 32768 | 4096 | 4096 | 3855 | compute | 8589.935 |
| prefill | 131072 | QKV_proj | 131072 | 12288 | 4096 | 6003 | compute | 103079.215 |
| prefill | 131072 | QK_T | 4194304 | 131072 | 128 | 255.7 | compute | 1099511.628 |
| prefill | 131072 | PV | 4194304 | 128 | 131072 | 255.7 | compute | 1099511.628 |
| prefill | 131072 | O_proj | 131072 | 4096 | 4096 | 4033 | compute | 34359.738 |
| decode | 4096 | QKV_proj | 1 | 12288 | 4096 | 1.999 | memory | 50.348 |
| decode | 4096 | QK_T | 32 | 4096 | 128 | 50.88 | memory | 0.659 |
| decode | 4096 | PV | 32 | 128 | 4096 | 50.88 | memory | 0.659 |
| decode | 4096 | O_proj | 1 | 4096 | 4096 | 1.999 | memory | 16.785 |
| decode | 32768 | QKV_proj | 1 | 12288 | 4096 | 1.999 | memory | 50.348 |
| decode | 32768 | QK_T | 32 | 32768 | 128 | 51.16 | memory | 5.247 |
| decode | 32768 | PV | 32 | 128 | 32768 | 51.16 | memory | 5.247 |
| decode | 32768 | O_proj | 1 | 4096 | 4096 | 1.999 | memory | 16.785 |
| decode | 131072 | QKV_proj | 1 | 12288 | 4096 | 1.999 | memory | 50.348 |
| decode | 131072 | QK_T | 32 | 131072 | 128 | 51.19 | memory | 20.976 |
| decode | 131072 | PV | 32 | 128 | 131072 | 51.19 | memory | 20.976 |
| decode | 131072 | O_proj | 1 | 4096 | 4096 | 1.999 | memory | 16.785 |

## Layer summary (sum of per-GEMM bound latencies)

| mode | seq | AI | bound | t_layer (us) |
|---|---:|---:|---|---:|
| decode | 4096 | 2.941 | memory | 68.452 |
| decode | 32768 | 8.645 | memory | 77.627 |
| decode | 131072 | 20.92 | memory | 109.085 |
| prefill | 4096 | 646.2 | compute | 6442.451 |
| prefill | 32768 | 314.7 | compute | 171798.692 |
| prefill | 131072 | 270.9 | compute | 2336462.209 |

## Sanity check (decode vs prefill on QK_T / PV)

Decode `QK_T`/`PV` AI should be below prefill and below the ridge, hence memory-bound.

- seq=4096: QK_T prefill AI=248 (compute) vs decode AI=50.88 (memory); ratio=4.87x
- seq=4096: PV prefill AI=248 (compute) vs decode AI=50.88 (memory); ratio=4.87x
- seq=32768: QK_T prefill AI=255 (compute) vs decode AI=51.16 (memory); ratio=4.98x
- seq=32768: PV prefill AI=255 (compute) vs decode AI=51.16 (memory); ratio=4.98x
- seq=131072: QK_T prefill AI=255.7 (compute) vs decode AI=51.19 (memory); ratio=5.00x
- seq=131072: PV prefill AI=255.7 (compute) vs decode AI=51.19 (memory); ratio=5.00x
