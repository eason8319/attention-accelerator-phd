# Timeloop + Accelergy Summary

## Method

- Official image: `timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64`
- Architecture: 32×32 INT8 MACs, 16 MiB global SRAM, DRAM
- Mapper: energy then delay; fixed C×K spatial mapping
- Workloads use the same bounded tiles and repetition counts as SCALE-Sim
- Energy source: Timeloop PAT; area source: Accelergy/CACTI/Aladdin

Official 2020 ISPASS tutorial exercise 00 passes in this image. The repository's newer v0.4 `example_designs` do not parse unchanged because their schema is newer than the bundled front-end; this project therefore uses the compatible legacy schema.

## Per-layer energy breakdown

| mode | seq | MAC | registers | SRAM | DRAM | total (mJ) |
|---|---:|---:|---:|---:|---:|---:|
| decode | 4096 | 0.027% | 0.096% | 89.176% | 10.701% | 94.9248 |
| decode | 32768 | 0.026% | 0.094% | 89.169% | 10.710% | 316.724 |
| decode | 131072 | 0.026% | 0.094% | 89.167% | 10.713% | 1077.18 |
| prefill | 4096 | 0.058% | 0.208% | 99.403% | 0.330% | 178745 |
| prefill | 32768 | 0.057% | 0.204% | 99.326% | 0.413% | 4.78692e+06 |
| prefill | 131072 | 0.057% | 0.203% | 99.302% | 0.437% | 6.51859e+07 |

## Area estimate

| component | instances | total area (mm²) |
|---|---:|---:|
| Register | 1024 | 0.085164 |
| mac | 1024 | 0.477568 |
| DRAM | 1 | 0 |
| GlobalBuffer | 1 | 99.9581 |
| **Total** | — | **100.521** |

## Interpretation

Under the bundled 45 nm PAT model, the 16 MiB global SRAM—not DRAM—dominates dynamic energy. This does not support a literal "DRAM energy dominates" claim; the robust conclusion from Roofline/SCALE-Sim is bandwidth pressure and low decode PE utilization. Absolute energy shares require technology and memory-model calibration before publication.
