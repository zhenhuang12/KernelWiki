---
id: doc-amd-cdna4-whitepaper
title: "AMD CDNA 4 / Instinct MI350 Series Architecture"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-white-paper.pdf
source_category: official-doc
architectures:
- cdna4
tags:
- mfma
- mfma-scale-f8f6f4
- lds
- xcd
- infinity-cache
- mxfp4
- mxfp6
- mxfp8
- block-scale
- fp8
- fp6
- fp4
- wavefront-64
retrieved_at: 2026-04-27
---

# AMD CDNA 4 / Instinct MI350 Series Architecture

## Overview

Official AMD whitepaper describing the CDNA 4 (Instinct MI350X / MI355X / gfx950) compute architecture. CDNA 4 retains the chiplet/XCD topology of CDNA 3 (8 XCDs) but doubles per-CU LDS to 160 KB, adds OCP MXFP4/MXFP6/MXFP8 block-scaled matrix instructions, and lifts HBM3E bandwidth to 8 TB/s with 288 GB capacity.

## Key Differences vs CDNA 3

| Aspect | CDNA 3 (MI300X) | CDNA 4 (MI355X) |
|--------|------------------|------------------|
| LDS per CU | 64 KB | 160 KB |
| Direct-to-LDS width | 32 bits (1 dword) | 128 bits (4 dwords) |
| Native dtypes | FP8 E4M3/E5M2 | + FP6 E2M3/E3M2, FP4 E2M1 |
| Block scaling | software emulation | hardware MXFP (UE8M0 scale, 32-elem block) |
| HBM | 192 GB HBM3 @ 5.3 TB/s | 288 GB HBM3E @ 8 TB/s |
| Infinity Cache | 256 MB | 256 MB |
| Process | N5 | N3 (compute die) |
| Partitioning | CPX / DPX | CPX / DPX |

## Block-Scaled MFMA (`v_mfma_scale_*`)

The headline CDNA 4 instruction family is `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`, which accepts mixed-precision operands (FP8/FP6/FP4) and an 8-bit per-block scale (UE8M0, 32-element blocks). This is the AMD analogue of Blackwell's `tcgen05.mma.kind::mxf4nvf4` and `kind::mxf8f6f4` instructions. Both operands carry their own block-scale stream — the wave issues:

```
v_mfma_scale_f32_32x32x64_f8f6f4 v[acc:acc+15],
                                  v[a:a+3], v[b:b+3],
                                  v[sa], v[sb],
                                  cbsz:0 blgp:0 abid:0
```

`cbsz` selects the A-operand sub-format (FP8/FP6/FP4) and `blgp` selects the B-operand sub-format; `abid` controls scale broadcast within the block.

## CPX / DPX Partitioning

CDNA 4 retains the Compute Partitioning (CPX) and Memory Partitioning (DPX) modes. CPX splits the GPU into 1, 2, 4, or 8 logical accelerators (1 XCD per partition in CPX-8), useful for tenant isolation and small-batch inference. Kernels written without partition awareness see fewer CUs in CPX modes; XCD-aware schedulers must consult the partition mask.

## Why It Matters

CDNA 4 closes most of the dtype gap with Blackwell: MXFP4/MXFP6/MXFP8 are the same OCP microscaling formats Blackwell calls NVFP4 (with UE8M0 scaling) — code paths and quantization recipes port across the two vendors. Wider direct-to-LDS (128 b vs 32 b) makes producer/consumer wave pipelines significantly easier to feed.
