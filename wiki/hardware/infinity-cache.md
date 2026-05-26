---
id: hw-infinity-cache
title: "Infinity Cache (MALL) — CDNA Last-Level Cache"
type: hardware
architectures: [cdna3, cdna4]
tags: [infinity-cache, xcd]
confidence: source-reported
related: [hw-xcd, technique-xcd-aware-scheduling, technique-tile-scheduling]
sources: [doc-amd-cdna3-whitepaper, doc-amd-cdna4-whitepaper, blog-gpuopen-mi300-architecture]
aliases: ["Infinity Cache", MALL, "memory-attached last-level cache"]
---

# Infinity Cache (MALL) — CDNA Last-Level Cache

## Overview

Infinity Cache (MALL — Memory-Attached Last-Level cache) is a 256 MB on-package SRAM cache shared by all 8 XCDs on MI300X and MI355X. It sits between the per-XCD L2 caches (4 MB each) and HBM, and provides ~17 TB/s aggregate read bandwidth — roughly 2× HBM3 peak and 2.1× HBM3E peak. Functionally analogous to NVIDIA's L2 (per-GPU, ~50 MB on H100, ~80 MB on B200), but ~3–5× larger.

## Sizing Implications

| Operand class | Fits in L2 (4 MB / XCD)? | Fits in Infinity Cache (256 MB)? |
|---------------|---------------------------|-----------------------------------|
| 4096×4096 BF16 matrix | No (32 MB) | Yes |
| 32 KV pages × 16384 BF16 | Yes (1 MB) | Yes |
| 8-expert MoE weight set (~250 MB) | No | Marginal |
| Attention K/V for 32K context, head_dim=128 | No (~16 MB) | Yes |

Most LLM inference operands fit in Infinity Cache, which is why K-stationary GEMMs see large gains from XCD-aware tile scheduling: keeping each B-tile resident in Infinity Cache while M-tiles cycle through.

## Measurement

The ROCm performance counter `TCC_HIT[0:3]` (per L2 slice) approximates L2 hit rate; `MALL_HIT` (firmware counter, available via rocprofv3) measures Infinity Cache hit rate. A typical un-optimized GEMM shows L2 hit ~5%, MALL hit ~70%; XCD-aware remap pushes L2 hit to ~60%, MALL hit to ~85%.

## Comparison with NVIDIA L2

| Aspect | NVIDIA H100 L2 | NVIDIA B200 L2 | AMD Infinity Cache (MI300X/MI355X) |
|--------|----------------|----------------|--------------------------------------|
| Size | 50 MB | ~80 MB | 256 MB |
| Bandwidth | ~10 TB/s | ~14 TB/s | ~17 TB/s |
| Slices | 12 partitions | 24 partitions | Per-XCD slicing |
| Cache policy hints | TMA / cp.async.bulk | TMA + L2 sectoring | `glc` / `slc` / `dlc` bits per load |

The size advantage is the most concrete CDNA differentiator: AMD inference kernels can amortize HBM reload over many more iterations.

## Cache Policy Control

AMD exposes per-instruction cache hints through buffer-descriptor bits:

- `glc=1` — bypass L1; force re-read from L2/MALL.
- `slc=1` — system-level coherent; bypass MALL on this read.
- `dlc=1` — Disable L1 caching for this access.

In practice, K-stationary GEMMs set `slc=1` on operand-A reads (consume once, don't pollute MALL) and leave `slc=0` on operand-B reads (cache the persistent B tile in MALL).

## Caveats

- Infinity Cache is a victim cache from HBM, not a write-back cache from L2. Writes that miss L2 always go to HBM, then are pulled back into MALL on subsequent reads.
- The CPX-mode partitioning (cf. [hw-xcd](xcd.md)) does not subdivide MALL; even in CPX-8, all partitions share the 256 MB cache.

## See Also

- [hw-xcd](xcd.md) — XCD topology and L2 partitioning
- [technique-xcd-aware-scheduling](../techniques/xcd-aware-scheduling.md) — exploiting MALL via tile mapping
- [blog-gpuopen-mi300-architecture](../../sources/blogs/gpuopen-mi300-architecture.md) — measured hit-rate behavior
