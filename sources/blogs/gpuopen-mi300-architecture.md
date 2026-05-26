---
id: blog-gpuopen-mi300-architecture
title: "AMD Instinct MI300X Architecture Deep Dive (GPUOpen)"
author: AMD GPUOpen
url: https://gpuopen.com/learn/amd-instinct-mi300x-deep-dive/
source_category: community-note
architectures:
- cdna3
tags:
- mfma
- lds
- xcd
- infinity-cache
- infinity-fabric
- agpr
- wavefront-64
retrieved_at: 2026-04-27
---

## Summary

GPUOpen's developer-facing deep dive into the MI300X CDNA 3 architecture. Covers XCD topology, AGPR/VGPR register file partitioning, LDS bank layout, Infinity Cache hit-rate measurement methodology, and tile-mapping strategies that exploit the 8-XCD chiplet structure. The article walks through measured Infinity Cache bandwidth (~17 TB/s aggregate) and shows how round-robin XCD dispatch hurts L2 locality vs an explicit XCD-aware permutation.

## Key Takeaways

- XCD-aware tile mapping (manually remapping `blockIdx.x` so adjacent K-tiles land on the same XCD) is the single biggest "free" speedup for cache-bound GEMMs on MI300X.
- The 512-entry VGPR pool per SIMD is shared with the 512-entry AGPR pool; over-allocating AGPRs for MFMA accumulators reduces VGPR headroom for software-pipelined loads.
- LDS is 32 banks × 4 B; XOR swizzles patterned on `(row ^ col)` reach full 128 B/cycle without conflicts for standard 16×16 and 32×32 MFMA tiles.

## Why It Matters

This blog is the most accessible deep-dive on what makes CDNA 3 perform — the official whitepaper covers the same ground but in less actionable detail. Cited by [hw-xcd](../../wiki/hardware/xcd.md), [technique-xcd-aware-scheduling](../../wiki/techniques/xcd-aware-scheduling.md).
