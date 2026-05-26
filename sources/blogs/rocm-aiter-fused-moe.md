---
id: blog-rocm-aiter-fused-moe
title: "AITER Fused MoE on MI300X: Top-K Routing + Grouped GEMM Fusion"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/artificial-intelligence/aiter-fused-moe/README.html
source_category: benchmark-blog
architectures:
- cdna3
- cdna4
tags:
- moe
- fused-kernel
- grouped-gemm
- mfma
- fp8
- mxfp4
- composable-kernel
- ep-dispatch-combine
retrieved_at: 2026-04-27
---

## Summary

ROCm Blogs case study on AITER's fused-MoE kernel. The kernel collapses Top-K routing, expert dispatch reshape, two grouped GEMMs (gate-up + down), SiLU activation, and the combine reduction into a single CK-Tile program — roughly the AMD analogue of vLLM's `fused_moe_kernel` with FlashInfer's grouped-GEMM expert path.

## Architecture

Three wave roles per CTA:

1. **Router waves** (1 wave) — load logits, compute Top-K, write per-expert offsets to LDS.
2. **Gate-up producer waves** (2 waves) — `buffer_load_dword_lds` issues for the expert weight tiles, indexed by the LDS offsets the router wrote.
3. **MFMA + activation + down + combine** (5 waves) — consume LDS, MFMA into AGPR accumulators, fuse SiLU and the down-projection, write to global with a final `atomicAdd`-based combine.

## Performance Claims

- Llama-3-70B-style MoE block (8 experts, Top-2), BF16 weights, MI300X: 1.6× speedup over the unfused (3-kernel) baseline at batch ≤ 32.
- Same shape, MI355X, MXFP4 weights: 2.4× speedup vs unfused FP8; expert weight reload from HBM3E saturates at 70% of peak.
- EP-dispatch path (RCCL `AllToAll` outside the kernel) is unchanged but the in-kernel combine eliminates a launch + global round-trip.

## Why It Matters

Fused MoE is the most compute-dense LLM kernel on Instinct. AITER's implementation is the reference for how to compose CK-Tile producer/consumer waves around per-expert grouped GEMM, and the only first-party MXFP4 MoE on MI355X. Cited by [kernel-aiter-fused-moe](../../wiki/kernels/aiter-fused-moe.md).
