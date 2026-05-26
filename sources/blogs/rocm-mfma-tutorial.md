---
id: blog-rocm-mfma-tutorial
title: "AMD Matrix Cores (MFMA) Tutorial — ROCm Blogs"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- mfma
- agpr
- wavefront-64
- hip
- amdgcn-asm
retrieved_at: 2026-04-27
---

## Summary

The canonical ROCm blog walk-through of CDNA Matrix Cores (MFMA). Covers all three programming layers — `__builtin_amdgcn_mfma_*` intrinsics, `rocwmma::` C++ headers, and direct `v_mfma_*` AMDGCN inline assembly — with worked examples computing a 16×16 BF16 tile from VGPR inputs into AGPR accumulators.

## Key Material

- Per-wave register layout for `v_mfma_f32_16x16x16f16`: 64 lanes contribute one BF16 pair each from a single VGPR; the tile's 256 elements are striped 4-per-lane.
- AGPR allocation: an `f32_16x16x16` MMA writes 4 AGPRs/lane (16×16 / 64 lanes); the LLVM AMDGPU backend allocates AGPRs separately and inserts `v_accvgpr_read/write` copies on demand.
- FP8 path: `v_mfma_f32_16x16x32_fp8_fp8` halves the K-cycle count vs FP16 for the same accumulator footprint.
- Bank conflict avoidance: a XOR swizzle `(row ^ (col >> 2))` on LDS reads keeps 16×16 tiles conflict-free for full-width `ds_read_b128`.

## Why It Matters

This is the practical "how do I actually call an MFMA from HIP" reference cited by most AMD kernel tutorials downstream. The intrinsics-vs-inline-asm comparison is exactly what a porting agent needs when deciding how deep to drop. Cited by [hw-mfma](../../wiki/hardware/mfma.md), [lang-hip](../../wiki/languages/hip.md), [lang-amdgcn-asm](../../wiki/languages/amdgcn-asm.md).
