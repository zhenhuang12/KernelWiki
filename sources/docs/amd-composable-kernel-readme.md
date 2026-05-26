---
id: doc-amd-composable-kernel
title: "AMD Composable Kernel (CK / CK-Tile) Documentation"
url: https://github.com/ROCm/composable_kernel
source_category: official-doc
architectures:
- cdna3
- cdna4
tags:
- composable-kernel
- mfma
- mfma-scale-f8f6f4
- lds
- buffer-load-lds
- gemm
- attention
- flash-attention
- wave-specialization
retrieved_at: 2026-04-27
---

# AMD Composable Kernel (CK / CK-Tile) Documentation

## Overview

ROCm/composable_kernel (CK) is AMD's CUTLASS analogue: a header-only C++ template library that composes high-performance GPU kernels (GEMM, convolutional, attention, normalization, fused-MoE) from reusable "tile programs". Two top-level subprojects share the repo:

- **CK (classic)** — the original tensor-program library used by rocBLAS, MIOpen, and migraphx.
- **CK-Tile** — newer, simpler C++ DSL focused on a per-wave tile-program model that closely mirrors CuTe's tile/copy/MMA abstractions.

CK-Tile is where most new CDNA 3/4 kernels are written (FlashAttention-2/3 forward and backward, fused-MoE, MXFP4 GEMM).

## Architecture

CK kernels are parameterized by a `BlockGemmPipeline` describing:

- Tile shapes (`MPerBlock`, `NPerBlock`, `KPerBlock`).
- MFMA shape (`MFMAShape<16,16,32>`, etc.).
- Pipeline scheduler (`v1` = single-stage, `v2` = double-buffered, `v3` = interwave / wave-specialized).
- LDS layout descriptors (`LdsDescriptor` with XOR-swizzle parameters for bank-conflict-free access).

The `CK_TILE_DEVICE` kernel entry then composes block-level tile programs from `BlockTileGemm`, `BlockTileFmha`, etc. Most performance work in CK is choosing the right block-shape × MFMA-shape × pipeline tuple.

## Reference Kernels

| Kernel | Path | Notes |
|--------|------|-------|
| GEMM FP16/BF16 | `example/ck_tile/03_gemm/` | Baseline GEMM, V3 pipeline |
| GEMM FP8 | `example/ck_tile/03_gemm/gemm_basic_fp8.cpp` | CDNA 3+ FP8 MFMA |
| GEMM MXFP4 | `example/ck_tile/15_grouped_gemm` (CDNA 4 branch) | block-scaled |
| Flash-Attention fwd/bwd | `example/ck_tile/01_fmha/` | Multi-Head Attention, group-query variant |
| Fused MoE | `example/ck_tile/16_fused_moe/` | Top-K routing + grouped GEMM fusion |

## Why It Matters

CK is the closest AMD equivalent of CUTLASS: it is the reusable template substrate underneath production AMD libraries and downstream kernels (AITER, vLLM AMD path, SGLang AMD path). Reading CK templates is the canonical way to understand how MFMA pipelines, LDS swizzling, and direct-to-LDS pipelining come together in real code. See [lang-composable-kernel](../../wiki/languages/composable-kernel.md), [kernel-ck-fp8-gemm-cdna3](../../wiki/kernels/ck-fp8-gemm-cdna3.md).
