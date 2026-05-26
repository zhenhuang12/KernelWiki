---
id: doc-amd-aiter-readme
title: "AMD AITER — High-Performance AI Tensor Engine Routines"
url: https://github.com/ROCm/aiter
source_category: official-doc
architectures:
- cdna3
- cdna4
tags:
- attention
- flash-attention
- mla
- moe
- fused-kernel
- mfma
- mfma-scale-f8f6f4
- mxfp4
- fp8
- composable-kernel
- hip
retrieved_at: 2026-04-27
---

# AMD AITER — AI Tensor Engine Routines

## Overview

[ROCm/aiter](https://github.com/ROCm/aiter) is AMD's flashinfer / cuBLASLt-equivalent: a Python+HIP/CK kernel library focused on LLM inference primitives — flash-attention (decode and prefill), MLA, fused MoE, paged-KV attention, rotary, RMSNorm, all-reduce. Roughly the AMD analogue of `flashinfer-ai/flashinfer` in scope.

## Kernel Catalogue (representative)

| Kernel | Module | Notes |
|--------|--------|-------|
| Paged-Attention decode | `aiter/ops/attention.py` → `attention_decode` | wave-specialized, group-query, FP8 KV |
| Flash-Attention prefill | `aiter/ops/attention.py` → `mha_fwd` | CK-Tile backbone, FP16/BF16/FP8 |
| MLA decode | `aiter/ops/mla.py` | DeepSeek V3 MLA, fused with rotary |
| Fused MoE | `aiter/ops/moe.py` → `fused_moe` | Top-K + grouped-GEMM + activation, FP8 / MXFP4 |
| MXFP4 GEMM | `aiter/ops/gemm_mxfp4.py` | CDNA 4 only, uses `v_mfma_scale_f32_*_f8f6f4` |
| RMSNorm + Quant fused | `aiter/ops/norm.py` | per-token FP8 / MXFP4 quantization |

AITER kernels are usually thin Python wrappers around either:
1. Hand-written HIP+CK templates (the production path), or
2. ROCm/triton-mlir generated kernels (for fast iteration on new shapes).

## Backend Selection

AITER auto-dispatches by architecture:

- `gfx942` (CDNA 3 / MI300X) → FP8 MFMA path.
- `gfx950` (CDNA 4 / MI355X) → MXFP4 / MXFP6 MFMA path with block-scaled accumulation; falls back to FP8 if `cbsz/blgp` not requested.
- Other gfx → MIOpen reference path.

## Why It Matters

AITER is the highest-leverage AMD source corpus for LLM inference kernel patterns. Its decode-attention and fused-MoE implementations are the AMD reference points compared against FlashInfer / vLLM / SGLang on Hopper/Blackwell. PRs in `ROCm/aiter` are a primary signal for what shapes and dtypes AMD optimizes first. See [kernel-aiter-fused-moe](../../wiki/kernels/aiter-fused-moe.md), [kernel-aiter-mla](../../wiki/kernels/aiter-mla.md).
