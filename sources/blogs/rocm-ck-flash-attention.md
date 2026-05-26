---
id: blog-rocm-ck-flash-attention
title: "Flash Attention on AMD MI300X with Composable Kernel — ROCm Blogs"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/artificial-intelligence/flash-attention-mi300/README.html
source_category: benchmark-blog
architectures:
- cdna3
tags:
- flash-attention
- attention
- composable-kernel
- mfma
- lds
- wave-specialization
- fp8
retrieved_at: 2026-04-27
---

## Summary

ROCm Blogs walk-through of the CK-Tile FlashAttention-2 forward kernel on MI300X. Details the wave-specialized producer/consumer structure (two waves load Q/K/V tiles into LDS while two waves run softmax+MFMA), the LDS double-buffered K/V layout, and the FP8 (E4M3) prefill variant that re-uses the same skeleton with `v_mfma_f32_16x16x32_fp8_fp8`.

## Key Performance Numbers

- BF16 prefill, head dim 128: ~480 TFLOPS sustained on MI300X (≈75% of peak BF16 MFMA throughput).
- FP8 prefill, head dim 128: ~840 TFLOPS, ≈68% of peak FP8 MFMA throughput.
- BF16 decode (group-query, kv-paged): ~32% of HBM3 bandwidth ceiling, limited by KV-cache reload pattern.

## Architecture Notes

- 4 waves per workgroup, 2 producer + 2 consumer. No `s_barrier` between producer/consumer — instead `s_waitcnt vmcnt(0) & lgkmcnt(0)` after the producer's `buffer_load_dword_lds` issues, then a single `s_barrier` per K iteration.
- LDS layout uses interleaved K-major / V-major banks to keep both consumer reads conflict-free.
- Online softmax follows the standard Flash-Attention recipe; rescaling factors stored in AGPRs to free VGPRs for next-stage operand loads.

## Why It Matters

This is the canonical worked example of a wave-specialized attention pipeline on CDNA 3, and the upstream pattern that AITER's production attention kernels are derived from. Cited by [kernel-aiter-mla-decode](../../wiki/kernels/aiter-mla-decode.md), [technique-wave-specialization](../../wiki/techniques/wave-specialization.md).
