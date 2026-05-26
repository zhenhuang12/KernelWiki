---
id: blog-rocm-cdna4-gemm-kernels
title: "FP8 GEMM Optimization on AMD CDNA 4 Architecture"
author: Jiahui Cao, Amanzhol Salykov, Andy Luo (ROCm Engineering)
url: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
source_category: benchmark-blog
architectures:
- cdna4
- cdna3
tags:
- gemm
- fp8
- mfma
- buffer-load-lds
- lds-swizzling
- ping-pong-scheduling
- wave-priority
- sched-barrier
retrieved_at: 2026-05-26
---

## Summary

Step-by-step ROCm engineering walk-through of building an FP8 GEMM kernel for AMD CDNA 4 (gfx950 / MI355X) in pure HIP/C++. The kernel evolves across nine stages, each lifting throughput on a 4096³ FP8 workload:

1. Naive scalar loop — **1.15 TFLOPS**
2. Add MFMA 16×16×128 — **30.05 TFLOPS**
3. Add `llvm.amdgcn.raw.buffer.load.lds` (direct global → LDS) — **506.7 TFLOPS**
4. Add LDS XOR swizzle (bank-conflict elimination) — improvement folded into next stage
5. Double buffering — **1166 TFLOPS**
6. Larger tile shape with proper register budget — further uplift
7. Multi-stage software pipeline — incremental
8. 8-wave block ping-pong with `s_setprio` + `sched_barrier` — **2680 TFLOPS** (4096³)
9. Same template at 8192³ — **3204 TFLOPS** (beats hipBLASLt's 3130)

Includes a CDNA 3 vs CDNA 4 contrast table covering MFMA shape availability, direct-to-LDS variants (32-bit vs 128-bit), and AGPR pool semantics.

## Why It Matters

This is the canonical AMD-authored reference for how each individual technique compounds on CDNA 4. Any wiki page citing FP8 GEMM perf or 9-stage optimization decomposition should anchor to this blog. The ping-pong + `s_setprio` recipe is the closest first-party documentation we have of AMD's analogue to NVIDIA warp-specialization scheduling discipline.

Cited by [kernel-ck-fp8-gemm-cdna3](../../wiki/kernels/ck-fp8-gemm-cdna3.md), [kernel-ck-mxfp4-gemm-cdna4](../../wiki/kernels/ck-mxfp4-gemm-cdna4.md), [technique-mfma-pipelining](../../wiki/techniques/mfma-pipelining.md), [technique-direct-to-lds](../../wiki/techniques/direct-to-lds.md), [technique-lds-swizzling](../../wiki/techniques/lds-swizzling.md).
