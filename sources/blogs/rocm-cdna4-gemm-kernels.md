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

Step-by-step ROCm engineering walk-through of building an FP8 GEMM kernel for AMD CDNA 4 (gfx950 / MI355X) in pure HIP/C++. The kernel evolves across nine stages on a 4096³ FP8 workload:

1. Naive baseline — **1.15 TFLOPS**
2. LDS tiling — **4.80 TFLOPS**
3. Matrix-core (MFMA) baseline — **30.05 TFLOPS**
4. MFMA + vectorized global loads — **336.88 TFLOPS**
5. MFMA + direct global→LDS load (`llvm.amdgcn.raw.buffer.load.lds`) — **506.70 TFLOPS**
6. LDS XOR swizzle stacked on direct-to-LDS — **497.43 TFLOPS** (slight regression; pays off later when paired with double buffering)
7. Double buffering — **1166.41 TFLOPS**
8. Multi-wave tuning (256×256 tile, 512 threads) — **2288.16 TFLOPS**
9. 8-wave block ping-pong with `s_setprio` + `sched_barrier` — **2680.33 TFLOPS**

At 4096³ the final kernel lands within ~2.5% of hipBLASLt (2680 vs 2750 TFLOPS). At 8192³ the same template reaches **3204.15 TFLOPS**, edging out hipBLASLt's **3130.21 TFLOPS**. All without dropping to assembly — the work stays at the HIP/C++ level throughout.

Includes a CDNA 3 vs CDNA 4 contrast table covering MFMA shape availability, direct-to-LDS variants (32-bit vs 128-bit), and AGPR pool semantics.

## Why It Matters

This is the canonical AMD-authored reference for how each individual technique compounds on CDNA 4. Any wiki page citing FP8 GEMM perf or 9-stage optimization decomposition should anchor to this blog. The ping-pong + `s_setprio` recipe is the closest first-party documentation we have of AMD's analogue to NVIDIA warp-specialization scheduling discipline.

Cited by [kernel-ck-fp8-gemm-cdna3](../../wiki/kernels/ck-fp8-gemm-cdna3.md), [kernel-ck-mxfp4-gemm-cdna4](../../wiki/kernels/ck-mxfp4-gemm-cdna4.md), [technique-mfma-pipelining](../../wiki/techniques/mfma-pipelining.md), [technique-direct-to-lds](../../wiki/techniques/direct-to-lds.md), [technique-lds-swizzling](../../wiki/techniques/lds-swizzling.md).
