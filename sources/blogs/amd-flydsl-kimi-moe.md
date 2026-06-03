---
id: blog-amd-flydsl-kimi-moe
title: "Accelerating Kimi-K2.5 on MI300X — FlyDSL Mixed-Precision Fused MoE"
author: "Bobo Fang, Chunhung Wang, Clement Lin, Dai Yan, Eveline Chen, Felix Li, Menghsuan Yang, Peng Sun (AMD)"
url: https://rocm.blogs.amd.com/artificial-intelligence/kimi-k2.5-optimize/README.html
source_category: benchmark-blog
architectures:
- cdna3
- cdna4
tags:
- flydsl
- moe
- fused-moe
- quantization
- fine-grained-quantization
- jit-compilation
retrieved_at: 2026-06-03
---

## Summary

AMD ROCm blog (2026-03-24) walking through a FlyDSL-authored mixed-precision **fused MoE** kernel
for **Kimi-K2.5** on MI300X (gfx942), replacing SGLang's default Triton `fused_moe`. This is the
strongest published FlyDSL performance case study and a concrete counterpoint to the
merged-then-reverted [pr-aiter-3117](../prs/aiter/PR-3117.md): here FlyDSL delivers a real,
benchmarked win. Backs [lang-flydsl](../../wiki/languages/flydsl.md) and
[technique-preshuffle-gemm](../../wiki/techniques/preshuffle-gemm.md).

## Bottleneck

Profiling Kimi-K2.5 found `fused_moe` (`fused_moe_kernel_gptq_awq`) dominates GPU time:
**87.8%** at concurrency=2, **89.7%** at concurrency=40 (~88–90%).

## The FlyDSL Kernel

A single FlyDSL fused-MoE kernel supporting both **BF16 (A16W16)** and **W4A16** paths, able to
run different MoE stages at different precision. `FLYDSL_W4A16_HYBRID=w2_bf16` keeps Stage 1
(gate/up) in W4A16 and runs Stage 2 (down projection) in BF16, trading accuracy vs throughput.
Authored in Python with FlyDSL's block→warp→thread→MFMA tiling control, explicit global→LDS→register
movement, swizzling and vectorization; lowered through MLIR (canonicalize / CSE / gpu-to-rocdl) to
gfx942/gfx950.

> Note: the blogs brand FlyDSL's layout IR **"FLIR" (Flexible Layout Intermediate Representation)**;
> the actual MLIR dialect in the repo source is named `fly`.

## Kernel Benchmark (most-invoked shape)

tokens=16384, dim=7168, inter=512, E=384, topk=8 (>half of all invocations):

| dtype | Triton | CK | FlyDSL |
|-------|--------|----|--------|
| bf16  | 12.09 ms | gpu_fault | **8.68 ms** |
| w4a16 | 31.43 ms | unsupported | **9.77 ms** |

CK lacks W4A16 and faulted on Kimi's E=384 shapes; Triton would need more tuning.

## End-to-End (SGLang + AITER, no accuracy loss; GSM8K 0.96 → 0.96)

| metric | concurrency=2 | concurrency=40 |
|--------|---------------|----------------|
| TTFT   | 2918 → 1014 ms (**−65.3%**) | 33478 → 17730 ms (**−47.0%**) |
| TPOT   | 38.77 → 28.26 ms (**−27.1%**) | 230.37 → 70.86 ms (**−69.2%**) |
| throughput | 45.04 → 66.24 tok/s (**+47.1%**) | 135.39 → 355.35 tok/s (**+162.4%**) |

Headline: up to **65% lower TTFT, 69% lower TPOT, 162% higher throughput**.

## Status caveat

The post says these optimizations "will be progressively merged into the upstream SGLang and
AITER repositories" — i.e. as of the blog the win is demonstrated but the upstreaming was in
progress (consistent with the in-flight / partially-merged FlyDSL-in-AITER PR landscape). A
follow-up blog covers W4A8/W8A8 quantization via AMD Quark + FlyDSL on MI325X.
