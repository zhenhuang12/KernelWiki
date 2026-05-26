---
id: kernel-ck-fp8-gemm-cdna3
title: "Composable Kernel FP8 GEMM (CK-Tile V3) on CDNA 3"
type: kernel
architectures: [cdna3]
tags: [gemm, fp8, mfma, lds, buffer-load-lds, wave-specialization, mfma-pipelining, composable-kernel]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [composable-kernel, hip, amdgcn-asm]
related: [hw-mfma, hw-buffer-load-lds, hw-lds, hw-agpr, technique-wave-specialization, technique-mfma-pipelining, technique-direct-to-lds, technique-lds-swizzling, technique-xcd-aware-scheduling, lang-composable-kernel]
sources: [pr-composable-kernel-1384, pr-composable-kernel-1853, blog-rocm-ck-flash-attention, blog-rocm-buffer-load-lds, blog-rocm-cdna4-gemm-kernels, doc-amd-cdna3-isa, doc-amd-composable-kernel]
performance_claims:
  - gpu: MI300X
    dtype: fp8_e4m3
    shape: "M=4096, N=4096, K=4096"
    metric: TFLOPS
    value: 1180
    utilization: "~76%"
    source_id: pr-composable-kernel-1384
    evidence_basis: source-reported
  - gpu: MI300X
    dtype: fp8_e4m3
    shape: "M=8192, N=8192, K=8192"
    metric: TFLOPS
    value: 1240
    utilization: "~80%"
    source_id: pr-composable-kernel-1853
    evidence_basis: source-reported
aliases: ["CK-Tile FP8 GEMM", "CK V3 pipeline FP8 GEMM"]
---

# Composable Kernel FP8 GEMM (CK-Tile V3) on CDNA 3

## Overview

The V3 pipeline in CK-Tile is the canonical wave-specialized FP8 GEMM for MI300X. It pairs `buffer_load_dword_lds` producers with `v_mfma_f32_*_fp8_fp8` consumers, drives a 2-stage LDS double-buffer (the CDNA-3 LDS budget of 64 KB/CU caps stages at 2 for the typical 256×256×64 block tile), and reaches ~76-80% of MI300X peak FP8 throughput on square BF16-output GEMMs.

Used in production by AITER's MoE GEMM, vLLM's AMD path, and ROCm/Megatron's FP8 training kernels.

## Block Shape

| Parameter | Value | Reason |
|-----------|-------|--------|
| BLOCK_M × BLOCK_N | 256 × 256 | Fills 2 waves of 32×32 MFMA across M, 8 across N |
| BLOCK_K | 64 | Matches `v_mfma_f32_32x32x16_fp8` four times per K-stage |
| MFMA shape | 32×32×16 FP8 | 4 AGPRs/lane per MMA × 8 in-flight = 32 AGPRs |
| Waves / WG | 4 (256 threads) | 2 producers + 2 consumers |
| Stages | 2 | LDS budget: 2 × (256·64 + 64·256) · 1 B = 64 KB |

## Pipeline Skeleton

```cpp
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_v3.hpp"

using GemmShape  = ck_tile::TileGemmShape<256, 256, 64>;
using MfmaShape  = ck_tile::sequence<32, 32, 16>;
using Pipeline   = ck_tile::BlockGemmPipelineV3<
    ck_tile::fp8_t,       // A dtype
    ck_tile::fp8_t,       // B dtype
    float,                // accumulator
    GemmShape, MfmaShape,
    /*num_warp_groups=*/2,    // 2 producer waves + 2 consumer waves
    /*pipeline_stages=*/2>;

__global__ void __launch_bounds__(256)
ck_fp8_gemm_v3(const fp8_t* A, const fp8_t* B, bf16* C,
               int M, int N, int K) {
    // XCD-aware tile remap — see technique-xcd-aware-scheduling
    int logical_bid = ck_tile::xcd_remap(blockIdx.x);
    auto [m_tile, n_tile] = ck_tile::tile_partition(logical_bid, M / 256, N / 256);

    __shared__ ck_tile::SmemStorage<Pipeline> smem;

    auto a_window = ck_tile::make_tile_window(A, {M, K}, {m_tile * 256, 0});
    auto b_window = ck_tile::make_tile_window(B, {K, N}, {0, n_tile * 256});

    auto acc = Pipeline::run(a_window, b_window, K / 64, smem);

    // Epilogue: AGPR -> VGPR -> bf16 -> global
    ck_tile::store_tile(C, {M, N}, {m_tile * 256, n_tile * 256},
                        ck_tile::cast<bf16>(acc));
}
```

The `BlockGemmPipelineV3` template hides the wave-id branching, `s_waitcnt`, `s_barrier`, XOR-swizzle LDS layout, and `__builtin_amdgcn_sched_barrier` annotations.

## What V3 Does That V1/V2 Doesn't

| Pipeline | Producer | Consumer | LDS stages | Peak FP8 % on MI300X |
|----------|----------|----------|------------|------------------------|
| V1 (legacy) | inline `buffer_load + ds_write` | MFMA after barrier | 1 | ~40% |
| V2 | inline `buffer_load + ds_write` | MFMA with prefetch | 2 | ~60% |
| V3 (this) | wave-specialized `buffer_load_dword_lds` | MFMA + sched_barrier | 2 | ~76-80% |

V3 is the first CK pipeline to fully wave-specialize and use direct-to-LDS.

## Inner Loop (Consumer Wave)

```cpp
// Pseudo-AMDGCN — what V3 emits for the consumer's K-step
__syncthreads();                                        // wait for producer stage

#pragma unroll
for (int kk = 0; kk < 64; kk += 16) {
    auto a_frag = ds_read_b128(smem_a[stage] + xor_swizzle(row, kk));
    auto b_frag = ds_read_b128(smem_b[stage] + xor_swizzle(kk, col));
    __builtin_amdgcn_sched_barrier(0x0);
    acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_frag, b_frag, acc, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0x0);
}

__syncthreads();                                        // release stage
```

The two `sched_barrier(0x0)` calls pin the four MFMA issues so the LLVM scheduler doesn't interleave the next K-stage's `ds_read` between them — empirically worth ~5% throughput.

## XCD-Aware Tile Mapping

This kernel is square-K-stationary: each (m_tile, n_tile) pair touches K/64 ≈ 64 producer rounds, so the A-row and B-column data is heavily reused within the L2. Without XCD-aware remap the 8-XCD round-robin scatters these tiles and L2 hit rate collapses; with remap it stays around 70%. See [technique-xcd-aware-scheduling](../techniques/xcd-aware-scheduling.md).

## When to Use

- Production FP8 inference on MI300X (vLLM, SGLang, Megatron paths use this).
- Training with FP8 forward + BF16 backward.
- Any GEMM where M, N ≥ 256 and K ≥ 1024.

For smaller GEMMs (M < 256 or skinny K), the V3 pipeline under-utilizes — fall back to V2 or the AITER batched-small-GEMM variant.

## Caveats

- 2 stages only: long K (>16 K) is bandwidth-bound on MI300X regardless of pipeline; CDNA 4 with 160 KB LDS lifts this to 3-4 stages.
- The `sched_barrier` annotations are sensitive to clang version — pin to ROCm ≥ 6.1 to avoid scheduler regressions seen on earlier builds.
- FP8 accumulation in MFMA is IEEE-ish but not bit-exact across compilers; do not compare bit-for-bit with the BF16 reference.

## See Also

- [pr-composable-kernel-1384](../../sources/prs/composable_kernel/PR-1384.md) — real FP8 GEMM optimized PR
- [pr-composable-kernel-1853](../../sources/prs/composable_kernel/PR-1853.md) — Compute V2 2-LDS ping-pong
- [lang-composable-kernel](../languages/composable-kernel.md) — CK-Tile primitives
- [technique-wave-specialization](../techniques/wave-specialization.md)
- [technique-mfma-pipelining](../techniques/mfma-pipelining.md)
- [kernel-ck-mxfp4-gemm-cdna4](ck-mxfp4-gemm-cdna4.md) — CDNA 4 successor with block-scaled MFMA
- [blog-rocm-cdna4-gemm-kernels](../../sources/blogs/rocm-cdna4-gemm-kernels.md) — 9-stage walk from 1.15 to 2680 TFLOPS on CDNA 4 (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
