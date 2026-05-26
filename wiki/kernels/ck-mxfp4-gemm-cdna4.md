---
id: kernel-ck-mxfp4-gemm-cdna4
title: "Composable Kernel MXFP4 GEMM on CDNA 4 (MI355X)"
type: kernel
architectures: [cdna4]
tags: [gemm, mxfp4, mfma-scale-f8f6f4, lds, buffer-load-lds, wave-specialization, mfma-pipelining, composable-kernel, fine-grained-quantization]
confidence: source-reported
reproducibility: snippet
kernel_types: [gemm]
languages: [composable-kernel, hip, amdgcn-asm]
related: [hw-mfma-scale-f8f6f4, hw-mxfp-cdna4, hw-buffer-load-lds, hw-lds, technique-wave-specialization, technique-mfma-pipelining, technique-direct-to-lds, technique-lds-swizzling, technique-xcd-aware-scheduling, technique-fine-grained-quantization, lang-composable-kernel]
sources: [pr-composable-kernel-3098, pr-composable-kernel-2488, blog-rocm-mi355-mxfp4-launch, doc-amd-cdna4-isa, doc-amd-composable-kernel]
performance_claims:
  - gpu: MI355X
    dtype: mxfp4
    shape: "M=4096, N=4096, K=4096"
    metric: TFLOPS
    value: 4350
    utilization: "~43% of MI355X dense MXFP4 peak (~10.07 PFLOPS)"
    source_id: blog-rocm-mi355-mxfp4-launch
    evidence_basis: source-reported
  - gpu: MI355X
    dtype: mxfp4
    shape: "M=8192, N=8192, K=8192"
    metric: TFLOPS
    value: 4720
    utilization: "~47% of MI355X dense MXFP4 peak (~10.07 PFLOPS)"
    source_id: blog-rocm-mi355-mxfp4-launch
    evidence_basis: source-reported
aliases: ["CK-Tile MXFP4 GEMM", "MI355X MXFP4 GEMM", "CDNA 4 block-scaled GEMM"]
---

# Composable Kernel MXFP4 GEMM on CDNA 4 (MI355X)

## Overview

CDNA 4 (gfx950) adds `v_mfma_scale_f32_32x32x64_f8f6f4` and `v_mfma_scale_f32_16x16x128_f8f6f4` — block-scaled MFMAs that consume two UE8M0 scale streams (cbsz for A, blgp for B) alongside the f8/f6/f4 operand streams. CK-Tile's MXFP4 GEMM wires those into the V3 wave-specialized pipeline plus a third producer stream for the scales, reaching ~4.7 PFLOPS on a 8192³ MXFP4 GEMM (about 47% of MI355X dense MXFP4 peak ~10.07 PFLOPS; ~23% of the ~20.1 PFLOPS sparse peak).

This is the AMD analogue of NVIDIA's NVFP4 GEMM with tcgen05.mma block scaling (see [kernel-nvfp4-gemm](nvfp4-gemm.md)).

## Block Shape

| Parameter | Value | Reason |
|-----------|-------|--------|
| BLOCK_M × BLOCK_N | 256 × 256 | Matches consumer wave count |
| BLOCK_K | 128 | One 32×32×64 MFMA covers K=64 → 2 MMAs per K-step |
| MFMA shape | 32×32×64 f8f6f4 | 16 AGPRs/lane per MMA |
| Waves / WG | 4 (256 threads) | 2 producers (A/B + scales) + 2 consumers |
| Stages | 3 | CDNA 4's 160 KB LDS allows 3 stages of the larger tile |
| Scale block | 32 elements / UE8M0 | OCP MX standard — see [hw-mxfp-cdna4](../hardware/mxfp-cdna4.md) |

## Pipeline Skeleton

```cpp
#include "ck_tile/ops/gemm/pipeline/block_gemm_pipeline_mxfp4.hpp"

using GemmShape  = ck_tile::TileGemmShape<256, 256, 128>;
using MfmaShape  = ck_tile::sequence<32, 32, 64>;
using Pipeline   = ck_tile::BlockGemmPipelineMxFp4<
    ck_tile::mxfp4_t,     // A operand dtype
    ck_tile::mxfp4_t,     // B operand dtype
    ck_tile::ue8m0_t,     // shared scale dtype (1 byte / 32 elem block)
    float,                // accumulator
    GemmShape, MfmaShape,
    /*num_warp_groups=*/2,
    /*pipeline_stages=*/3>;

__global__ void __launch_bounds__(256)
ck_mxfp4_gemm(
    const uint8_t*  A_packed,      // 2 mxfp4 per byte
    const uint8_t*  B_packed,
    const uint8_t*  A_scales,      // ue8m0, one per K=32 block per M-row
    const uint8_t*  B_scales,      // ue8m0, one per K=32 block per N-col
    bf16*           C,
    int M, int N, int K)
{
    int logical_bid = ck_tile::xcd_remap(blockIdx.x);
    auto [m_tile, n_tile] = ck_tile::tile_partition(logical_bid, M / 256, N / 256);

    __shared__ ck_tile::SmemStorage<Pipeline> smem;

    auto a_win    = ck_tile::make_tile_window(A_packed, {M, K / 2}, {m_tile * 256, 0});
    auto b_win    = ck_tile::make_tile_window(B_packed, {K / 2, N}, {0, n_tile * 256});
    auto sa_win   = ck_tile::make_scale_window(A_scales, {M, K / 32}, {m_tile * 256, 0});
    auto sb_win   = ck_tile::make_scale_window(B_scales, {K / 32, N}, {0, n_tile * 256});

    auto acc = Pipeline::run(a_win, b_win, sa_win, sb_win, K / 128, smem);

    ck_tile::store_tile(C, {M, N}, {m_tile * 256, n_tile * 256},
                        ck_tile::cast<bf16>(acc));
}
```

## Inner Loop (Consumer Wave)

```cpp
// Two 32x32x64 MFMAs per K-step (BLOCK_K = 128, MFMA_K = 64)
#pragma unroll
for (int kk = 0; kk < 128; kk += 64) {
    auto a   = ds_read_b128(smem_a[stage] + xor_swizzle(row, kk));      // 32 mxfp4
    auto b   = ds_read_b128(smem_b[stage] + xor_swizzle(kk, col));
    auto sa  = ds_read_b32 (smem_sa[stage] + kk / 32);                  // 2 ue8m0
    auto sb  = ds_read_b32 (smem_sb[stage] + kk / 32);

    __builtin_amdgcn_sched_barrier(0x0);
    // Illustrative — verify the intrinsic signature against the target clang
    // version. See hw-mfma-scale-f8f6f4 for the canonical form.
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
              a, b, acc,
              /*cbsz=*/MXFP4, /*blgp=*/MXFP4,
              sa, sb,
              /*sa_sel=*/0, /*sb_sel=*/0);
    __builtin_amdgcn_sched_barrier(0x0);
}
```

The `cbsz`/`blgp` encoding selects the f8/f6/f4 sub-format per operand. For mixed-precision (e.g., MXFP6 weights × MXFP4 activations) the encoding differs but the pipeline structure is identical — see [hw-mfma-scale-f8f6f4](../hardware/mfma-scale-f8f6f4.md).

## Why 3 Stages

CDNA 3 (64 KB LDS) caps this kernel at 2 stages. CDNA 4 (160 KB LDS) plus the 4× wider `buffer_load_dwordx4_lds` per issue means the producer wave's issue rate matches the consumer's MFMA rate at 3 stages, fully hiding HBM latency for K ≥ 2048.

## XCD-Aware + Persistent

MXFP4 has a 4× operand-bandwidth advantage over FP8, so this kernel is compute-bound earlier. The XCD-aware remap is still worth ~15% on the 4096³ case. Combining XCD-aware + persistent kernel (one work-group per CU, processing `tile_count / num_cu` tiles) adds another ~5%.

## When to Use

- MI355X / CDNA 4 inference with MXFP4-quantized weights (the dominant MoE quant scheme as of 2026).
- Mixed-precision GEMM: A in MXFP4, B in MXFP6, accumulate to FP32.
- Any case where you would have used FP8 on MI300X — MXFP4 doubles the compute throughput at near-equivalent accuracy when the scale block is tuned.

## Caveats

- The two scale streams add a third LDS region; budget LDS as `(256·128/2 + 128·256/2 + 256·4 + 4·256) × 3 stages` ≈ 102 KB (per-stage region is `packedA + packedB + scalesA + scalesB`, not doubled).
- UE8M0 scales saturate at very small magnitudes — verify that activation calibration doesn't produce scales below 2⁻¹²⁷.
- Mixed cbsz/blgp formats (MXFP4 × MXFP6) work but the LLVM scheduler sometimes fails to interleave their `ds_read` issue widths optimally — `sched_barrier` annotation is mandatory.

## See Also

- [pr-composable-kernel-3098](../../sources/prs/composable_kernel/PR-3098.md) — upstream MXFP4 GEMM PR
- [pr-composable-kernel-2488](../../sources/prs/composable_kernel/PR-2488.md) — companion mfma-scale enablement
- [hw-mfma-scale-f8f6f4](../hardware/mfma-scale-f8f6f4.md) — instruction reference
- [hw-mxfp-cdna4](../hardware/mxfp-cdna4.md) — OCP MX format
- [kernel-nvfp4-gemm](nvfp4-gemm.md) — NVIDIA Blackwell counterpart
- [kernel-ck-fp8-gemm-cdna3](ck-fp8-gemm-cdna3.md) — CDNA 3 predecessor
