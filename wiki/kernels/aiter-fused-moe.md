---
id: kernel-aiter-fused-moe
title: "AITER Fused MoE Kernel on CDNA 3 / CDNA 4"
type: kernel
architectures: [cdna3, cdna4]
tags: [moe, fused-moe, grouped-gemm, gemm, mfma, lds, buffer-load-lds, wave-specialization, mfma-pipelining, xcd-aware-scheduling, fine-grained-quantization]
confidence: source-reported
reproducibility: snippet
kernel_types: [moe, grouped-gemm]
languages: [composable-kernel, hip]
related: [hw-mfma, hw-mfma-scale-f8f6f4, hw-xcd, hw-infinity-cache, technique-wave-specialization, technique-mfma-pipelining, technique-xcd-aware-scheduling, technique-direct-to-lds, technique-fine-grained-quantization, lang-composable-kernel]
sources: [pr-aiter-297, pr-aiter-3117, pr-aiter-2911, blog-rocm-aiter-fused-moe, blog-rocm-mlperf-inference-v51, doc-amd-aiter-readme, doc-amd-composable-kernel]
performance_claims:
  - gpu: MI300X
    dtype: fp8_e4m3
    shape: "E=128, top-k=8, tokens=4096, hidden=4096, inter=14336"
    metric: "tokens/sec"
    value: 14200
    utilization: "~2.1x vs unfused vLLM baseline"
    source_id: pr-aiter-297
    evidence_basis: source-reported
  - gpu: MI355X
    dtype: mxfp4
    shape: "E=128, top-k=8, tokens=4096, hidden=4096, inter=14336"
    metric: "tokens/sec"
    value: 32400
    utilization: "~2.3x over fp8 baseline"
    source_id: pr-aiter-2911
    evidence_basis: source-reported
aliases: ["AITER fused MoE", "ROCm fused MoE", "AMD fused MoE GEMM"]
---

# AITER Fused MoE Kernel on CDNA 3 / CDNA 4

## Overview

AITER's fused MoE kernel implements `(gating + top-k + permute) → grouped_gemm_w13 → SiLU·mul → grouped_gemm_w2 → unpermute → reduce` as one launch on MI300X / MI355X. It is the production fused-MoE used by SGLang and vLLM-AMD for DeepSeek-V3, Mixtral, and Qwen-MoE inference.

The kernel demonstrates three AMD-specific techniques in combination:

1. Wave-specialized direct-to-LDS pipelining of the two grouped GEMMs.
2. XCD-aware tile mapping so each expert's K-stationary tiles stay on one chiplet.
3. Idle producer-wave reuse: the producer waves of the W13 grouped GEMM also compute the top-k routing for the *next* batch of tokens, eliminating a separate kernel launch.

## Architecture

```
   ┌─────────────────────────────────────────────────────────────┐
   │  Persistent kernel — one work-group per CU                  │
   │                                                              │
   │  Wave 0..1 (producer):                                       │
   │    - buffer_load_dwordx4_lds(A_perm, B_w13, stage)           │
   │    - in idle cycles: compute top-k for next-batch tokens     │
   │    - s_waitcnt vmcnt(0); s_barrier                           │
   │                                                              │
   │  Wave 2..3 (consumer):                                       │
   │    - ds_read_b128 + v_mfma_f32_32x32x16_fp8                  │
   │    - inline SiLU·mul fusion between W13 and W2               │
   │    - ds_read_b128 + v_mfma for W2                            │
   │                                                              │
   │  Epilogue:                                                   │
   │    - per-token, per-expert unpermute via atomic_add to       │
   │      a workspace, then warp-shuffle reduce                   │
   └─────────────────────────────────────────────────────────────┘
```

## Kernel Skeleton

```cpp
#include "aiter/moe/fused_moe_v3.hpp"

aiter::FusedMoeConfig cfg{
    .num_experts    = 128,
    .top_k          = 8,
    .hidden_size    = 4096,
    .intermediate   = 14336,
    .dtype_a        = aiter::DType::FP8_E4M3,   // or MXFP4 on MI355X
    .dtype_w        = aiter::DType::FP8_E4M3,   // or MXFP4
    .quant_w13      = aiter::QuantScheme::PER_EXPERT_PER_CHANNEL,
    .quant_w2       = aiter::QuantScheme::PER_EXPERT_PER_CHANNEL,
    .activation     = aiter::Activation::SILU_AND_MUL,
    .pipeline       = aiter::Pipeline::V3_WAVE_SPECIALIZED,
    .tile_scheduler = aiter::TileScheduler::XCD_AWARE_PERSISTENT,
};

aiter::fused_moe(
    /*out=*/   output,            // [num_tokens, hidden_size]
    /*x=*/     hidden_states,     // [num_tokens, hidden_size]
    /*w13=*/   w13_weight,        // [num_experts, intermediate*2, hidden_size]
    /*w2=*/    w2_weight,         // [num_experts, hidden_size, intermediate]
    /*router=*/router_logits,     // [num_tokens, num_experts]
    /*sa13=*/  scales_a_w13,      // per-token quant scales
    /*sa2=*/   scales_a_w2,
    /*sw13=*/  scales_w13,        // per-channel weight scales
    /*sw2=*/   scales_w2,
    cfg, stream);
```

The C++ entry point dispatches to one of ~20 specialized kernel template instantiations (FP8 vs MXFP4, top-k 1/2/4/8, hidden-size buckets). All instantiations share the V3 pipeline structure.

## Why It Wins

| Optimization | Standalone gain | Cumulative |
|--------------|------------------|------------|
| Baseline (vLLM unfused) | — | 1.00× |
| + grouped-GEMM (single launch per layer) | +25% | 1.25× |
| + V3 wave-specialized pipeline | +35% | 1.69× |
| + XCD-aware persistent tile mapping | +20% | 2.03× |
| + idle-producer top-k overlap | +5% | 2.13× (MI300X) |

The 2.1× headline is over a well-tuned vLLM baseline — over a naive PyTorch implementation it's ~6×.

## Idle-Producer Trick

CK-Tile V3 producer waves with `buffer_load_dwordx4_lds` (128 b/issue on CDNA 4) drain their issue list before the consumer drains the stage. AITER fills these cycles by having the producer wave run the top-k routing computation for the *next* batch of tokens (sort + scatter-index build). This is enabled by the wave-id check:

```cpp
if (wave_id < 2) {
    direct_to_lds_load(...);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");

    // Idle cycles before s_barrier: compute next-batch routing
    if (next_batch_routing_pending && lane < 32) {
        compute_topk_chunk(next_router_logits, ...);
    }

    __syncthreads();
}
```

Only worthwhile because top-k routing is small and overlap-friendly. Larger work items (e.g., layer norm) would defeat the producer's primary role.

## CDNA 3 vs CDNA 4 differences

- **CDNA 3 (FP8)**: uses `v_mfma_f32_32x32x16_fp8_fp8` for both grouped GEMMs, 2-stage LDS.
- **CDNA 4 (MXFP4)**: uses `v_mfma_scale_f32_32x32x64_f8f6f4` with UE8M0 per-channel scales, 3-stage LDS, +130% throughput per token for the same shapes.

## Caveats

- Per-expert tile counts are imbalanced; the persistent-kernel scheduler can leave 1-2 CUs idle on the long tail of the heaviest expert. AITER mitigates with a work-stealing tail handler but doesn't eliminate it.
- The fused activation must be SiLU·mul; ReLU/GELU variants need a different template instantiation.
- `top_k > 8` falls back to a non-fused path because the per-token expert-list metadata exceeds the LDS budget reserved for routing.

## See Also

- [pr-aiter-297](../../sources/prs/aiter/PR-297.md) — fmoe heuristic / fused MoE (real 2× win)
- [pr-aiter-3117](../../sources/prs/aiter/PR-3117.md) — MXFP4 fused-MoE stage2 on gfx950
- [pr-aiter-2911](../../sources/prs/aiter/PR-2911.md) — MXFP4 fused-MoE path for CDNA 4
- [blog-rocm-aiter-fused-moe](../../sources/blogs/rocm-aiter-fused-moe.md) — measured wins
- [blog-rocm-mlperf-inference-v51](../../sources/blogs/rocm-mlperf-inference-v51.md) — MXFP4 GEMMs ≈ 62% of MLPerf Llama2-70B cost; MI355X 2.7× over MI325X (https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html)
- [kernel-fused-moe](fused-moe.md) — NVIDIA Blackwell counterpart
- [kernel-ck-mxfp4-gemm-cdna4](ck-mxfp4-gemm-cdna4.md) — underlying GEMM pipeline
