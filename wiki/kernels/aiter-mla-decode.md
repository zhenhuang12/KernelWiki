---
id: kernel-aiter-mla-decode
title: "AITER MLA Decode Kernel on CDNA 3 / CDNA 4"
type: kernel
architectures: [cdna3, cdna4]
tags: [mla, attention, decode, paged-attention, mfma, lds, buffer-load-lds, wave-specialization, fine-grained-quantization]
confidence: source-reported
reproducibility: snippet
kernel_types: [mla, attention, decode]
languages: [composable-kernel, hip, amdgcn-asm]
related: [hw-mfma, hw-buffer-load-lds, hw-lds, hw-agpr, technique-wave-specialization, technique-mfma-pipelining, technique-direct-to-lds, technique-lds-swizzling, lang-composable-kernel, lang-amdgcn-asm, kernel-flashmla]
sources: [pr-aiter-3128, pr-aiter-2917, doc-amd-aiter-readme, doc-amd-cdna3-isa, blog-rocm-ck-flash-attention]
performance_claims:
  - gpu: MI300X
    dtype: bf16
    shape: "batch=128, seqlen=8192, heads=128, head_dim=512+64"
    metric: "TB/s effective"
    value: 4.2
    utilization: "~80% of MI300X HBM3 5.3 TB/s"
    source_id: pr-aiter-3128
    evidence_basis: source-reported
  - gpu: MI300X
    dtype: fp8_e4m3
    shape: "batch=128, seqlen=8192, heads=128, head_dim=512+64"
    metric: "TB/s effective"
    value: 4.6
    utilization: "~87% of MI300X HBM3"
    source_id: pr-aiter-3128
    evidence_basis: source-reported
aliases: ["AITER MLA decode", "ROCm MLA decode", "AMD FlashMLA decode"]
---

# AITER MLA Decode Kernel on CDNA 3 / CDNA 4

## Overview

AITER's MLA decode kernel is the ROCm production decode-path for DeepSeek-V2/V3 multi-head latent attention on MI300X (with CDNA 4 forward-compatibility). It implements the latent-projection variant where the KV cache stores compressed latents (head_dim=512 + 64 rope) instead of full K and V, and reconstructs Q·K and attention·V on-chip.

The kernel reaches ~80-87% of MI300X HBM bandwidth on long-context decode, matching FlashMLA on H800 for the same shape. The HBM-bound regime means MFMA pipelining matters less than (a) careful paged-KV layout, (b) avoiding `ds_read` bank conflicts, and (c) the QK fence pattern that lets the AMDGPU LLVM scheduler interleave the two MFMA groups.

## Why MLA Decode is HBM-bound

For batch=128, seq=8192, head_dim=512:

- KV cache traffic: `128 × 8192 × 512 × 2 (K+V latents) × 2 B = 2.0 GB / decode step` (BF16-element accounting; the FP8 KV path above halves this to ~1 GB).
- MFMA compute: `128 × 128 × 8192 × 512 × 2 = 137 GFLOPs / decode step` (this counts only the Q·K phase; the attn·V phase doubles the work, so total ≈ 274 GFLOPs and AI ≈ 137 FLOPs/byte against the same 2 GB traffic).
- Arithmetic intensity ≈ 137 GFLOPs / 2.0 GB ≈ 68.5 FLOPs/byte (Q·K only), or ≈ 274 GFLOPs / 2.0 GB ≈ 137 FLOPs/byte (Q·K + attn·V) → HBM bound on MI300X either way. Roofline ridge points (BF16-specific analysis): FP8 = 2614.9 / 5.3 ≈ 493 FLOPs/byte; BF16 = 1307.4 / 5.3 ≈ 246 FLOPs/byte. MLA decode at 68-137 FLOPs/byte sits well below both — memory-bound.

So the win comes from *not wasting HBM bandwidth* — paged-KV access pattern, coalesced loads, and avoiding redundant K-cache reads across Q-tiles for the same sequence.

## Kernel Skeleton

```cpp
#include "aiter/attention/mla_decode.hpp"

aiter::MLADecodeConfig cfg{
    .num_heads     = 128,
    .head_dim_qk   = 576,    // 512 latent + 64 rope
    .head_dim_v    = 512,
    .kv_lora_rank  = 512,
    .qk_rope_dim   = 64,
    .dtype_kv      = aiter::DType::FP8_E4M3,
    .page_size     = 64,     // tokens per page in the KV cache
    .pipeline      = aiter::Pipeline::V3_WAVE_SPECIALIZED,
};

aiter::mla_decode(
    /*out=*/         out,                  // [batch, num_heads, head_dim_v]
    /*q=*/           query,                // [batch, num_heads, head_dim_qk]
    /*kv_cache=*/    kv_cache,             // [num_pages, page_size, head_dim_qk]
    /*kv_scales=*/   kv_scales,            // per-token fp8 dequant scales
    /*block_table=*/ page_table,           // [batch, max_pages]
    /*seq_lens=*/    seq_lens,             // [batch]
    /*sm_scale=*/    sm_scale,
    cfg, stream);
```

## QK Fence Pattern

The MLA decode inner loop alternates two MFMA chains: one computing `Q·Kᵀ` (head_dim_qk = 576 → 9 MMAs of 64) and one computing `attention·V` (head_dim_v = 512 → 8 MMAs of 64). If the LLVM scheduler interleaves them naively, the AGPR accumulators for QK and PV collide and one spills to VGPR. AITER pins the boundary with an inline-asm `s_waitcnt lgkmcnt(0)` fence:

```cpp
// Inside one KV-page iteration (consumer wave).
// kk advances 64 per iteration = 4 MFMA K-steps (4 × MFMA_K=16); the unrolled
// body issues 4 back-to-back mfma_f32_32x32x16 ops over the 64-element ds_read
// payload. Loop iteration count abbreviated for readability.
#pragma unroll
for (int kk = 0; kk < 576; kk += 64) {
    auto q  = ds_read_b128(smem_q + xor_swizzle(head, kk));
    auto k  = ds_read_b128(smem_k[stage] + xor_swizzle(kk, tok));
    qk_acc = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(q, k, qk_acc, 0, 0, 0);
}

// Online softmax (rescale, max, exp) operates on qk_acc
softmax_step(qk_acc, m_state, l_state);

// Fence: drain ds_read counter before issuing PV reads to prevent overlap with
// the QK ds_read chain, which would otherwise force AGPR/VGPR spill.
asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
__builtin_amdgcn_sched_barrier(0x0);

#pragma unroll
for (int kk = 0; kk < 512; kk += 64) {
    auto p  = qk_to_pv_lds_read(qk_acc, kk);
    auto v  = ds_read_b128(smem_v[stage] + xor_swizzle(kk, head_v));
    pv_acc  = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(p, v, pv_acc, 0, 0, 0);
}
```

The fence is documented in `aiter/csrc/include/attention/mla_decode_impl.hpp` and is the single highest-leverage line in the kernel (~12% win when present).

## Paged-KV Layout

Pages are `[page_size=64, head_dim_qk=576]` fp8 with one `fp32` scale per token at the end of each page. Two reasons:

1. Per-page scales let the scale read coalesce with the page load (one `buffer_load_dwordx4_lds` covers both).
2. The 64-token page matches one consumer-wave's K-stage perfectly, so the `block_table` lookup happens once per K-stage instead of per token.

The page-table lookup uses scalar memory (`s_load_dwordx4` of the block_table entry) — see [lang-amdgcn-asm](../languages/amdgcn-asm.md) for SMEM constraint usage.

## XCD-Aware not Applied

MLA decode with paged KV scatters K/V reads across pages anyway, so the L2 hit rate is already low — XCD-aware tile mapping helps only marginally (~2%) for this kernel. AITER skips it.

## When to Use

- DeepSeek-V2 / V3 / R1 decode on MI300X with batch ≥ 16.
- Any MLA variant where head_dim_v ≥ 256 and the KV cache fits in HBM but not L2.
- For prefill, use the AITER MLA prefill kernel (separate template instantiation) which is compute-bound and uses XCD-aware mapping.

## Caveats

- The QK fence pattern is sensitive to ROCm version — verified on ROCm 6.0/6.1; the LLVM 19 update in ROCm 6.2 changed scheduler heuristics and AITER had to retune.
- `head_dim_qk > 576` (uncommon) requires a different template instantiation with deeper LDS staging — the kernel currently caps at 576.
- Batch 1 falls off bandwidth — there's not enough Q-tile concurrency to hide HBM latency. Use the AITER MLA single-query variant.

## See Also

- [pr-aiter-3128](../../sources/prs/aiter/PR-3128.md) — upstream MLA decode PR
- [pr-aiter-2917](../../sources/prs/aiter/PR-2917.md) — companion MLA work
- [kernel-flashmla](flashmla.md) — NVIDIA SM100 / SM90 counterpart
- [technique-wave-specialization](../techniques/wave-specialization.md)
- [lang-amdgcn-asm](../languages/amdgcn-asm.md) — the fence-pattern primitive
