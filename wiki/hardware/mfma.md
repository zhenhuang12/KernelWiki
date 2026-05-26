---
id: hw-mfma
title: "MFMA — AMD CDNA Matrix Cores"
type: hardware
architectures: [cdna3, cdna4]
tags: [mfma, agpr, wavefront-64, fp8, fp6, fp4]
confidence: source-reported
related: [hw-mfma-scale-f8f6f4, hw-agpr, hw-lds, hw-buffer-load-lds, lang-hip, lang-composable-kernel, lang-amdgcn-asm, technique-mfma-pipelining, technique-wave-specialization, migration-cuda-to-hip]
sources: [doc-amd-cdna3-whitepaper, doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-mfma-tutorial]
aliases: [MFMA, "Matrix Fused Multiply-Add", "Matrix Core", "AMD tensor core"]
---

# MFMA — AMD CDNA Matrix Cores

## Overview

MFMA (Matrix Fused Multiply-Add) is the family of `v_mfma_*` instructions implementing CDNA's matrix-multiply tensor cores. Issued at wave (64-lane) granularity, MFMAs read two source matrices from the VGPR file and accumulate into the AGPR (Accumulation VGPR) file. On CDNA 3 they support FP64/FP32/FP16/BF16/INT8/FP8; CDNA 4 adds block-scaled FP4/FP6/FP8 via `v_mfma_scale_f32_*_f8f6f4` (covered separately in [hw-mfma-scale-f8f6f4](mfma-scale-f8f6f4.md)).

## CDNA 3 vs NVIDIA Hopper Tensor Cores

| Aspect | NVIDIA Hopper wgmma | AMD CDNA 3 MFMA |
|--------|---------------------|-----------------|
| Issue unit | Warpgroup (4 warps / 128 threads) | Single wave (64 lanes) |
| Operand source | A from registers/SMEM, B from SMEM | Both A and B from VGPR |
| Accumulator | Registers (shared across warpgroup) | AGPR (per-wave, 4 AGPR/lane typical) |
| Async | Yes (commit/wait groups) | No (synchronous; pipeline via instruction-level overlap) |
| Operand load | ldmatrix | Caller responsibility (`ds_read_b128` from LDS into VGPR) |

The single-wave, VGPR-sourced model means AMD kernels stage operands LDS → VGPR → MFMA, while NVIDIA goes SMEM → wgmma directly. This is why AMD wave-specialized kernels devote one wave-pair to LDS loads and another to MFMA issue — there is no "single thread issues the MMA" shortcut as on Blackwell.

## MFMA Shape Catalogue (CDNA 3, partial)

| Instruction | M×N×K | dtype |
|-------------|-------|-------|
| `v_mfma_f32_16x16x16_f16` | 16×16×16 | FP16 → FP32 |
| `v_mfma_f32_32x32x8_f16` | 32×32×8 | FP16 → FP32 |
| `v_mfma_f32_16x16x16_bf16` | 16×16×16 | BF16 → FP32 |
| `v_mfma_f32_16x16x32_fp8_fp8` | 16×16×32 | FP8 (E4M3) → FP32 |
| `v_mfma_f32_32x32x16_fp8_fp8` | 32×32×16 | FP8 → FP32 |
| `v_mfma_i32_16x16x32_i8` | 16×16×32 | INT8 → INT32 |
| `v_mfma_f64_16x16x4_f64` | 16×16×4 | FP64 → FP64 |

CDNA 4 adds wider-K variants: `v_mfma_f32_16x16x32_f16` (2× CDNA 3 K), `v_mfma_f32_32x32x16_bf16`, plus the block-scaled family.

## Three Programming Layers

```cpp
// 1. AMDGCN inline asm — full control, used in CK inner loops
asm volatile(
    "v_mfma_f32_16x16x32_fp8_fp8 v[%0:%0+3], v[%1:%1+1], v[%2:%2+1], v[%3:%3+3]"
    : "=v"(acc_lo)
    : "v"(a_lo), "v"(b_lo), "v"(acc_lo));

// 2. Clang intrinsic — one-to-one with the instruction
acc = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);

// 3. rocWMMA — portable WMMA-style API
using FragA = rocwmma::fragment<rocwmma::matrix_a, 16, 16, 32, rocwmma::float8_t, rocwmma::row_major>;
FragA a_frag;
rocwmma::load_matrix_sync(a_frag, a_ptr, lda);
rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

CK-Tile and AITER use layer 2 (intrinsics) almost exclusively; rocBLAS uses layer 3 for portability across gfx generations.

## AGPR Allocation

A `v_mfma_f32_16x16x16_f16` writes 4 FP32 accumulators per lane → 4 AGPRs/lane.  
A `v_mfma_f32_32x32x8_f16` writes 16 FP32 accumulators per lane → 16 AGPRs/lane.

The VGPR + AGPR pool is **512 32-bit registers per SIMD per lane shared**. Over-allocating AGPRs for a 32×32 tile reduces the VGPR headroom for software-pipelined operand loads, which is the single largest occupancy lever on AMD. The LLVM AMDGPU backend inserts `v_accvgpr_read_b32` / `v_accvgpr_write_b32` copies to bridge AGPR↔VGPR when needed.

## Typical Wave Body

```cpp
// CDNA 3 BF16 GEMM inner loop (16 K-stages, 16x16x16 tile)
#pragma unroll
for (int k = 0; k < K_per_block; k += 16) {
    // Load A/B tile from LDS into VGPR
    auto a_frag = lds_load_a_tile(k);  // 4 VGPRs / lane
    auto b_frag = lds_load_b_tile(k);  // 4 VGPRs / lane
    // MFMA accumulate
    acc = __builtin_amdgcn_mfma_f32_16x16x16_bf16(a_frag, b_frag, acc, 0, 0, 0);
}
// acc lives in AGPRs; epilogue copies to VGPR before store
```

## Caveats

- MFMA is synchronous — there's no async-commit / wait-group concept. Pipelining is purely via the compiler scheduler interleaving `ds_read`, MFMA, and global stores; place `__builtin_amdgcn_sched_barrier(mask)` to fence schedule regions you care about.
- 32×32 MFMAs deliver higher peak throughput but consume 16 AGPRs/lane; under heavy register pressure 16×16 MFMAs are often net faster.
- FP8 MFMAs require both operands in FP8 format; mixed FP8/BF16 must be staged through a conversion in VGPR.

## See Also

- [hw-mfma-scale-f8f6f4](mfma-scale-f8f6f4.md) — CDNA 4 block-scaled MFMA family
- [hw-agpr](agpr.md) — Accumulation VGPR pool
- [lang-amdgcn-asm](../languages/amdgcn-asm.md) — inline-asm wrapping
- [technique-mfma-pipelining](../techniques/mfma-pipelining.md) — software pipelining around MFMA
