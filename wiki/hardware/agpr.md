---
id: hw-agpr
title: "AGPR — Accumulation VGPR Pool"
type: hardware
architectures: [cdna3, cdna4]
tags: [agpr, mfma]
confidence: source-reported
related: [hw-mfma, hw-mfma-scale-f8f6f4, technique-mfma-pipelining, technique-register-budgeting]
sources: [doc-amd-cdna3-whitepaper, doc-amd-cdna3-isa, blog-rocm-mfma-tutorial]
aliases: [AGPR, "Accumulation VGPR", "acc VGPR", "accumulation register"]
---

# AGPR — Accumulation VGPR Pool

## Overview

AGPRs (Accumulation VGPRs) are a second 32-bit register file class on CDNA, used exclusively as the destination of MFMA accumulators and the source/destination of `v_accvgpr_read_b32` / `v_accvgpr_write_b32`. The AGPR pool is *shared* with the regular VGPR pool.

Sizing is **per SIMD per lane**. Each CU has 4 SIMD16 units; a wave64 instruction issues over 4 cycles per SIMD, and each lane has its own VGPR/AGPR file. The figures below refer to one lane's view of one SIMD's register file:

- CDNA 3 provides 512 VGPR-equivalent 32-bit registers per SIMD per lane.
- On gfx90a the AGPR-addressable cap is 256 per wave; on gfx942 (CDNA 3) and gfx950 (CDNA 4) the VGPR/AGPR split is flexible, totaling 512 entries per SIMD per lane, with up to 256 addressable as AGPRs at single-wave-per-SIMD occupancy.
- The compiler partitions the per-SIMD-per-lane budget between "plain" VGPRs and AGPRs at compile time; AGPRs spent on MFMA accumulators directly reduce the per-SIMD-per-lane VGPR headroom available for staging, loop-carried values, and software pipelining.

## Why a Separate Register Class

MFMA instructions write multi-cycle accumulator results back into AGPRs while the wave continues issuing independent VGPR-only instructions (loads, swizzles, scalar ops). The separation lets the compiler treat MFMA latency as independent of VGPR-side scheduling — but the shared per-SIMD-per-lane budget (512 32-bit registers per SIMD per lane on CDNA 3, of which up to 256 are AGPR-addressable) means heavy AGPR use directly reduces VGPR headroom.

## Allocation Examples

| MFMA shape | Accumulators / lane | AGPRs / lane |
|------------|---------------------|--------------|
| `f32_16x16x16_f16` | 4 FP32 | 4 |
| `f32_32x32x8_f16` | 16 FP32 | 16 |
| `f32_32x32x16_fp8_fp8` | 16 FP32 | 16 |
| `f32_16x16x32_fp8_fp8` | 4 FP32 | 4 |
| `f64_16x16x4_f64` | 4 FP64 | 8 |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 16 FP32 | 16 |

A typical CDNA 3 GEMM with 2×2 wave-tile of 32×32×16 FP8 MFMAs needs 4 × 16 = 64 AGPRs per SIMD per lane just for accumulators, leaving 448 VGPRs per SIMD per lane for everything else (out of the 512-per-SIMD-per-lane budget). Note: the 448 figure assumes single-wave-per-SIMD occupancy; at 2 waves/SIMD the per-wave budget halves to 256 entries, so the same accumulator footprint leaves only ~192 VGPRs/wave for staging.

## VGPR / AGPR Budgeting

The per-lane 512-register cap drives the central tradeoff:

- **Larger MFMA shapes** (32×32) → higher peak FLOPS but more AGPRs → less VGPR for pipeline staging.
- **Smaller MFMA shapes** (16×16) → fewer AGPRs but more issues per K-step → more compiler-pressure on the scheduler.

In practice, FP16/BF16 GEMMs default to 16×16×16 MFMAs (4 AGPRs/lane) to leave room for 4-stage software pipelines; FP8 GEMMs can afford 32×32×16 (16 AGPRs/lane) because the operand-bytes-per-MFMA ratio is more favorable.

## AGPR ↔ VGPR Copies

```asm
; Copy AGPR → VGPR (e.g. for store epilogue)
v_accvgpr_read_b32 v0, a0
; Copy VGPR → AGPR (e.g. for initial accumulator zero)
v_accvgpr_write_b32 a0, v0
```

The LLVM AMDGPU backend inserts these automatically when the live range of an MFMA accumulator overflows or when the epilogue needs to store accumulator values via a VGPR-source instruction.

## CDNA 4 Notes

CDNA 4 keeps the same per-lane 512-register total, and the block-scaled MFMA family writes the same 16 AGPRs/lane footprint for `v_mfma_scale_f32_32x32x64_f8f6f4` as the existing FP8 MFMAs do. The MXFP4 path therefore inherits the same register-budgeting tradeoffs.

## Caveats

- Trying to "save VGPRs" by aggressively spilling to AGPRs only works if those AGPRs are not already needed for MFMA accumulators in flight — the LLVM allocator will refuse and spill to scratch instead.
- AGPR copies (`v_accvgpr_*`) cost cycles; minimize them by keeping accumulator scope tight (init in AGPR, accumulate in AGPR, copy to VGPR once in the epilogue).

## See Also

- [hw-mfma](mfma.md) — accumulator destination semantics
- [technique-mfma-pipelining](../techniques/mfma-pipelining.md) — interleaving MFMA against loads
- [technique-register-budgeting](../techniques/register-budgeting.md) — occupancy vs ILP trade-offs
