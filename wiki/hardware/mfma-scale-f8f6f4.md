---
id: hw-mfma-scale-f8f6f4
title: "MFMA Scale F8F6F4 — CDNA 4 Block-Scaled Matrix Cores"
type: hardware
architectures: [cdna4]
tags: [mfma, mfma-scale-f8f6f4, mxfp4, mxfp6, mxfp8, block-scale]
confidence: source-reported
related: [hw-mfma, hw-mxfp-cdna4, hw-nvfp4, technique-fine-grained-quantization, kernel-ck-mxfp4-gemm-cdna4, lang-composable-kernel]
sources: [doc-amd-cdna4-whitepaper, doc-amd-cdna4-isa, blog-rocm-mi355-mxfp4-launch, pr-composable-kernel-3098, pr-aiter-2911]
aliases: ["v_mfma_scale_f32_16x16x128_f8f6f4", "v_mfma_scale_f32_32x32x64_f8f6f4", "scaled MFMA", "block-scaled MFMA", "MXFP MFMA"]
---

# MFMA Scale F8F6F4 — CDNA 4 Block-Scaled Matrix Cores

## Overview

The `v_mfma_scale_f32_*_f8f6f4` instruction family on CDNA 4 (gfx950) is AMD's hardware-native block-scaled MFMA — the direct analogue of NVIDIA Blackwell's `tcgen05.mma.kind::mxf4nvf4` / `kind::mxf8f6f4`. It accepts mixed-precision operands (FP8 E4M3/E5M2, FP6 E2M3/E3M2, FP4 E2M1) and consumes a UE8M0 (8-bit, unsigned, all-exponent) per-block scale stream with 32-element block granularity.

## Instructions

```
v_mfma_scale_f32_16x16x128_f8f6f4 acc, a, b, scaleA, scaleB cbsz blgp abid
v_mfma_scale_f32_32x32x64_f8f6f4  acc, a, b, scaleA, scaleB cbsz blgp abid
```

Encoding fields:

| Field | Bits | Meaning |
|-------|------|---------|
| `cbsz` | 3 | A-operand sub-format (0=FP8 E4M3, 1=FP8 E5M2, 2=FP6 E2M3, 3=FP6 E3M2, 4=FP4 E2M1) |
| `blgp` | 3 | B-operand sub-format (same encoding) |
| `abid` | 4 | Scale broadcast control within 32-elem block |
| `scaleA`, `scaleB` | UE8M0 dwords | Per-block scales, interleaved by the assembler |

> **Verify against AMD CDNA4 ISA reference.** The specific sub-format codes
> listed above (E4M3=0, E5M2=1, E2M3=2, E3M2=3, E2M1=4) should be cross-checked
> against the official gfx950 ISA document before relying on them in production
> codegen.

## Block-Scaling Semantics

A 32-element block of operand values is multiplied by `2^(scale - 127)` (UE8M0 bias) before accumulation. This is identical to the OCP MX specification and matches NVFP4's scaling on Blackwell. The kernel-author responsibility is to keep the scale stream cache-coherent with the operand stream — typically the same `buffer_load_dwordx4_lds` issues fetch both.

Note: UE8M0 reserves `0xFF` as NaN per the OCP MX specification. UE8M0 has no zero encoding — `0x00` decodes to `2^-127`, not zero.

## Idiomatic Use

The signature below is **illustrative**, not a literal Clang declaration: the exact
parameter order and the position of `cbsz` / `blgp` / scale-select operands for
`__builtin_amdgcn_mfma_scale_f32_*_f8f6f4` varies across Clang 18 / 19 / 20
release lines. Always verify against the AMDGPU intrinsic reference for your
target Clang version: <https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html>.

```cpp
// CK-Tile MXFP4 GEMM inner loop (CDNA 4) — illustrative pseudo-intrinsic form
// 32x32x64 MFMA, MXFP4 operands, FP32 accumulator
auto a_frag    = lds_load_mxfp4_a_tile(k);   // 4 VGPRs / lane
auto b_frag    = lds_load_mxfp4_b_tile(k);   // 4 VGPRs / lane
auto a_scales  = lds_load_ue8m0_scales_a();  // 1 VGPR / lane
auto b_scales  = lds_load_ue8m0_scales_b();  // 1 VGPR / lane

// Illustrative — argument order MUST be cross-checked against AMDGPUUsage
// for the Clang version you build with.
acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    a_frag, b_frag, acc, a_scales, b_scales,
    /*cbsz=*/4 /*FP4 E2M1*/, /*abid=*/0, /*blgp=*/4 /*FP4 E2M1*/);
```

For W4A8 (MXFP4 weights, FP8 activations), set `cbsz=4` and `blgp=0` (subject to
the same Clang-version caveat above).

## Comparison with Blackwell

| Aspect | Blackwell tcgen05.mma.kind::mxf4nvf4 | CDNA 4 v_mfma_scale_*_f8f6f4 |
|--------|----------------------------------------|------------------------------|
| Issue | Single thread | Single wave (64 lanes) |
| Accumulator | TMEM (256 KB CTA-visible) | AGPR (per-wave) |
| Block size | 32 elements | 32 elements |
| Scale dtype | UE8M0 (NVFP4) or UE4M3 (MXFP4) | UE8M0 only |
| Block layout | TMEM-resident scale tile | VGPR-resident, refilled each K-step |
| Mixed precision | Yes (via `kind::mxf8f6f4`) | Yes (`cbsz` ≠ `blgp`) |

Both architectures converge on the OCP MX format. Quantization recipes (Hadamard rotations, calibration, scale clipping) developed for NVFP4 transfer with no math changes.

## Caveats

- Only `f8f6f4` mixed-precision is hardware-supported; pure-MXFP8 GEMMs without scaling use the regular `v_mfma_f32_*_fp8_fp8` family.
- The scale stream layout (one UE8M0 byte per 32-element operand block) must be packed in the operand-major direction; software-side packing bugs are a common failure mode.
- `abid` controls how the scale is broadcast across the 32-element block — typically 0 (uniform) for standard MXFP, but available for asymmetric schemes.

## See Also

- [hw-mxfp-cdna4](mxfp-cdna4.md) — MXFP4/6/8 dtype storage layout
- [hw-nvfp4](nvfp4.md) — NVIDIA equivalent
- [kernel-ck-mxfp4-gemm-cdna4](../kernels/ck-mxfp4-gemm-cdna4.md) — production reference
- [technique-fine-grained-quantization](../techniques/fine-grained-quantization.md) — block-scaling theory
