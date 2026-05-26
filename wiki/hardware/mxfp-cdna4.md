---
id: hw-mxfp-cdna4
title: "MXFP4 / MXFP6 / MXFP8 — CDNA 4 OCP Microscaling Formats"
type: hardware
architectures: [cdna4]
tags: [mxfp4, mxfp6, mxfp8, block-scale, fp4, fp6, fp8]
confidence: source-reported
related: [hw-mfma-scale-f8f6f4, hw-nvfp4, technique-fine-grained-quantization, kernel-ck-mxfp4-gemm-cdna4]
sources: [doc-amd-cdna4-whitepaper, doc-amd-cdna4-isa, blog-rocm-mi355-mxfp4-launch, pr-composable-kernel-3098, pr-aiter-2911]
aliases: [MXFP4, MXFP6, MXFP8, "OCP microscaling", "MX FP"]
---

# MXFP4 / MXFP6 / MXFP8 — CDNA 4 OCP Microscaling Formats

## Overview

CDNA 4 (gfx950) implements the OCP Microscaling (MX) format family natively in matrix-core hardware: MXFP4 (FP4 E2M1 + UE8M0 scale), MXFP6 (FP6 E2M3 or E3M2 + UE8M0 scale), MXFP8 (FP8 E4M3 or E5M2 + UE8M0 scale). These are bit-identical to NVIDIA Blackwell's NVFP4 (MXFP4 with UE8M0 scales), so quantization recipes transfer directly across vendors.

## Storage Layout

A 32-element block is stored as:

```
[ data: 32 × {FP4/FP6/FP8 elements} ] [ scale: 1 × UE8M0 byte ]
```

UE8M0 is an 8-bit unsigned, all-exponent format encoding `2^(s - 127)`, with `s=0` reserved for zero. Effective range is `2^-126 … 2^128`, which covers every reasonable per-block scale.

| Format | Element bits | Per-block bytes | Effective bits-per-element |
|--------|--------------|------------------|----------------------------|
| MXFP4 | 4 | 32×4/8 + 1 = 17 | 4.25 |
| MXFP6 | 6 | 32×6/8 + 1 = 25 | 6.25 |
| MXFP8 | 8 | 32×8/8 + 1 = 33 | 8.25 |

The 0.25-bit overhead per element is the cost of block scaling — well-amortized for inference weight matrices.

## How CDNA 4 Hardware Uses Them

The `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` instructions accept:

- Operand A in any of FP8/FP6/FP4 (selected by `cbsz` field).
- Operand B in any of FP8/FP6/FP4 (selected by `blgp` field).
- A UE8M0 scale stream for A and B, one byte per 32-element block.

Operand and scale streams are issued in lock-step via paired `buffer_load_dwordx4_lds` instructions, then fed to MFMA which applies the scale before accumulation. See [hw-mfma-scale-f8f6f4](mfma-scale-f8f6f4.md).

## Comparison with NVIDIA NVFP4

| Aspect | NVIDIA NVFP4 (Blackwell) | AMD MXFP4 (CDNA 4) |
|--------|---------------------------|---------------------|
| Element format | FP4 E2M1 | FP4 E2M1 |
| Block size | 16 (NVFP4-16) or 32 (NVFP4-32) | 32 (OCP standard) |
| Scale dtype | UE4M3 (NVFP4) or UE8M0 (MXFP4) | UE8M0 only |
| Hardware MMA | tcgen05.mma.kind::mxf4nvf4 | v_mfma_scale_f32_*_f8f6f4 |
| Accumulator | TMEM | AGPR |

NVIDIA's NVFP4-16 (16-element blocks) is denser scale-wise; CDNA 4 supports only 32-element blocks, matching the OCP MX spec. Both vendors converge on the same quantization recipe family.

## Practical Notes

- The UE8M0 scale value `s` is sampled per 32-element block; the block boundary is the contiguous-K direction for operand A and the contiguous-K direction for operand B.
- Hadamard rotation pre-conditioning improves MXFP4 PPL by ~0.2 — the same recipe used for NVFP4 on Blackwell.
- W4A8 (MXFP4 weights × FP8 activations) is the most common mixed-precision config; `cbsz=4`, `blgp=0`.

## See Also

- [hw-mfma-scale-f8f6f4](mfma-scale-f8f6f4.md) — the MFMA instruction wrapping
- [hw-nvfp4](nvfp4.md) — NVIDIA counterpart
- [kernel-ck-mxfp4-gemm-cdna4](../kernels/ck-mxfp4-gemm-cdna4.md) — production CK reference
- [technique-fine-grained-quantization](../techniques/fine-grained-quantization.md) — block-scaling theory
