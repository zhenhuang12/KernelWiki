---
id: doc-amd-cdna4-isa
title: "AMD Instinct MI350 / CDNA 4 Instruction Set Architecture Reference"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi350-cdna4-instruction-set-architecture.pdf
source_category: official-doc
architectures:
- cdna4
tags:
- mfma
- mfma-scale-f8f6f4
- mxfp4
- mxfp6
- mxfp8
- block-scale
- lds
- buffer-load-lds
- amdgcn-asm
- wavefront-64
retrieved_at: 2026-04-27
---

# AMD Instinct MI350 / CDNA 4 ISA Reference

## Overview

The official AMDGCN ISA reference for the gfx950 family (CDNA 4). Adds the block-scaled MFMA family `v_mfma_scale_*_f8f6f4`, widens direct-to-LDS to 128 bits, and exposes MXFP block-scale operands.

## CDNA 4-Specific Additions

### Block-Scaled MFMA Instructions

```
v_mfma_scale_f32_16x16x128_f8f6f4 acc, a, b, scaleA, scaleB cbsz blgp abid
v_mfma_scale_f32_32x32x64_f8f6f4  acc, a, b, scaleA, scaleB cbsz blgp abid
```

Encoding fields:

- `cbsz` — 3-bit A-operand format selector: 0=FP8 E4M3, 1=FP8 E5M2, 2=FP6 E2M3, 3=FP6 E3M2, 4=FP4 E2M1, 5=BF8, …
- `blgp` — 3-bit B-operand format selector (same encoding).
- `abid` — scale-broadcast control inside the 32-element block.
- `scaleA / scaleB` — UE8M0 (8-bit unsigned, all-exponent) per-block scales.

The scale stream is interleaved by the assembler / compiler — typically one scale dword per 32 operand elements per matrix. CDNA 4 implements these natively (no software scaling).

### Wider Direct-to-LDS

`buffer_load_dwordx4_lds` issues a 128-bit (4-dword) global-to-LDS transfer per lane, 4× the CDNA 3 width. This makes producer waves capable of feeding a 32×128 BF16 tile per instruction without intermediate VGPR traffic.

### Larger LDS

160 KB per CU (vs 64 KB on CDNA 3). Permits deeper pipeline stages and larger persistent caches for K-stationary GEMMs. Bank count and 4 B granularity unchanged.

## MFMA Shape Catalogue (incremental over CDNA 3)

| Shape | Instruction | Notes |
|-------|-------------|-------|
| 16×16×128 mixed FP8/6/4 | `v_mfma_scale_f32_16x16x128_f8f6f4` | block-scaled |
| 32×32×64 mixed FP8/6/4 | `v_mfma_scale_f32_32x32x64_f8f6f4` | block-scaled |
| 16×16×32 BF16 | `v_mfma_f32_16x16x32_bf16` | new dtype combo |
| 16×16×32 FP16 | `v_mfma_f32_16x16x32_f16` | wider K |

## Why It Matters

CDNA 4 is the AMD vehicle for MXFP-format LLM inference. Quantization recipes designed for Blackwell NVFP4 (UE8M0 scaling, 32-element blocks) transfer directly to MXFP4 on CDNA 4 — only the instruction wrapping differs. See [hw-mfma-scale-f8f6f4](../../wiki/hardware/mfma-scale-f8f6f4.md), [hw-mxfp-cdna4](../../wiki/hardware/mxfp-cdna4.md).
