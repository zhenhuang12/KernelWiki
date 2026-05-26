---
id: doc-amd-cdna3-isa
title: "AMD Instinct MI300 / CDNA 3 Instruction Set Architecture Reference"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
source_category: official-doc
architectures:
- cdna3
tags:
- mfma
- lds
- agpr
- buffer-load-lds
- global-load-lds
- s-barrier
- s-waitcnt
- wavefront-64
- amdgcn-asm
retrieved_at: 2026-04-27
---

# AMD Instinct MI300 / CDNA 3 ISA Reference

## Overview

The official AMDGCN ISA reference for the gfx940 / gfx941 / gfx942 family (CDNA 3). Authoritative source for instruction encodings, MFMA shapes, and synchronization primitives (`s_waitcnt`, `s_barrier`, `s_setprio`).

## Instruction Categories Relevant to Kernel Authors

- **MFMA**: `v_mfma_f32_*` (FP32 accumulate), `v_mfma_f64_*` (FP64), `v_mfma_i32_*` (INT8), `v_mfma_f32_*_f8` (FP8). Operands in VGPR, accumulator in AGPR.
- **LDS access**: `ds_read_*` / `ds_write_*` (single-dword to 128 b); `ds_swizzle_b32` for in-lane permutations.
- **Direct-to-LDS**: `buffer_load_dword_lds` (single dword global→LDS, no VGPR round-trip); `global_load_lds_dword` (CDNA 3 only, 32 b variant).
- **Synchronization**: `s_waitcnt vmcnt(N) & lgkmcnt(M)` (wait for outstanding vector / LDS / scalar memory ops); `s_barrier` (workgroup barrier — note CDNA has no wave-level barrier equivalent to NVIDIA `__syncwarp`); `s_sendmsg` for performance counter / debug messages.
- **Scheduler hints**: `s_setprio` (wave priority 0–3); `__builtin_amdgcn_sched_barrier` (compiler intrinsic to constrain LLVM AMDGPU backend instruction scheduling).

## MFMA Shape Catalogue

Representative shapes (full table in §7 of the ISA reference):

| Shape | Instruction | Cycles/wave |
|-------|-------------|-------------|
| 16×16×16 FP16 | `v_mfma_f32_16x16x16f16` | 32 |
| 32×32×8 FP16 | `v_mfma_f32_32x32x8f16` | 64 |
| 16×16×32 FP8 | `v_mfma_f32_16x16x32_fp8_fp8` | 32 |
| 32×32×16 FP8 | `v_mfma_f32_32x32x16_fp8_fp8` | 64 |

Each MFMA reads two source matrices from VGPRs and accumulates into AGPRs. The compiler may copy AGPR↔VGPR via `v_accvgpr_read_b32` / `v_accvgpr_write_b32` to free regular VGPR pressure.

## s_waitcnt Counters

Three counters relevant to kernel pipelining:

- `vmcnt` — outstanding vector memory loads/stores (HBM / Infinity Cache traffic).
- `lgkmcnt` — outstanding LDS, GDS, scalar memory, and `s_sendmsg` operations.
- `expcnt` — outstanding exports (graphics; rarely relevant for compute).

Correctly placing `s_waitcnt` after async direct-to-LDS issues is the AMD analogue of NVIDIA's `cp.async.commit_group` / `wait_group` pattern.

## Why It Matters

The ISA reference is the source of truth for any AMDGCN inline-asm in HIP kernels and for understanding what Composable Kernel templates actually emit. See [lang-amdgcn-asm](../../wiki/languages/amdgcn-asm.md), [hw-mfma](../../wiki/hardware/mfma.md), [hw-buffer-load-lds](../../wiki/hardware/buffer-load-lds.md).
