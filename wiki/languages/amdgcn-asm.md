---
id: lang-amdgcn-asm
title: "AMDGCN Inline Assembly"
type: language
tags: [amdgcn-asm, mfma, lds, s-waitcnt, buffer-load-lds, hip]
related: [lang-hip, lang-composable-kernel, hw-mfma, hw-mfma-scale-f8f6f4, hw-buffer-load-lds, hw-lds]
sources: [doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-mfma-tutorial, blog-rocm-buffer-load-lds]
reproducibility: snippet
architectures: [cdna3, cdna4]
confidence: source-reported
aliases: [AMDGCN, "AMDGCN assembly", "GCN assembly", "amdgcn ISA"]
---

# AMDGCN Inline Assembly

## Overview

AMDGCN is the ISA of AMD CDNA / RDNA GPUs. In HIP, AMDGCN inline assembly is the escape hatch for cases where the compiler scheduler or intrinsic coverage falls short — analogous to inline PTX in CUDA C++.

The bulk of CK-Tile's hot inner loops, and a few AITER hand-tuned kernels (e.g. the MLA decode QK fence pattern), use AMDGCN inline asm to pin specific instruction sequences.

## Inline-Asm Syntax

```cpp
asm volatile(
    "v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]\n"
    : /* outputs */
    : "{a0}"(acc_lo)
    : "memory");
```

Constraint letters:

| Letter | Class | Notes |
|--------|-------|-------|
| `v` | VGPR | regular vector register |
| `a` | AGPR | accumulation VGPR |
| `s` | SGPR | scalar register |
| `{vN}` / `{aN}` / `{sN}` | fixed register | pin to specific register number |
| `n` | immediate | constant |

Multi-register operands use range syntax: `v[0:3]` is 4 consecutive VGPRs starting at v0.

## Frequently Used Idioms

### Pin an MFMA to known register

```cpp
// Force accumulator to a[0:3], operands to v[0:1] and v[2:3]
asm volatile(
    "v_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[0:3]"
    : "+a"(acc) : "v"(a_frag), "v"(b_frag));
```

### Scheduler barrier with explicit mask

```cpp
// Block compiler reordering between asm blocks of distinct classes.
// Mask bits below are illustrative — the canonical reference is LLVM
// AMDGPUUsage's `__builtin_amdgcn_sched_barrier` mask table
// (https://llvm.org/docs/AMDGPUUsage.html#llvm-amdgcn-sched-barrier).
// 0x1 = ALU, 0x2 = VMEM (vector memory), 0x4 = SALU, 0x8 = SMEM, 0x20 = LDS
__builtin_amdgcn_sched_barrier(0x0);   // hard barrier (no reorder)
__builtin_amdgcn_sched_barrier(0x1);   // allow only ALU to cross
```

### Drain memory counters explicitly

```cpp
asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
```

This is the AMD equivalent of `__threadfence_block()` for ordering an async load (vmcnt) and an LDS operation (lgkmcnt) before a `s_barrier`.

### Wave priority

```cpp
asm volatile("s_setprio 3" :::);        // boost this wave's priority
// ... critical section ...
asm volatile("s_setprio 0" :::);
```

Useful in MoE routing waves to ensure the dispatch wave drains before MFMA waves issue.

### Direct-to-LDS load (when the intrinsic isn't available)

```cpp
// CDNA 4 buffer_load_dwordx4 into LDS at offset %0
asm volatile(
    "buffer_load_dwordx4 v[%0:%0+3], v[%1:%1+1], s[%2:%2+3], 0 idxen lds offset:%3"
    :: "v"(lds_off), "v"(g_off), "s"(rsrc), "n"(0));
```

## When to Reach for Inline Asm

| Situation | Use asm? |
|-----------|----------|
| Single MFMA call | No — use `__builtin_amdgcn_mfma_*` |
| Pinning a specific AGPR for accumulator continuity | Yes |
| Wave-priority / sleep hints | Yes |
| Scheduler barrier with custom mask | Use `__builtin_amdgcn_sched_barrier` |
| Counter drain in a fence pattern | Yes |
| New instruction not yet in clang | Yes |

## Caveats

- Inline asm is opaque to the LLVM AMDGPU scheduler — it can't reorder across `asm volatile` blocks. Over-use will pessimize otherwise-schedulable code.
- Constraint letters differ subtly across LLVM versions; pin clang to a known-good ROCm release for production kernels.
- AMDGCN instruction encodings change between gfx generations (gfx942 vs gfx950 add/remove some instructions); guard with `#if __gfx950__`.

## See Also

- [doc-amd-cdna3-isa](../../sources/docs/amd-cdna3-isa.md) — gfx942 instruction reference
- [doc-amd-cdna4-isa](../../sources/docs/amd-cdna4-isa.md) — gfx950 instruction reference
- [hw-mfma](../hardware/mfma.md) — MFMA shapes
- [hw-buffer-load-lds](../hardware/buffer-load-lds.md) — direct-to-LDS encodings
