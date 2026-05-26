---
id: technique-mfma-pipelining
title: "MFMA Software Pipelining on AMD CDNA"
type: technique
architectures: [cdna3, cdna4]
tags: [mfma-pipelining, mfma, lds, buffer-load-lds, sched-barrier]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma, hw-lds, hw-buffer-load-lds, hw-agpr]
related: [technique-wave-specialization, technique-direct-to-lds, technique-double-buffering, technique-register-budgeting, lang-amdgcn-asm]
sources: [doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-mfma-tutorial, blog-rocm-buffer-load-lds, pr-composable-kernel-1384]
aliases: ["MFMA pipelining", "MFMA software pipeline", "K-stage MFMA pipeline"]
symptoms: ["MFMA throughput well below peak", "high VGPR pressure with low occupancy", "compute-bound but not at roofline"]
---

# MFMA Software Pipelining on AMD CDNA

## Overview

AMD MFMA instructions are synchronous — there is no async commit/wait model like NVIDIA's wgmma or tcgen05. ILP must come from interleaving MFMA against `ds_read` (LDS → VGPR staging) and `buffer_load_dwordx4_lds` (HBM → LDS prefetch; `buffer_load_dwordx4_lds` is CDNA 4 / gfx950 only, CDNA 3 / gfx942 has only the 32-bit `buffer_load_dword_lds` variant). The technique of doing this explicitly via loop unrolling, AGPR double-buffering, and `__builtin_amdgcn_sched_barrier` is what AMD docs call MFMA pipelining.

A well-pipelined inner loop reaches 80-90% of peak MFMA throughput. A naive loop typically gets 30-50%.

## Three Levels of Pipeline

### Level 1: K-loop unroll + LDS double-buffer

```cpp
// Single-stage producer drains LDS each iteration — simplest, lowest occupancy
__shared__ bf16 smem_a[2][BLOCK_M * BLOCK_K];   // double-buffered
__shared__ bf16 smem_b[2][BLOCK_K * BLOCK_N];

for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
    int buf = (k_tile / BLOCK_K) & 1;
    cooperative_load(smem_a[buf], A + k_tile);
    cooperative_load(smem_b[buf], B + k_tile);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BLOCK_K; kk += MFMA_K) {
        auto a = lds_read(smem_a[buf], kk);
        auto b = lds_read(smem_b[buf], kk);
        acc = mfma_16x16x16(a, b, acc);
    }
    __syncthreads();
}
```

### Level 2: Operand prefetch + AGPR continuity

```cpp
// Prefetch next-iteration LDS reads into VGPR before this iteration's MFMA
auto a_curr = lds_read(smem_a, 0);
auto b_curr = lds_read(smem_b, 0);

#pragma unroll
for (int kk = 0; kk < BLOCK_K - MFMA_K; kk += MFMA_K) {
    // Prefetch next K-step
    auto a_next = lds_read(smem_a, kk + MFMA_K);
    auto b_next = lds_read(smem_b, kk + MFMA_K);

    // MFMA on current — overlaps the ds_read latency
    acc = mfma_16x16x16(a_curr, b_curr, acc);

    a_curr = a_next;
    b_curr = b_next;
}
// Tail
acc = mfma_16x16x16(a_curr, b_curr, acc);
```

The AGPR accumulator stays resident across all K-steps; LLVM does not need to spill it. The `ds_read` latency (~8 cycles) overlaps with MFMA throughput.

### Level 3: Wave-specialized producer + scheduler barrier

```cpp
// Inside consumer wave; producer wave is separately issuing buffer_load_*_lds
#pragma unroll
for (int kk = 0; kk < BLOCK_K; kk += MFMA_K) {
    auto a = lds_read(smem_a[stage], kk);
    auto b = lds_read(smem_b[stage], kk);

    // Prevent LLVM from hoisting the next K-stage's ds_read before this MFMA;
    // 0x308 = MFMA (0x008) | DS read (0x100) | DS write (0x200) — see L120 below.
    __builtin_amdgcn_sched_barrier(0x308);
    acc = mfma_32x32x16(a, b, acc);
    __builtin_amdgcn_sched_barrier(0x308);
}
```

This is the V3 CK-Tile pipeline pattern. The producer wave is independently issuing `buffer_load_dwordx4_lds` (CDNA 4 / gfx950 only; on CDNA 3 / gfx942 the producer must issue four 32-bit `buffer_load_dword_lds` operations per 16 B chunk instead), so HBM latency is fully hidden by the consumer's MFMA loop.

## AGPR Pressure Management

A 32×32×16 BF16 MFMA writes 16 AGPRs per lane. If your wave runs 4 such MFMAs in flight (4-stage AGPR pipeline), that's 64 AGPRs/lane just for accumulators — out of the shared 512-register total. Two strategies:

- **Smaller MFMA (16×16×16)**: 4 AGPRs/lane per MMA → 4-stage pipeline costs 16 AGPRs, leaves 496 VGPRs.
- **Single-buffered AGPR + tight scope**: keep one accumulator live, copy to VGPR at epilogue. Highest peak throughput but no MFMA-MFMA overlap.

CK-Tile picks per-shape automatically; hand-rolled HIP kernels should measure with `rocprofv3 --pmc SQ_VGPR_LANE_RATIO,SQ_INSTS_VALU_MFMA`.

## Scheduler Barrier Mask Reference

`__builtin_amdgcn_sched_barrier(mask)` accepts a bitmask of instruction classes that *may* cross the barrier. The canonical bit assignments per the [AMDGPU LLVM Usage Guide](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html) (`llvm.amdgcn.sched.barrier`) are:

| Bit | Meaning |
|-----|---------|
| 0x0000 | none — hard barrier; no instructions may cross |
| 0x0001 | non-memory, non-side-effecting instructions may cross |
| 0x0002 | VALU instructions may cross |
| 0x0004 | SALU instructions may cross |
| 0x0008 | MFMA / WMMA instructions may cross |
| 0x0010 | all VMEM instructions may cross |
| 0x0020 | VMEM read instructions may cross |
| 0x0040 | VMEM write instructions may cross |
| 0x0080 | all DS instructions may cross |
| 0x0100 | DS read instructions may cross |
| 0x0200 | DS write instructions may cross |
| 0x0400 | transcendental instructions may cross |

Bits combine with bitwise OR. To protect an MFMA inner loop from being polluted by global-memory ops while still letting LDS traffic and MFMAs reorder, use `__builtin_amdgcn_sched_barrier(0x0008 | 0x0100 | 0x0200)` — i.e. `0x308`, which permits MFMAs and DS reads/writes to cross but pins all VMEM and VALU/SALU operations in place. (The legacy combination `0x60` shown in older blog posts maps to "VMEM-write + all-DS" under the canonical mask and is rarely what callers actually want.)

## Caveats

- Aggressive unrolling balloons code size. CDNA's instruction cache is 32 KB / CU; an unrolled 32-iteration K-loop with 4 MFMAs/iter can spill it. Profile with `SQ_INSTS_ICACHE_MISS`.
- LLVM's MFMA scheduler heuristic improved substantially between ROCm 5.7 and 6.0; older builds may need explicit `sched_barrier` annotations that newer builds make unnecessary.
- Mixing `v_mfma_*` and `v_mfma_scale_*` in the same loop is supported on CDNA 4 but the LLVM scheduler may not interleave them optimally — measure with `rocprofv3`.

## See Also

- [hw-mfma](../hardware/mfma.md) — MFMA instruction family
- [hw-agpr](../hardware/agpr.md) — accumulator pressure
- [technique-wave-specialization](wave-specialization.md) — pairs with this technique
- [technique-direct-to-lds](direct-to-lds.md) — producer-side primitive
