---
id: technique-wave-specialization
title: "Wave Specialization on AMD CDNA"
type: technique
architectures: [cdna3, cdna4]
tags: [wave-specialization, mfma, lds, buffer-load-lds]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma, hw-lds, hw-buffer-load-lds]
related: [technique-warp-specialization, technique-mfma-pipelining, technique-direct-to-lds, technique-lds-swizzling, hw-mfma, hw-lds, lang-composable-kernel]
sources: [doc-amd-cdna3-whitepaper, blog-rocm-ck-flash-attention, blog-rocm-buffer-load-lds, pr-composable-kernel-1384, pr-composable-kernel-2110]
aliases: ["wave specialization", "wavefront specialization", "producer-consumer wave"]
---

# Wave Specialization on AMD CDNA

## Overview

Wave specialization assigns distinct functional roles to the wavefronts within a workgroup — typically producer waves that drive direct-to-LDS loads while consumer waves run MFMA accumulation. It is the AMD analogue of NVIDIA warp specialization, but with two structural differences that make it harder:

1. **No wave-level barrier**: NVIDIA `__syncwarp` has no AMD equivalent. Wave-pair handoffs must use `s_barrier` (workgroup-wide) or LDS-resident flags polled with `s_waitcnt lgkmcnt(0)`.
2. **Synchronous MFMA**: AMD MFMAs are not async; ILP comes from interleaving MFMA against `ds_read` and `buffer_load_lds`, not from a commit/wait model.

Despite these constraints, wave-specialized kernels (CK-Tile V3 pipeline) routinely reach 70-85% of peak MFMA throughput on CDNA 3 / CDNA 4.

## Canonical 4-Wave Structure

| Wave | Role | Responsibility |
|------|------|----------------|
| 0, 1 (CDNA 3) | Producer | Issue `buffer_load_dword_lds` for A/B tiles (32 b/lane, widest direct-to-LDS on gfx942); arrive at `s_barrier` after `s_waitcnt vmcnt(0)` |
| 0, 1 (CDNA 4) | Producer | Issue `buffer_load_dwordx4_lds` for A/B tiles (128 b/lane, gfx950 only); arrive at `s_barrier` after `s_waitcnt vmcnt(0)` |
| 2, 3 | Consumer | Wait at `s_barrier`; `ds_read_b128` from LDS into VGPR; issue `v_mfma_*` accumulating to AGPR |

For attention kernels, the consumer waves also run the online softmax (rescale, max, exp) between MFMA blocks.

## Skeleton (HIP + CK-Tile-style pseudocode)

```cpp
__global__ void __launch_bounds__(256)
gemm_wave_specialized(const bf16* A, const bf16* B, float* C,
                       int M, int N, int K) {
    const int wave_id = threadIdx.x / 64;
    const int lane    = threadIdx.x % 64;

    __shared__ bf16 smem_a[STAGES][BLOCK_M * BLOCK_K];
    __shared__ bf16 smem_b[STAGES][BLOCK_K * BLOCK_N];

    if (wave_id < 2) {
        // =========== PRODUCER WAVES ===========
        for (int k_tile = 0, stage = 0; k_tile < K;
             k_tile += BLOCK_K, stage = (stage + 1) % STAGES) {

            // Issue all direct-to-LDS loads for this K-stage
            direct_to_lds_load_a(smem_a[stage], A, k_tile, wave_id, lane);
            direct_to_lds_load_b(smem_b[stage], B, k_tile, wave_id, lane);

            // Drain HBM/L2 traffic — required before consumer reads LDS
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();   // handoff to consumers

            // Wait for consumers to drain this stage before reusing
            __syncthreads();
        }
    } else {
        // =========== CONSUMER WAVES ===========
        f32x16 acc = {0};   // AGPR accumulator (16 AGPRs/lane for 32x32 MFMA)

        for (int k_tile = 0, stage = 0; k_tile < K;
             k_tile += BLOCK_K, stage = (stage + 1) % STAGES) {

            __syncthreads();   // wait for producer's stage

            // MFMA inner loop
            #pragma unroll
            for (int kk = 0; kk < BLOCK_K; kk += MFMA_K) {
                auto a = lds_read_b128(smem_a[stage], wave_id, lane, kk);
                auto b = lds_read_b128(smem_b[stage], wave_id, lane, kk);
                acc = __builtin_amdgcn_mfma_f32_32x32x16bf16(a, b, acc, 0, 0, 0);
            }

            __syncthreads();   // release stage back to producers
        }

        // Epilogue: copy AGPR → VGPR → global store
        store_acc_to_global(C, acc, wave_id, lane);
    }
}
```

## Scheduling Notes

- The double `__syncthreads()` per stage (handoff + release) replaces NVIDIA's per-stage mbarrier pair. Each barrier is a `s_barrier` workgroup-wide.
- `s_waitcnt vmcnt(0)` after the producer's last load is critical — without it the consumer can read pre-fetch LDS contents.
- Place `__builtin_amdgcn_sched_barrier(0x308)` around the MFMA inner loop if the LLVM scheduler is interleaving `ds_read` from a future K-stage into the current one (rare with the V3 pipeline but possible with custom HIP). `0x308 = MFMA (0x008) | DS read (0x100) | DS write (0x200)` — matches the canonical inner-loop mask documented in `mfma-pipelining.md`, letting MFMAs and LDS traffic reorder while pinning VMEM and VALU/SALU in place.
- `s_setprio` is the CDNA wave-priority primitive that complements `sched_barrier`. It is an imm16 SOPP encoding where only the low 2 bits encode the priority (0-3); issuing `s_setprio 1` raises the calling wave's hardware scheduling priority so the issue arbiter favors it on contended cycles. The typical wave-specialized pattern runs producer waves at priority 1 while consumer/MFMA waves stay at the default priority 0, biasing the arbiter toward keeping the direct-to-LDS load pipeline full and ahead of MFMA consumption. Together with `sched_barrier` (instruction-group fencing) and `s_barrier` (wave synchronization), `s_setprio` is the third leg of software wave specialization on CDNA, which lacks any hardware warp-specialization machinery analogous to Hopper's specialized warps.

## CDNA 3 vs CDNA 4

- **LDS budget**: 64 KB on CDNA 3 caps the stage count at 2 for large tiles; 160 KB on CDNA 4 enables 3-4 stages.
- **Direct-to-LDS width**: 32 b/lane on CDNA 3 means more producer issues per K-step; 128 b/lane on CDNA 4 frees producer waves to do work other than just load.
- **Scaled MFMA**: CDNA 4 producer waves also pull scale streams (UE8M0 bytes) alongside operand streams for MXFP4/6.

## Why It's Harder than NVIDIA Warp Specialization

| Aspect | NVIDIA SM100 warp-spec | AMD CDNA wave-spec |
|--------|-------------------------|---------------------|
| MMA dispatch | Single thread (warp 1, lane 0) | Whole wave (all 64 lanes) |
| Async commit/wait | Yes (tcgen05.commit, wait) | No (synchronous MFMA) |
| Per-warp barrier | `__syncwarp` | none — use `s_barrier` |
| Async load sync | mbarrier with `expect_tx` | `s_waitcnt vmcnt` |
| Producer width | TMA descriptor (any size) | Per-lane (256-1024 B / wave / issue) |

The AMD model is more like a software-pipelined CPU producer/consumer than NVIDIA's hardware-supported async model. The compensation is higher VGPR/AGPR capacity per CU.

## Caveats

- Workgroup-wide `s_barrier` is more expensive than NVIDIA per-warpgroup `bar.sync` would be — keep stage counts small (2-4) to amortize.
- Producer wave under-utilization is common: with 128-bit direct-to-LDS on CDNA 4, a producer wave often finishes its issue list before the consumer drains the stage, leaving cycles on the table. AITER's fused-MoE kernel uses idle producer waves for the top-K routing computation.

## See Also

- [technique-warp-specialization](warp-specialization.md) — NVIDIA counterpart
- [technique-mfma-pipelining](mfma-pipelining.md) — software pipeline detail
- [technique-direct-to-lds](direct-to-lds.md) — producer-side primitive
- [lang-composable-kernel](../languages/composable-kernel.md) — V3 scheduler in CK-Tile
