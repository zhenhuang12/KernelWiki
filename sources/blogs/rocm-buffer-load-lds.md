---
id: blog-rocm-buffer-load-lds
title: "Direct-to-LDS Loads: buffer_load and global_load LDS Variants"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/software-tools-optimization/buffer-load-lds/README.html
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- buffer-load-lds
- global-load-lds
- direct-to-lds
- lds
- mfma-pipelining
- s-waitcnt
retrieved_at: 2026-04-27
---

## Summary

Walk-through of CDNA's "direct-to-LDS" load instructions — `buffer_load_dword_lds` (CDNA 3, 32 b), `global_load_lds_dword` (CDNA 3, 32 b), and `buffer_load_dwordx4_lds` (CDNA 4, 128 b). These instructions bypass VGPRs entirely: data goes from L1/Infinity Cache straight into LDS, freeing the VGPR file for MFMA operand staging.

## Why It's the AMD TMA

NVIDIA Hopper/Blackwell hides global→shared transfers behind `cp.async.bulk` (TMA), a single descriptor-driven bulk copy. AMD does not have a single bulk descriptor — instead each lane issues its own `buffer_load_*_lds` and the wave collectively transfers a tile. Differences:

| Aspect | NVIDIA TMA | AMD buffer_load_lds |
|--------|------------|---------------------|
| Issue granularity | Single thread, descriptor | Per-lane (wave-collective) |
| Transfer width | Up to 5D tile per descriptor | 32 b (CDNA 3) / 128 b (CDNA 4) per lane |
| Sync primitive | mbarrier with byte-count expectation | s_waitcnt vmcnt |
| Address calc | Hardware (from descriptor) | Software (per lane) |

## Idiomatic Producer Wave Body

```cpp
// CDNA 4 producer: 128-bit direct-to-LDS load
__device__ void producer_load_kv_tile(
    const __global float4* k_ptr,   // global memory K tile
    __local float4* k_lds            // LDS K tile destination
) {
    const int lane = threadIdx.x % 64;
    // Each lane streams one float4 (128 bits) directly into LDS.
    __builtin_amdgcn_buffer_load_lds(
        /*rsrc=*/__builtin_amdgcn_make_buffer_rsrc(k_ptr, /*stride=*/0, /*num_records=*/INT_MAX, /*flags=*/0),
        /*offset=*/lane * sizeof(float4),
        /*lds_offset=*/lane * sizeof(float4),
        /*size=*/16);
    // Issue, then drain at the consumer-handoff barrier:
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();
}
```

## Why It Matters

Direct-to-LDS is the only way to keep the VGPR file free for accumulator pipelining on AMD. Every wave-specialized GEMM and attention kernel on CDNA 3/4 (CK, AITER) is built around this primitive. Cited by [hw-buffer-load-lds](../../wiki/hardware/buffer-load-lds.md), [technique-mfma-pipelining](../../wiki/techniques/mfma-pipelining.md).
