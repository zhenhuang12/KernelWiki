---
id: hw-buffer-load-lds
title: "buffer_load_lds / global_load_lds — Direct Global-to-LDS Loads"
type: hardware
architectures: [cdna3, cdna4]
tags: [buffer-load-lds, global-load-lds, lds, direct-to-lds, s-waitcnt]
confidence: source-reported
related: [hw-lds, hw-tma, technique-direct-to-lds, technique-mfma-pipelining, technique-wave-specialization, lang-amdgcn-asm, migration-cuda-to-hip]
sources: [doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-buffer-load-lds, pr-composable-kernel-1384, pr-composable-kernel-3098]
aliases: [buffer_load_lds, "direct-to-LDS", DirectToLds, buffer_load_dword_lds, buffer_load_dwordx4_lds, global_load_lds_dword]
---

# buffer_load_lds / global_load_lds — Direct Global-to-LDS Loads

## Overview

`buffer_load_*_lds` and `global_load_lds_*` are CDNA instructions that move data directly from global memory (through L1 / Infinity Cache) into LDS, bypassing the VGPR file entirely. This is the CDNA capability that most closely fills the role of NVIDIA's `cp.async.bulk` (TMA): it lets a producer wave keep the VGPR file free for accumulator pipelining while the load is in flight.

## Variants

| Instruction | Arch | Width / lane | Notes |
|-------------|------|--------------|-------|
| `buffer_load_dword_lds` | available on GCN / CDNA 1 / CDNA 2 / CDNA 3 / CDNA 4 | 32 bits | Buffer-descriptor addressed; pre-existing, not new in CDNA 3 |
| `global_load_lds_dword` | available on CDNA 3 / CDNA 4 | 32 bits | Flat-address variant |
| `buffer_load_dwordx4_lds` | **new in CDNA 4** | 128 bits | 4× wider; primary CDNA 4 path |

Per-lane width × 64 lanes/wave = per-wave transfer width:

- CDNA 3: 64 × 4 B = 256 B per wave per issue.
- CDNA 4: 64 × 16 B = 1024 B per wave per issue.

A typical MFMA producer wave issues a small unrolled sequence (2–8) of these per K-step.

## Why It's the AMD "TMA"

| Property | NVIDIA TMA (cp.async.bulk) | AMD buffer_load_lds |
|----------|----------------------------|---------------------|
| Issuer | Single thread (descriptor-driven) | Per-lane (wave-collective) |
| Width | Up to 5D tile per descriptor | 32 b (CDNA 3) / 128 b (CDNA 4) per lane per issue |
| Address calc | Hardware (from descriptor) | Software (per lane) |
| Sync primitive | mbarrier.arrive.expect_tx | s_waitcnt vmcnt(N) |
| Cache control | L2 hint via descriptor | `glc` / `slc` / `dlc` bits per instruction |
| Tile reorder | Built-in (TMA can swizzle) | Software-side address calc |

The wave-collective, per-lane nature is the biggest difference: AMD code has to handcraft the per-lane address arithmetic that TMA encodes into a single descriptor.

## Idiomatic Producer Wave (CDNA 4, BF16 K tile)

```cpp
// 128 b / lane × 64 lanes = 1024 B per issue; 16 issues = 16 KB K-tile
__device__ void producer_load_k_tile(
    const __global float4* k_ptr,  // global K tile, contiguous
    __local float4* k_lds,         // LDS K tile (XOR-swizzled layout)
    int k_stage)                    // pipeline stage 0..N-1
{
    const int lane = threadIdx.x & 63;

    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
        k_ptr, /*stride=*/0, /*num_records=*/INT_MAX, /*flags=*/0);

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int g_off = (i * 64 + lane) * sizeof(float4);
        int l_off = lds_swizzle(i, lane) * sizeof(float4) + k_stage * STAGE_BYTES;
        // 128-bit per-lane direct-to-LDS load (CDNA 4)
        __builtin_amdgcn_raw_buffer_load_lds(
            rsrc,
            /*lds_ptr =*/ (__attribute__((address_space(3))) void*)((char*)k_lds + l_off),
            /*size    =*/ 16,   // bytes per lane; 16 on CDNA 4, 4 on CDNA 3
            /*voffset =*/ g_off,
            /*soffset =*/ 0,
            /*offset  =*/ 0,
            /*aux     =*/ 0);
    }
    // Producer issues all loads, then the consumer-handoff drain happens
    // at the next pipeline barrier:
    //   s_waitcnt vmcnt(0)
    //   s_barrier
}
```

## Synchronization Pattern

```
Producer wave:
  issue buffer_load_dwordx4_lds × N    ← all loads in flight
  s_waitcnt vmcnt(0)                    ← drain HBM/L2 traffic
  s_barrier                             ← release consumer

Consumer wave:
  s_barrier                             ← wait for producer drain
  ds_read_b128 from LDS                 ← stage operand into VGPR
  v_mfma_*                              ← accumulate
```

The `s_waitcnt vmcnt(0)` is the critical fence — without it, `s_barrier` may release before the async loads have landed in LDS. There is no `expect_tx`-style byte-counted completion as on NVIDIA mbarrier; you must drain the counter.

## CDNA 3 → CDNA 4 Width Implications

Quadrupling per-lane width means CDNA 4 producer waves saturate HBM3E with fewer in-flight issues. A 32 KB tile that took 32 issues on CDNA 3 now takes 8 on CDNA 4, freeing schedule slots for compute overlap. Most CDNA 3 producer-wave loops translate by changing the intrinsic and adjusting the unroll factor; the surrounding `s_waitcnt` / `s_barrier` scaffolding is unchanged.

## Caveats

- Per-lane address arithmetic is your job — bugs here are silent corruption, not faults. Test against a reference CPU-computed LDS layout.
- LDS write conflicts (two lanes writing the same dword) are race conditions; the XOR-swizzle pattern must guarantee one writer per dword.
- `glc` / `slc` cache-policy bits on the buffer descriptor matter for K-stationary GEMMs where you want operand A to bypass L2 (set `slc=1`) while operand B is cached.

## See Also

- [hw-lds](lds.md) — LDS bank model and `ds_read` / `ds_write`
- [technique-direct-to-lds](../techniques/direct-to-lds.md) — pipelining pattern
- [hw-tma](tma.md) — NVIDIA counterpart
- [migration-cuda-to-hip](../migration/cuda-to-hip.md) — TMA → buffer_load_lds porting
