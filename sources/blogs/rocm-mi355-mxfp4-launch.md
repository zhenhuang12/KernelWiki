---
id: blog-rocm-mi355-mxfp4-launch
title: "AMD Instinct MI350 / MI355X: MXFP4 Block-Scaled Matrix Cores"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/artificial-intelligence/mi350-mxfp4/README.html
source_category: benchmark-blog
architectures:
- cdna4
tags:
- mfma
- mfma-scale-f8f6f4
- mxfp4
- mxfp6
- mxfp8
- block-scale
- composable-kernel
- gemm
retrieved_at: 2026-04-27
---

## Summary

AMD's developer-facing introduction to the CDNA 4 block-scaled MFMA family. Walks through the OCP MXFP4 storage layout (32-element blocks, one UE8M0 8-bit scale per block), the `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` instructions, and how CK-Tile's `BlockGemmPipelineMxFp4` template wraps them.

## Code Sketch from the Post

```cpp
// CK-Tile MXFP4 GEMM block program (CDNA 4)
using BlockGemm = ck_tile::BlockGemmPipelineMxFp4<
    /*Problem=*/ck_tile::GemmFp4Problem<
        ALayout, BLayout, CLayout,
        /*ADataType=*/ck_tile::fp4_t,
        /*BDataType=*/ck_tile::fp4_t,
        /*AccDataType=*/float,
        /*CDataType=*/ck_tile::bf16_t,
        /*BlockTile=*/ck_tile::sequence<256, 256, 128>,
        /*WarpTile=*/ck_tile::sequence<32, 32, 64>>,
    /*Scheduler=*/ck_tile::GemmPipelineSchedulerV3>;

// Inside the kernel body:
auto a_tile = load_global_to_lds_mxfp4(a_ptr, ...);
auto b_tile = load_global_to_lds_mxfp4(b_ptr, ...);
ck_tile::block_sync_lds();
auto acc = BlockGemm::run(a_tile, b_tile, /*scaleA=*/sA, /*scaleB=*/sB);
```

The pipeline issues `buffer_load_dwordx4_lds` (CDNA 4's 128-bit direct-to-LDS) for both operand and scale streams, schedules the MFMA group with `__builtin_amdgcn_sched_barrier` to keep the `v_mfma_scale_f32_32x32x64_f8f6f4` issue cadence locked to LDS bandwidth, and accumulates into a 16-AGPR tile.

## Performance Claims (from the post)

- 256×256×8192 MXFP4 GEMM: ~3.4 PFLOPS on MI355X (~80% of peak MXFP4 MFMA throughput).
- Same shape FP8: ~1.7 PFLOPS — MXFP4 doubles dense throughput at iso-tile-size.

## Why It Matters

MXFP4 on CDNA 4 is the direct AMD analogue of NVFP4 on Blackwell. The CK-Tile pipeline shown is the closest mirror of CUTLASS SM100's NVFP4 GEMM, and the same quantization recipes carry over (UE8M0 scales, 32-element blocks). Cited by [hw-mxfp-cdna4](../../wiki/hardware/mxfp-cdna4.md), [hw-mfma-scale-f8f6f4](../../wiki/hardware/mfma-scale-f8f6f4.md), [kernel-ck-mxfp4-gemm-cdna4](../../wiki/kernels/ck-mxfp4-gemm-cdna4.md).
