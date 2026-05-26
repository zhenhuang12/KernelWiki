---
id: doc-amd-cdna3-whitepaper
title: "AMD CDNA 3 Architecture Whitepaper"
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
source_category: official-doc
architectures:
- cdna3
tags:
- mfma
- lds
- agpr
- xcd
- infinity-cache
- infinity-fabric
- xgmi
- wavefront-64
- fp8
retrieved_at: 2026-04-27
---

# AMD CDNA 3 Architecture Whitepaper

## Overview

The official AMD whitepaper describing the CDNA 3 (Instinct MI300X / MI300A / gfx942) compute architecture. CDNA 3 introduces a chiplet-based design with 8 Accelerator Complex Dies (XCDs), each containing 38 active compute units (304 CUs total on MI300X), 256 MB of shared Infinity Cache, and 192 GB of HBM3 at 5.3 TB/s.

## Key Hardware Features

### Matrix Cores (MFMA)

CDNA 3 implements the third-generation Matrix Fused Multiply-Add (MFMA) instructions. Per-CU peak rates:

| Data type | Ops/cycle/CU | Notes |
|-----------|--------------|-------|
| FP64 (matrix) | 256 | 2x CDNA 2 |
| FP32 | 256 | TF32 path |
| FP16 / BF16 | 1024 | |
| INT8 | 2048 | |
| FP8 (E4M3 / E5M2) | 2048 | OCP FP8 |

MFMA outputs accumulate in the Accumulation VGPR (AGPR) file — a 512×32-bit register pool per SIMD shared with the regular VGPR pool (total 512 VGPRs per SIMD, dynamically partitioned between VGPR / AGPR roles).

### Local Data Share (LDS)

64 KB per CU, 32 banks × 4 B (128 B / cycle peak), accessed via `ds_read` / `ds_write` instructions. LDS bank conflicts are resolved on dword granularity. Direct global-to-LDS prefetch is supported via `buffer_load_dword_lds` (32 b) and `global_load_lds_dword` (32 b) — the CDNA 3 equivalent of an asynchronous SMEM copy, but at single-dword granularity.

### Accelerator Complex Die (XCD)

8 XCDs per MI300X, each with its own L2 cache. Work-group dispatch is round-robin across XCDs by default, which can fragment L2 locality for cooperative kernels. XCD-aware scheduling (manually remapping `blockIdx.x` to keep adjacent tiles on the same XCD) typically gains 10-50% on cache-bound GEMMs.

### Infinity Cache (MALL)

256 MB memory-attached last-level cache shared by all XCDs, ~17 TB/s aggregate bandwidth. Hits dramatically reduce HBM pressure for K-stationary GEMMs and attention.

### Infinity Fabric / xGMI

8 xGMI links per GPU at 64 GB/s bidirectional per link, used for inter-GPU communication in 8-GPU OAM systems.

## Comparison with NVIDIA Hopper

| Aspect | NVIDIA Hopper (H100) | AMD CDNA 3 (MI300X) |
|--------|----------------------|----------------------|
| Compute units | 132 SMs | 304 CUs (8 XCDs) |
| MMA primitive | wgmma.mma_async (warpgroup) | MFMA (single wave) |
| MMA accumulator | Registers | AGPR pool |
| Shared memory | 228 KB / SM (configurable) | 64 KB LDS / CU |
| Async global→shmem | cp.async.bulk (TMA) | buffer_load_lds / global_load_lds |
| Cluster / chiplet | Thread Block Cluster | 8 XCDs |
| Last-level cache | 50 MB L2 | 256 MB Infinity Cache |
| Warp / wave size | 32 | 64 (wave64) |

## Why It Matters

CDNA 3 is the reference architecture for porting Hopper/Blackwell kernels to AMD Instinct. Most performance-critical decisions — wave-specialized producer/consumer, LDS swizzling, XCD-aware tile mapping, direct-to-LDS prefetch — are direct analogues of patterns that work on NVIDIA, but with different granularities and synchronization primitives. See [hw-mfma](../../wiki/hardware/mfma.md), [hw-lds](../../wiki/hardware/lds.md), [hw-xcd](../../wiki/hardware/xcd.md).
