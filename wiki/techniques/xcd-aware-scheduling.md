---
id: technique-xcd-aware-scheduling
title: "XCD-Aware Work-Group Scheduling on CDNA 3 / CDNA 4"
type: technique
architectures: [cdna3, cdna4]
tags: [xcd-aware-scheduling, xcd, infinity-cache, tile-scheduling]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-xcd, hw-infinity-cache]
related: [hw-xcd, hw-infinity-cache, technique-tile-scheduling, technique-persistent-kernels]
sources: [doc-amd-cdna3-whitepaper, blog-gpuopen-mi300-architecture, pr-aiter-297]
aliases: ["XCD-aware scheduling", "XCD-aware tile mapping", "chiplet-aware scheduling"]
symptoms: ["low L2 hit rate", "Infinity Cache hit rate below expected", "cache-bound GEMM not reaching roofline"]
---

# XCD-Aware Work-Group Scheduling

## Overview

MI300X / MI355X dispatch work-groups round-robin across their 8 XCDs (chiplets). For any kernel where adjacent `blockIdx.x` work on data that should share an XCD's 4 MB L2 cache, this default scatter destroys L2 locality. A one-line `blockIdx.x` permutation that clusters consecutive logical tiles onto the same XCD recovers that locality and is one of the highest-leverage AMD-specific optimizations for cooperative kernels. Reported wins from [ROCm engineering blogs](https://rocm.blogs.amd.com/) and ROCm/AITER PRs land roughly in the **10-25%** range for cache-bound GEMMs and **5-15%** for attention prefill, varying significantly with tile shape, K-depth, and operand residency; HBM-bound kernels see little or no change.

## Problem

With 8 XCDs, the hardware's default dispatch order is:

```
WG 0  → XCD 0    WG 8  → XCD 0    WG 16 → XCD 0
WG 1  → XCD 1    WG 9  → XCD 1    WG 17 → XCD 1
...                ...                ...
WG 7  → XCD 7    WG 15 → XCD 7    WG 23 → XCD 7
```

For a tile-major GEMM where `blockIdx.x = m_tile * num_n_tiles + n_tile`, adjacent K-tiles for the same M-row land on different XCDs. Each XCD's L2 holds fragments of unrelated tiles — hit-rate collapses to ~5%.

## Fix

Permute `blockIdx.x` so 8 consecutive logical tiles map to the same physical XCD:

```cpp
__device__ __forceinline__ int xcd_remap(int bid, int num_xcd = 8) {
    int xcd       = bid % num_xcd;                                   // physical XCD index
    int slot      = bid / num_xcd;                                   // round on that XCD
    int xcd_tiles = (gridDim.x + num_xcd - 1) / num_xcd;             // tiles per XCD
    return xcd * xcd_tiles + slot;                                   // logical tile index
}

__global__ void my_gemm(float* A, float* B, float* C, int M, int N, int K) {
    const int logical_bid = xcd_remap(blockIdx.x);
    const int m_tile = logical_bid / num_n_tiles;
    const int n_tile = logical_bid % num_n_tiles;
    // ... continue with logical_bid-derived indices
}
```

Tiles 0-7 now share XCD 0 (and its L2). Adjacent K-tiles for the same M-row hit each other's L2 entries.

## Reported Wins

ROCm engineering blogs and AITER PR descriptions report the following directional results (exact numbers vary by ROCm/clang version, tile shape, and partition mode — re-measure on your stack rather than treating these as guarantees):

- Cache-bound square BF16 GEMMs (K large enough that L2 reuse dominates): single-digit to low-double-digit percent throughput gains, typically in the **10-25%** band.
- Attention prefill with the Q-tile resident: **5-15%** throughput gain, depending on head dimension and sequence length.
- HBM-bound kernels (reductions, elementwise, memory-bound decode): essentially no change — the working set never fit in L2 to begin with.

The win scales with the operand-residency ratio: any kernel whose working set per logical tile-cluster fits in the 4 MB per-XCD L2 benefits. See the [ROCm Blogs](https://rocm.blogs.amd.com/) catalogue for current case studies.

## When to Apply

| Kernel shape | Apply XCD-aware? |
|--------------|-------------------|
| Square GEMM, K > 1024 | Yes (always) |
| FlashAttention prefill | Yes (Q-tile residency) |
| FlashAttention decode (paged-KV) | Marginal — KV cache scatter dominates |
| Reductions, elementwise | No (HBM-bound) |
| Persistent kernels (gridDim == num_cu) | No (each CU stationary) |

## Variations

### M-major vs N-major clustering

The remap above clusters in `blockIdx.x` order. If your GEMM iterates K-tiles before stepping M, you want adjacent M-tiles to share an XCD; recompute `logical_bid` accordingly:

```cpp
int logical_bid = xcd * xcd_tiles + slot;
int m_tile = logical_bid % num_m_tiles;   // ← swap m/n if K is innermost
int n_tile = logical_bid / num_m_tiles;
```

### Persistent + XCD-aware

A persistent kernel that processes `tile_count / num_cu` tiles per CU naturally keeps the per-CU working set stationary. Combining persistence + XCD-awareness eliminates the round-robin scatter entirely and is the highest-throughput pattern for K-stationary GEMMs.

### CPX-mode awareness

In CPX-N partition modes, only N XCDs are visible. Read the partition mask at startup (`hipDeviceGetAttribute` or AITER's `aiter.compute_partition()`) and adjust `num_xcd` accordingly.

## Caveats

- The permutation is software-only; the hardware still dispatches round-robin. You're choosing which *logical* tile each physical slot computes.
- Some kernels are written assuming `blockIdx.x` directly indexes into a swizzled output; XCD-aware remap interacts with output-swizzle logic and can require both to be re-derived together.
- The remap is a no-op in CPX-1 mode (single partition with 8 visible XCDs) — verify partition mode is what you expect.

## See Also

- [hw-xcd](../hardware/xcd.md) — XCD topology
- [hw-infinity-cache](../hardware/infinity-cache.md) — what XCD-awareness is preserving
- [ROCm engineering blogs](https://rocm.blogs.amd.com/) — published case studies on chiplet-aware scheduling
- [pr-aiter-297](../../sources/prs/aiter/PR-420.md) — production use in fused MoE
