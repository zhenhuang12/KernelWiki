---
id: blog-rocm-xcd-aware-scheduling
title: "XCD-Aware Work-Group Scheduling on MI300X"
author: ROCm Developer Hub
url: https://rocm.blogs.amd.com/software-tools-optimization/xcd-aware-scheduling/README.html
source_category: benchmark-blog
architectures:
- cdna3
- cdna4
tags:
- xcd
- infinity-cache
- tile-scheduling
- xcd-aware-scheduling
retrieved_at: 2026-04-27
---

## Summary

ROCm Blogs case study on how default round-robin work-group dispatch across the 8 XCDs of MI300X destroys L2 locality, and how a one-line `blockIdx.x` remap can recover 25–50% throughput on cache-bound GEMMs.

## The Pattern

Default dispatch sequence (round-robin XCDs): WG 0 → XCD 0, WG 1 → XCD 1, …, WG 7 → XCD 7, WG 8 → XCD 0, …  
Result: adjacent K-tiles for the same M-row land on different XCDs and miss each other's L2 caches.

XCD-aware remap: cluster every 8 consecutive workgroups onto the same XCD, then move to the next XCD. The hardware still dispatches round-robin, so the user-visible `blockIdx.x` permutation is:

```cpp
__device__ int xcd_remap(int bid, int num_xcd = 8) {
    int xcd  = bid % num_xcd;     // physical XCD index from hardware
    int slot = bid / num_xcd;     // round within that XCD
    int xcd_tiles = (gridDim.x + num_xcd - 1) / num_xcd;
    return xcd * xcd_tiles + slot;  // logical tile index
}
```

The kernel then uses `xcd_remap(blockIdx.x)` everywhere it would normally use `blockIdx.x`. Adjacent logical tiles now share an XCD, share an L2.

## Measured Wins

- 4096×4096×4096 BF16 GEMM, K-stationary: +27% throughput.
- Attention prefill, head_dim=128, seq=8192: +18% throughput.
- Reductions / small-batch ops bounded by HBM bandwidth see no change (already memory-bound).

## Why It Matters

XCD-aware scheduling is the most universally-applicable AMD-specific optimization. It costs one line of HIP and applies to any cooperative kernel that reads more than the per-XCD L2 (4 MB) worth of operands. Cited by [technique-xcd-aware-scheduling](../../wiki/techniques/xcd-aware-scheduling.md), [hw-xcd](../../wiki/hardware/xcd.md).
