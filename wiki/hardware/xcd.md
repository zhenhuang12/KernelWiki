---
id: hw-xcd
title: "XCD — Accelerator Complex Die (CDNA Chiplet)"
type: hardware
architectures: [cdna3, cdna4]
tags: [xcd, infinity-cache]
confidence: source-reported
related: [hw-infinity-cache, technique-xcd-aware-scheduling, technique-tile-scheduling]
sources: [doc-amd-cdna3-whitepaper, doc-amd-cdna4-whitepaper, blog-gpuopen-mi300-architecture, blog-rocm-xcd-aware-scheduling]
aliases: [XCD, "Accelerator Complex Die", "AMD chiplet", "compute die"]
---

# XCD — Accelerator Complex Die

## Overview

CDNA 3 / CDNA 4 GPUs are chiplet-based: each GPU contains 8 Accelerator Complex Dies (XCDs), each XCD packages its own L2 cache (4 MB) and shares access to the central 256 MB Infinity Cache (MALL). Per-XCD CU counts differ by generation: MI300X (CDNA 3) has 38 active CUs per XCD (40 physical, 2 disabled for yield) → 304 active CUs total; MI355X (CDNA 4) has 32 active CUs per XCD (36 physical) → 256 active CUs total. This is structurally different from NVIDIA's monolithic SM topology.

## Topology

```
   MI300X (CDNA 3) — diagram represents MI300X specifically
   ┌─────────────────────────┐
   │  XCD0  XCD1  XCD2  XCD3 │   ← 4 XCDs above
   │   ↕     ↕     ↕     ↕   │
   │  ╔═════════════════════╗│   ← 256 MB Infinity Cache (MALL)
   │   ↕     ↕     ↕     ↕   │
   │  XCD4  XCD5  XCD6  XCD7 │   ← 4 XCDs below
   └─────────────────────────┘
   MI300X (CDNA 3): 40 physical / 38 active CUs per XCD (304 total),
   4 MB private L2, 8 HBM3 stacks shared.
```

CDNA 4 (MI355X) keeps the same 8-XCD layout (same diagram applies) but with a smaller per-XCD CU count and HBM3E stacks — see the MI355X CU counts called out in the Overview above (32 active / 36 physical CUs per XCD, 256 active CUs total).

There is no equivalent on NVIDIA Hopper or Blackwell — the closest analogue is the GB200 "die-pair" but the unit of scheduling is still the SM, not a die.

## Scheduling Implications

Work-groups are dispatched round-robin across XCDs by default:

```
WG 0 → XCD 0,  WG 1 → XCD 1, ..., WG 7 → XCD 7
WG 8 → XCD 0,  WG 9 → XCD 1, ..., WG 15→ XCD 7
```

For a tile-major GEMM where adjacent `blockIdx.x` work on adjacent K-tiles of the same M-row, this scatter destroys L2 locality — each XCD's 4 MB L2 holds a fragment of unrelated work.

**XCD-aware remap** (one-line fix):

```cpp
__device__ int xcd_remap(int bid, int num_xcd = 8) {
    int xcd  = bid % num_xcd;
    int slot = bid / num_xcd;
    int xcd_tiles = (gridDim.x + num_xcd - 1) / num_xcd;
    return xcd * xcd_tiles + slot;   // logical → physical
}
int logical_bid = xcd_remap(blockIdx.x);
```

Adjacent logical tiles now share an XCD and therefore an L2; cache hit-rates jump from ~5% to ~60% on cache-bound GEMMs, with measured throughput gains of 18-50%. See [technique-xcd-aware-scheduling](../techniques/xcd-aware-scheduling.md).

## Compute Partitioning (CPX / DPX)

CDNA 3/4 support compute-partition (CPX) and memory-partition (DPX) modes that logically subdivide the GPU into 1/2/4/8 partitions. Each partition gets a contiguous subset of XCDs and HBM stacks. CPX-8 is one XCD per partition — useful for tenant isolation and small-batch inference where a single workload cannot fill the full die (304 CUs on MI300X, 256 on MI355X). Kernels written without partition awareness simply see fewer CUs (the runtime hides the partition split).

## Cache Hierarchy

| Level | Size | Scope | Bandwidth |
|-------|------|-------|-----------|
| L1 (vector) | 32 KB | Per-CU | per-CU peak |
| L2 | 4 MB | Per-XCD | ~4.3 TB/s per XCD (~34 TB/s aggregate across 8 XCDs) |
| Infinity Cache (MALL) | 256 MB | Per-GPU | ~17 TB/s aggregate |
| HBM3 / HBM3E | 192 GB / 288 GB | Per-GPU | 5.3 / 8.0 TB/s |

K-stationary GEMMs that fit operand B in Infinity Cache (256 MB total) but not in per-XCD L2 (4 MB / XCD) are where XCD-aware scheduling matters most.

## Caveats

- The hardware always dispatches round-robin; you can't disable that — you can only permute `blockIdx.x` to match.
- Persistent kernels (gridDim.x == num_cu) sidestep most of the issue, since each CU keeps its tile-range stationary.
- Cross-XCD communication is via Infinity Cache / HBM — no LDS-style fast path.

## See Also

- [technique-xcd-aware-scheduling](../techniques/xcd-aware-scheduling.md) — the remap pattern in production
- [hw-infinity-cache](infinity-cache.md) — last-level cache shared by all XCDs
- [blog-gpuopen-mi300-architecture](../../sources/blogs/gpuopen-mi300-architecture.md) — measured cache behavior
