---
id: hw-lds
title: "LDS — AMD Local Data Share (Shared Memory)"
type: hardware
architectures: [cdna3, cdna4]
tags: [lds, wavefront-64]
confidence: source-reported
related: [hw-mfma, hw-buffer-load-lds, technique-lds-swizzling, technique-mfma-pipelining, lang-hip, lang-amdgcn-asm]
sources: [doc-amd-cdna3-whitepaper, doc-amd-cdna4-whitepaper, doc-amd-cdna3-isa, blog-rocm-mfma-tutorial]
aliases: [LDS, "Local Data Share", "AMD shared memory", "shared memory (AMD)", ds_read, ds_write]
---

# LDS — AMD Local Data Share

## Overview

LDS is CDNA's per-compute-unit shared memory. Functionally equivalent to NVIDIA's `__shared__` (SMEM), but with different sizing, bank layout, and direct-load capability:

| Property | NVIDIA Hopper SMEM | CDNA 3 LDS | CDNA 4 LDS |
|----------|---------------------|------------|------------|
| Size per SM/CU | 228 KB (configurable) | 64 KB | 160 KB |
| Banks | 32 × 4 B | 32 × 4 B | 32 × 4 B |
| Peak bandwidth | 128 B/cycle | 128 B/clk (~268 GB/s/CU @ ~2.1 GHz) | ~537 GB/s per CU (derived: 256 B/clk × ~2.1 GHz; see `blog-rocm-cdna4-gemm-kernels`) |
| Async global→shmem | cp.async.bulk (TMA) | buffer_load_lds (32 b/lane) | buffer_load_dwordx4_lds (128 b/lane) |
| Access | ld.shared / st.shared | ds_read_* / ds_write_* | ds_read_* / ds_write_* |

## Bank Conflict Model

Identical to NVIDIA: 32 banks × 4 B; lanes accessing different addresses in the same bank serialize. Standard avoidance patterns (padding, XOR swizzles) apply.

For 16×16 MFMA tiles in BF16/FP16, an XOR swizzle `addr = base + (row * stride) ^ ((row & 7) << log2(bank_width))` reaches full 128 B/cycle bandwidth. CK-Tile codegen produces this pattern automatically when `LdsDescriptor` is configured with `SwizzleXor=true`.

## Idiomatic Access

```cpp
// Wide LDS load (128 b/lane) via ds_read_b128
auto v = __builtin_amdgcn_ds_read_b128(lds_ptr_addr);

// HIP-level (compiler emits ds_read_b128 when alignment + type allow)
__shared__ float4 tile[64];   // 1 KB LDS
float4 v = tile[lane];

// Wide LDS store
__builtin_amdgcn_ds_write_b128(lds_ptr_addr, v);
```

`ds_read_b128` is the workhorse for staging MFMA operands; pair with XOR-swizzled layouts to keep all 64 lanes hitting distinct banks.

## Direct-to-LDS Loads (the AMD TMA-equivalent)

The unique CDNA capability is direct-to-LDS loads — global memory streams straight into LDS without VGPR round-trip:

- **CDNA 3**: `buffer_load_dword_lds` (32 b / lane), `global_load_lds_dword` (32 b / lane).
- **CDNA 4**: `buffer_load_dwordx4_lds` (128 b / lane).

This is the most important CDNA capability for kernel pipelining — see [hw-buffer-load-lds](buffer-load-lds.md) for details.

## CDNA 4 Implications of 160 KB LDS

The 2.5× LDS growth in CDNA 4 enables:

- Deeper producer/consumer pipelines (4 stages × 32 KB tile, vs 2 stages on CDNA 3).
- Larger persistent-K caches for K-stationary GEMMs.
- MoE expert weight residency for small-batch decoding (entire expert tile fits in LDS).

## Caveats

- LDS is per-CU, not per-XCD. Cross-CU sharing requires HBM / Infinity Cache.
- `s_barrier` synchronizes only within the workgroup mapped to one CU; cross-workgroup synchronization needs atomics on global memory.
- LDS is uninitialized at workgroup start; bulk-zero with `ds_write_b128` if needed.

## See Also

- [hw-buffer-load-lds](buffer-load-lds.md) — direct global-to-LDS loads
- [technique-lds-swizzling](../techniques/lds-swizzling.md) — XOR / Morton patterns
- [hw-mfma](mfma.md) — MFMA operand staging from LDS
