---
id: doc-amd-hip-programming-guide
title: "AMD HIP Programming Guide / ROCm Documentation"
url: https://rocm.docs.amd.com/projects/HIP/en/latest/programming_guide.html
source_category: official-doc
architectures:
- cdna3
- cdna4
tags:
- hip
- wavefront-64
- lds
- mfma
- s-barrier
- s-waitcnt
retrieved_at: 2026-04-27
---

# AMD HIP Programming Guide

## Overview

The official HIP (Heterogeneous Interface for Portability) programming guide and ROCm documentation set. HIP is the AMD C++ kernel language; it is source-compatible with a subset of CUDA C++ via `hipify-perl` / `hipify-clang` translation and adds AMD-specific intrinsics for matrix cores, wavefront operations, and direct-to-LDS.

## Key Concepts for Kernel Authors

### Launch Model

- **Workgroup** ≡ CUDA thread block, scheduled to one CU.
- **Wavefront** = 64 threads (CDNA) — vs NVIDIA's 32-thread warp. Wave-aware code must account for the larger lane count when packing tile dimensions.
- **Grid / block dimensions**: `dim3` identical to CUDA; `__global__`, `__device__`, `__shared__` keywords carry over verbatim.

### Memory Hierarchy

- `__shared__` → LDS (64 KB on CDNA 3, 160 KB on CDNA 4 per CU).
- Global memory accessed via standard `*` dereferences or `__builtin_amdgcn_global_load_*` for explicit cache-policy control.
- No equivalent of NVIDIA `__constant__` (HIP maps to read-only globals).

### Matrix Core Access

Three layers, finest to coarsest:

1. `__builtin_amdgcn_mfma_*` clang intrinsics (one-to-one with `v_mfma_*` instructions). Used by performance kernels and CK templates.
2. `rocwmma::` C++ headers (`<rocwmma/rocwmma.hpp>`) — portable WMMA-style API.
3. `composable_kernel::tile_program::mfma_tile` — high-level wave-level tile that auto-selects shape.

### Wavefront Intrinsics

```cpp
__shfl(value, srcLane);            // 64-lane shuffle
__shfl_xor(value, mask);
__ballot(predicate);               // 64-bit ballot (vs 32-bit on NVIDIA)
__any(predicate); __all(predicate);
__builtin_amdgcn_readfirstlane(v); // pin VGPR → SGPR (scalar)
```

### Synchronization

```cpp
__syncthreads();                              // → s_barrier
__threadfence();                              // → memory ordering
__builtin_amdgcn_s_waitcnt(0);                // explicit vmcnt/lgkmcnt drain
__builtin_amdgcn_sched_barrier(mask);         // compiler scheduling barrier
__builtin_amdgcn_s_setprio(prio);             // wave priority hint
```

Notably absent: `__syncwarp()` has no direct equivalent (wavefronts run in lockstep on CDNA, so `__syncwarp` is implicitly a no-op for divergence-free code — but cross-wave divergence resyncs require `s_barrier`).

## Hipify Translation Notes

CUDA→HIP source porting is mostly mechanical:

- `cudaXxx` → `hipXxx` (host runtime).
- `cublas`, `cufft` → `hipblas`, `hipfft` (or `hipblaslt` for the LT-style GEMM API).
- `__shfl_sync(mask, …)` → `__shfl(…)` (no mask; wave is implicit).
- `cuda_fp16.h` → `hip_fp16.h`.

What *does not* translate automatically: TMA / cp.async.bulk (use `buffer_load_lds`), thread-block-cluster operations (no AMD equivalent), `wgmma`/`tcgen05` (use MFMA), and any PTX inline-asm (rewrite as AMDGCN inline asm).

## Why It Matters

HIP is the entry-point language for AMD kernel work. Everything in CK, AITER, and rocBLAS is HIP at the top layer with AMDGCN inline-asm or MFMA intrinsics at the hot inner loops. See [lang-hip](../../wiki/languages/hip.md), [migration-cuda-to-hip](../../wiki/migration/cuda-to-hip.md).
