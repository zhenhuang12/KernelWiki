---
id: lang-hip
title: "HIP — AMD's C++ Kernel Language"
type: language
tags: [hip, mfma, lds, wavefront-64, amdgcn-asm]
related: [lang-composable-kernel, lang-amdgcn-asm, lang-cuda-cpp, hw-mfma, hw-lds, hw-buffer-load-lds, migration-cuda-to-hip]
sources: [doc-amd-hip-programming-guide, doc-amd-cdna3-isa, blog-rocm-mfma-tutorial]
reproducibility: snippet
architectures: [cdna3, cdna4]
confidence: source-reported
aliases: [HIP, "Heterogeneous Interface for Portability", hipcc]
---

# HIP — AMD's C++ Kernel Language

## Overview

HIP (Heterogeneous Interface for Portability) is AMD's C++ kernel language. It is source-compatible with most of CUDA C++ (compiles either via `hipcc` to AMDGCN for AMD or via NVCC to PTX for NVIDIA) and adds AMD-specific intrinsics for MFMA, direct-to-LDS, AMDGCN inline assembly, wave-priority hints, and scheduling barriers.

HIP is the entry-point language for AMD kernel work — everything in CK, AITER, rocBLAS, and MIOpen is HIP at the top layer.

## CUDA → HIP Cheat Sheet

| CUDA | HIP | Notes |
|------|-----|-------|
| `__global__` | `__global__` | identical |
| `__shared__` | `__shared__` | → LDS |
| `__syncthreads()` | `__syncthreads()` | → `s_barrier` |
| `__syncwarp(mask)` | `__syncthreads()` or omit | wavefronts are lockstep; no per-warp barrier |
| `__shfl_sync(mask, v, src)` | `__shfl(v, src)` | mask implicit; wave is 64 lanes |
| `__ballot_sync(mask, p)` | `__ballot(p)` | 64-bit result on CDNA |
| `cudaMalloc` | `hipMalloc` | runtime API |
| `cuBLAS` | `hipBLAS` / `hipBLASLt` | hipBLASLt for LT-style API |
| `cp.async.bulk` (TMA) | `__builtin_amdgcn_raw_buffer_load_lds` | per-lane, not descriptor-driven |
| `wgmma.mma_async` | `__builtin_amdgcn_mfma_*` | single wave, sync; cf. [hw-mfma](../hardware/mfma.md) |
| inline PTX | inline AMDGCN asm | cf. [lang-amdgcn-asm](amdgcn-asm.md) |

`hipify-perl` / `hipify-clang` do most of this translation mechanically. The hard parts that don't auto-translate are noted in [migration-cuda-to-hip](../migration/cuda-to-hip.md).

## Idiomatic HIP MFMA GEMM Skeleton

```cpp
// HIP BF16 GEMM with MFMA — one CTA = one block tile
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

constexpr int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32;
constexpr int WARP_M  = 32,  WARP_N  = 32;
constexpr int MFMA_M  = 16,  MFMA_N  = 16, MFMA_K = 16;

__global__ void __launch_bounds__(256)
hip_gemm_bf16(const __hip_bfloat16* A, const __hip_bfloat16* B,
               float* C, int M, int N, int K) {
    using bf16x4 = __attribute__((__vector_size__(8))) __hip_bfloat16;
    using f32x4  = __attribute__((__vector_size__(16))) float;

    const int wave_id = threadIdx.x / 64;
    const int lane    = threadIdx.x % 64;

    __shared__ __hip_bfloat16 smem_a[BLOCK_M * BLOCK_K];
    __shared__ __hip_bfloat16 smem_b[BLOCK_K * BLOCK_N];

    f32x4 acc = {0, 0, 0, 0};

    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Cooperative load (every lane participates) — direct-to-LDS in production code
        cooperative_load(smem_a, A + k_tile, M, K);
        cooperative_load(smem_b, B + k_tile * N, K, N);
        __syncthreads();

        // Wave-level MFMA inner loop
        #pragma unroll
        for (int kk = 0; kk < BLOCK_K; kk += MFMA_K) {
            bf16x4 a_frag = lds_load_a_frag(smem_a, wave_id, lane, kk);
            bf16x4 b_frag = lds_load_b_frag(smem_b, wave_id, lane, kk);
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                a_frag, b_frag, acc, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
        }
        __syncthreads();
    }

    // Epilogue: store accumulator (AGPRs auto-copied to VGPR by compiler)
    store_acc_to_global(C, acc, wave_id, lane);
}
```

## Wave-Level Intrinsics

```cpp
__shfl(v, src);                       // 64-lane shuffle
__shfl_xor(v, mask);
__ballot(predicate);                  // returns uint64_t
__any(predicate); __all(predicate);
__builtin_amdgcn_readfirstlane(v);    // VGPR → SGPR
```

## Synchronization / Scheduling Intrinsics

```cpp
__syncthreads();                                 // s_barrier
__threadfence();                                 // memory ordering
__builtin_amdgcn_s_waitcnt(0);                   // drain all counters
__builtin_amdgcn_sched_barrier(MASK);            // constrain compiler scheduling
__builtin_amdgcn_s_setprio(prio);                // wave priority 0-3
__builtin_amdgcn_s_sleep(N);                     // wait N cycles
```

`__builtin_amdgcn_sched_barrier` is the AMD analogue of NVIDIA `__syncwarp` for forcing the compiler to fence instruction reordering — but its mask field gives fine-grained control over which instruction classes can cross.

## Build / Run

```bash
hipcc -O3 --offload-arch=gfx942 -o kernel kernel.cpp    # CDNA 3 / MI300X
hipcc -O3 --offload-arch=gfx950 -o kernel kernel.cpp    # CDNA 4 / MI355X
```

Multi-arch fat binaries: `--offload-arch=gfx942,gfx950`.

## Caveats

- The `_sync` variants of `__shfl` / `__ballot` are not standard HIP; use the unmasked forms.
- HIP `dim3` is column-major (x is innermost), same as CUDA — but the wavefront is 64 lanes, so `threadIdx.x % 64` is the lane id.
- `__constant__` memory exists in HIP but is implemented as a read-only global; no dedicated cache as on NVIDIA.

## See Also

- [lang-composable-kernel](composable-kernel.md) — C++ template DSL built on HIP
- [lang-amdgcn-asm](amdgcn-asm.md) — inline-asm escape hatch
- [migration-cuda-to-hip](../migration/cuda-to-hip.md) — porting guide
- [doc-amd-hip-programming-guide](../../sources/docs/amd-rocm-hip-docs.md) — official reference
