---
id: migration-cuda-to-hip
title: "Migrating CUDA Kernels to HIP / CDNA"
type: migration
from_arch: sm90
to_arch: cdna3
tags: [hip, mfma, lds, buffer-load-lds, wave-specialization, mfma-pipelining, xcd-aware-scheduling]
related: [lang-hip, lang-composable-kernel, hw-mfma, hw-buffer-load-lds, hw-lds, hw-agpr, hw-xcd, technique-wave-specialization, technique-mfma-pipelining, technique-direct-to-lds, technique-xcd-aware-scheduling]
sources: [doc-amd-hip-programming-guide, doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-mfma-tutorial, blog-gpuopen-mi300-architecture, doc-amd-composable-kernel]
blackwell_relevance: "Counterpart guide for NVIDIA SM90/SM100 users porting kernels to AMD Instinct MI300X/MI355X. Pairs with migration-wgmma-to-tcgen05 conceptually — both document the asynchronous-MMA paradigm shift that AMD's wave-specialized direct-to-LDS pipeline approximates with different primitives."
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma, hw-buffer-load-lds, hw-lds, hw-xcd]
---

# Migrating CUDA Kernels to HIP / CDNA

## Overview

Porting a CUDA kernel to HIP/CDNA is two distinct efforts stacked on top of each other:

1. **Mechanical translation**: rename `cuda*` → `hip*`, swap `__half` → `__hip_bfloat16` patterns, run `hipify-clang`. This gets you a *working* kernel.
2. **Architecture re-platforming**: replace `cp.async` with `buffer_load_*_lds`, `wgmma`/`tcgen05` with MFMA, warp specialization with wave specialization, and apply XCD-aware tile mapping. This gets you a *fast* kernel.

Skipping step 2 is the most common reason "ported" CUDA kernels reach only 30-40% of MI300X peak.

## Migration Checklist

```
Mechanical (hours):
  [ ] hipify-clang on source files
  [ ] swap headers: cuda_runtime.h        -> hip/hip_runtime.h
                    cuda_bf16.h           -> hip/hip_bf16.h
                    cooperative_groups.h  -> hip/hip_cooperative_groups.h
  [ ] check warp size: 32 -> 64 (every divisor in mask/shuffle code)
  [ ] CUB/Thrust -> hipCUB/rocThrust
  [ ] NCCL       -> RCCL (drop-in symbol-compatible)
  [ ] CUTLASS    -> Composable Kernel (no drop-in; see CK-Tile)

Architectural (days, per kernel):
  [ ] MMA: wgmma / tcgen05.mma -> v_mfma_* via __builtin_amdgcn_mfma_*
  [ ] Async global load: cp.async.bulk (TMA) -> buffer_load_dwordx4_lds (direct-to-LDS;
        CDNA 4 / gfx950 only — on CDNA 3 / gfx942 the widest direct-to-LDS is the 32-bit
        buffer_load_dword_lds, so issue 4x as many loads)
  [ ] Mbarrier / commit/wait -> s_waitcnt vmcnt(0) + s_barrier
  [ ] Warp specialization -> wave specialization (no __syncwarp; use s_barrier)
  [ ] SMEM bank padding -> LDS XOR swizzle
  [ ] Output tile scheduling -> XCD-aware blockIdx.x remap
  [ ] Tile sizes: 128x128x32 (Hopper) -> 256x256x64 (CDNA 3) / 256x256x128 (CDNA 4)
  [ ] Register accumulators in VGPR -> AGPR (16 per 32x32x16 MFMA)
```

## Concept Mapping

| CUDA / Hopper / Blackwell | HIP / CDNA 3 / CDNA 4 | Notes |
|---------------------------|------------------------|-------|
| Warp = 32 threads | Wave = 64 threads | Affects every mask, shuffle, ballot |
| `__syncwarp()` | none — use `s_barrier` workgroup-wide | The biggest wave-specialization friction |
| `wgmma.mma_async` / `tcgen05.mma` | `v_mfma_*` (synchronous) | No async commit/wait; ILP via interleaving |
| Register / TMEM accumulators | AGPR (shared 512-reg pool) | See [hw-agpr](../hardware/agpr.md) |
| `cp.async.ca/cg` (SM80) | `buffer_load_dword_lds` (CDNA 3) | 32 b/lane writer to LDS |
| `cp.async.bulk` / TMA (SM90/100) | `buffer_load_dwordx4_lds` (CDNA 4) | 128 b/lane — closest TMA analogue |
| Mbarrier `expect_tx` + wait | `s_waitcnt vmcnt(0)` then `s_barrier` | No per-warp barrier; whole workgroup |
| `__shfl_sync` (32-wide) | `__shfl` (64-wide on AMD) | Lane mask widths differ |
| Tensor Memory (TMEM) | LDS + AGPR | TMEM has no AMD equivalent |
| `cp.async.commit_group` | (no analogue — `s_waitcnt` is in-order) | Counter-based, not group-based |
| Cluster (`__cluster_dims__`) | none — XCD-aware tile scheduling | Software remap, no hardware cluster |
| SM (132 on H100 SXM5, 148 on B200) | CU (304 on MI300X = 38/XCD × 8 XCDs, 256 on MI355X = 32/XCD × 8 XCDs) | Higher CU count, distributed across chiplets |
| L2 (50 MB on H100 SXM5, ~100 MB across the two B200 dies) | L2 (4 MB per XCD × 8) + Infinity Cache (256 MB on MI300X) | Two-level on-package cache |

## Worked Example: WGMMA GEMM → MFMA GEMM

### Before (Hopper / CUDA C++)

```cpp
__global__ void gemm_hopper(const __half* A, const __half* B, float* C, ...) {
    __shared__ alignas(128) __half smem_a[2][128 * 32];
    __shared__ alignas(128) __half smem_b[2][32 * 128];

    cooperative_groups::thread_block_tile<128> wg =
        cooperative_groups::tiled_partition<128>(this_block());

    float acc[64];   // register accumulator
    for (int k = 0; k < K; k += 32) {
        int stage = (k / 32) & 1;
        cp_async_bulk(smem_a[stage], A + k * 128, mbar);
        cp_async_bulk(smem_b[stage], B + k * 128, mbar);
        mbar.arrive_and_wait();
        wgmma_mma_async(acc, smem_a[stage], smem_b[stage]);
        wgmma_commit_group();
    }
    wgmma_wait_group<0>();
    store_acc_to_global(C, acc);
}
```

### After (CDNA 3 / HIP, wave-specialized)

```cpp
__global__ void __launch_bounds__(256) gemm_cdna3(
    const __hip_bfloat16* A, const __hip_bfloat16* B, float* C, int M, int N, int K) {

    int wave = threadIdx.x / 64;
    int lane = threadIdx.x % 64;

    // XCD-aware tile remap
    int logical_bid = xcd_remap(blockIdx.x);
    int m_tile = (logical_bid / (N / 256)) * 256;
    int n_tile = (logical_bid % (N / 256)) * 256;

    // 2 × (256·64 + 64·256) × sizeof(bf16) = 128 KB of LDS — fits CDNA 4
    // (gfx950, 160 KB/CU) but exceeds CDNA 3's 64 KB budget. On gfx942 drop
    // to 1 stage or shrink BLOCK_M/BLOCK_N to 128 to fit.
    __shared__ __hip_bfloat16 smem_a[2][256 * 64];     // 2 stages, CDNA 4 only
    __shared__ __hip_bfloat16 smem_b[2][64 * 256];

    using f32x16 = __attribute__((__vector_size__(64))) float;

    if (wave < 2) {
        // ===== producer waves =====
        for (int k = 0, stage = 0; k < K; k += 64, stage ^= 1) {
            direct_to_lds_load(smem_a[stage], A + m_tile * K + k, wave, lane);
            direct_to_lds_load(smem_b[stage], B + k * N + n_tile, wave, lane);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            __syncthreads();           // hand off to consumers
            __syncthreads();           // wait for consumers to drain
        }
    } else {
        // ===== consumer waves =====
        f32x16 acc = {0};              // AGPR accumulator (16 AGPRs/lane)
        for (int k = 0, stage = 0; k < K; k += 64, stage ^= 1) {
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < 64; kk += 16) {
                auto a = ds_read_b128(smem_a[stage], wave, lane, kk);
                auto b = ds_read_b128(smem_b[stage], wave, lane, kk);
                acc = __builtin_amdgcn_mfma_f32_32x32x16bf16(a, b, acc, 0, 0, 0);
            }
            __syncthreads();
        }
        store_acc_to_global(C, acc, wave, lane);
    }
}
```

The mechanical port (replace `__half` → `__hip_bfloat16`, `wgmma_mma_async` → some HIP MMA wrapper) would compile and run but reach maybe 35% of MI300X peak. The wave-specialized + direct-to-LDS + XCD-aware version reaches 76-80%.

## Don't-Bother List

Things that look like they should port but rarely earn their keep on CDNA:

- **`cuda::pipeline` / pipeline stages abstraction**: CK-Tile's wave-spec pipeline is structurally different; trying to express CDNA pipelines in `cuda::pipeline`-style abstractions just hides the wave-id branching that needs to be explicit.
- **Asynchronous copy with `commit_group`**: CDNA's `vmcnt` counter is in-order and cumulative; the SM80 group-based async model doesn't map.
- **Cluster launch / DSMEM**: no hardware equivalent. The closest analogue is XCD-aware tile scheduling, which is solving a different problem (cache locality, not cross-CTA SMEM access).

## Bridge: Stay in CUDA Source via HIPIFY?

`hipify-clang` produces source that compiles with both `hipcc` and `nvcc`-with-shims. This is fine for tooling/build-system reasons but does not buy any performance — the hipified kernel still uses the original CUDA algorithm. Architectural rewrites (the second checklist above) cannot be expressed in a portable subset.

For new code targeting both architectures, prefer:

- **High-level DSL**: Triton, FlyDSL, or CK-Tile (with parallel CUTLASS implementation).
- **Two implementations**: maintain a CUTLASS GEMM and a CK-Tile GEMM side by side. CUTLASS-style template kernels rarely port well to AMD as-is.

## See Also

- [lang-hip](../languages/hip.md) — HIP API cheat sheet
- [lang-composable-kernel](../languages/composable-kernel.md) — CK-Tile as the CUTLASS equivalent
- [migration-wgmma-to-tcgen05](wgmma-to-tcgen05.md) — the NVIDIA-internal counterpart
- [technique-wave-specialization](../techniques/wave-specialization.md)
- [technique-xcd-aware-scheduling](../techniques/xcd-aware-scheduling.md)
