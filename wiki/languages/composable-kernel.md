---
id: lang-composable-kernel
title: "Composable Kernel (CK / CK-Tile) — AMD's CUTLASS-Equivalent"
type: language
tags: [composable-kernel, mfma, mfma-scale-f8f6f4, lds, buffer-load-lds, gemm, attention, hip]
related: [lang-hip, lang-amdgcn-asm, hw-mfma, hw-mfma-scale-f8f6f4, hw-lds, hw-buffer-load-lds, technique-wave-specialization, technique-mfma-pipelining, kernel-ck-fp8-gemm-cdna3, kernel-ck-mxfp4-gemm-cdna4]
sources: [doc-amd-composable-kernel, blog-rocm-ck-flash-attention, blog-rocm-mi355-mxfp4-launch, pr-composable-kernel-1384, pr-composable-kernel-3098, pr-composable-kernel-2110]
reproducibility: snippet
architectures: [cdna3, cdna4]
confidence: source-reported
aliases: [CK, "Composable Kernel", "CK-Tile", CKTile]
---

# Composable Kernel (CK / CK-Tile)

## Overview

[ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) is AMD's header-only C++ template library that composes high-performance GPU kernels from reusable tile programs. Two coexisting layers:

- **CK (classic)** — original tensor-program library used by rocBLAS, MIOpen, migraphx. Heavy template metaprogramming.
- **CK-Tile** — newer per-wave tile-program DSL, simpler API, closer to CuTe's tile/copy/MMA abstractions. Where new CDNA 3/4 kernels live.

CK-Tile fills the same niche on AMD that CUTLASS 4.x fills on NVIDIA: a reusable C++ substrate that AITER, vLLM AMD path, SGLang AMD path, and downstream research kernels build on.

## Architecture

A CK-Tile kernel is parameterized by:

1. **Problem traits** — datatypes, layouts, block tile shape.
2. **Block pipeline** — V1 (single-stage), V2 (double-buffered), V3 (interwave / wave-specialized).
3. **MFMA shape** — `<16,16,32>`, `<32,32,16>`, etc.
4. **LDS descriptor** — XOR-swizzle parameters, padding policy.
5. **Scheduler** — interleaves loads, MFMAs, and stores via `__builtin_amdgcn_sched_barrier`.

## CK-Tile GEMM Skeleton

```cpp
#include <ck_tile/core.hpp>
#include <ck_tile/ops/gemm.hpp>

using namespace ck_tile;

// 1. Problem: BF16 inputs, FP32 accumulator, BF16 output
using Problem = GemmProblem<
    /*ADataType=*/bf16_t,
    /*BDataType=*/bf16_t,
    /*AccDataType=*/float,
    /*CDataType=*/bf16_t,
    /*ALayout=*/tensor_layout::gemm::RowMajor,
    /*BLayout=*/tensor_layout::gemm::ColumnMajor,
    /*CLayout=*/tensor_layout::gemm::RowMajor,
    /*BlockTile=*/sequence<256, 256, 64>,
    /*WarpTile=*/sequence<32, 32, 16>,
    /*Scheduler=*/GemmPipelineSchedulerV3>;   // wave-specialized

// 2. Compose a kernel from problem + epilogue
using Kernel = GemmKernel<Problem, DefaultEpilogue>;

// 3. Launch
auto kargs = Kernel::MakeKargs(a_ptr, b_ptr, c_ptr, M, N, K, stride_a, stride_b, stride_c);
Kernel::Launch(stream, kargs);
```

The V3 scheduler issues a 4-wave producer/consumer split (2 producer waves driving `buffer_load_dword_lds`, 2 consumer waves driving MFMAs) automatically, with the LDS XOR-swizzle, AGPR budget, and `s_waitcnt` placement chosen by the template machinery.

## CK-Tile Attention (FMHA) Skeleton

```cpp
#include <ck_tile/ops/fmha.hpp>

using FmhaProblem = FmhaFwdProblem<
    /*QDataType=*/bf16_t,
    /*KDataType=*/bf16_t,
    /*VDataType=*/bf16_t,
    /*OAccDataType=*/float,
    /*ODataType=*/bf16_t,
    /*HeadDim=*/128,
    /*BlockTile=*/FmhaShape<128, 128>,   // M_q × M_kv per CTA
    /*Pipeline=*/FmhaFwdPipelineV3>;     // wave-specialized

using Kernel = FmhaFwdKernel<FmhaProblem>;
Kernel::Launch(stream, kargs);
```

The same V3 scheduler abstraction generalizes from GEMM to attention — 2 waves load K/V tiles into LDS, 2 waves run online softmax + QK + AV MFMAs.

## When to Use What

| Need | Use |
|------|-----|
| Production-grade GEMM / FMHA | CK-Tile with V3 scheduler |
| Custom epilogue (per-token quant, gating) | CK-Tile + custom `Epilogue` template |
| MXFP4 / MXFP6 on CDNA 4 | `BlockGemmPipelineMxFp4` / `MxFp6` |
| Convolution / batched primitives | CK classic |
| Quick experiment, don't care about peak | Triton-AMD or plain HIP |

## Caveats

- Compile times are long — CK templates instantiate aggressively. Use `-fno-template-backtrace-limit=0` and parallel builds.
- Error messages are CUTLASS-class horrors. Save your sanity by always starting from a working example under `example/ck_tile/`.
- `BlockGemmPipelineSchedulerV3` requires gfx942+ (CDNA 3+) — older targets fall back to V2.

## See Also

- [lang-hip](hip.md) — base language layer
- [hw-mfma](../hardware/mfma.md) — MFMA instruction family CK wraps
- [kernel-ck-fp8-gemm-cdna3](../kernels/ck-fp8-gemm-cdna3.md) — reference kernel
- [doc-amd-composable-kernel](../../sources/docs/amd-composable-kernel-readme.md) — upstream README
