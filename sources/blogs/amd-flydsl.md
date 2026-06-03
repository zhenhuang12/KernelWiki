---
id: blog-amd-flydsl
title: "FlyDSL — AMD's Flexible Layout DSL for GPU Tile Programs"
author: AMD Research
url: https://github.com/ROCm/FlyDSL
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- flydsl
- composable-kernel
- mfma
- lds
- hip
retrieved_at: 2026-04-27
---

## Summary

[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) is AMD's research Python+MLIR DSL for authoring GPU tile programs targeting CDNA 3/4. Roughly the AMD analogue of NVIDIA's CuTe-DSL or Triton — Python front-end, layout-aware tile abstractions, lowers through MLIR (`amdgpu`, `rocdl`, `gpu` dialects) to AMDGCN.

## Programming Model

Kernels are authored with the `@flyc.kernel` / `@flyc.jit` decorators over CuTe-style layout
algebra in the `fly` MLIR dialect:

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu

@flyc.kernel
def vec_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int]):
    idx = gpu.block_idx.x * 256 + gpu.thread_idx.x
    # ... layout ops (make_layout/crd2idx), copy/MMA atoms, buffer intrinsics ...

@flyc.jit
def vec_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int],
            stream: fx.Stream = fx.Stream(None)):
    vec_add_kernel(A, B, C, N).launch(grid=(N // 256,), block=(256,), stream=stream)
```

The author declares LDS swizzle, MFMA-shape, and pipeline structure via layouts; the compiler
lowers them (it does not auto-tune these — autotuning is roadmap, see [lang-flydsl](../../wiki/languages/flydsl.md)).
Lowering picks MFMA atoms against the target ISA's catalogue (cf. [doc-amd-cdna3-isa](../docs/amd-cdna3-isa.md),
[doc-amd-cdna4-isa](../docs/amd-cdna4-isa.md)) and emits the appropriate intrinsic chain. The
concrete tile-GEMM machinery (preshuffle B, XOR16 swizzle, ping-pong LDS) is captured in
[blog-amd-flydsl-gemm](amd-flydsl-gemm.md).

## Status

Pre-1.0 research project. Most production AMD kernels are still authored in CK-Tile C++; FlyDSL is the AMD bet on a high-level tile DSL that can match CK-Tile performance with less code. Read for design ideas; not yet production-ready.

## Why It Matters

FlyDSL is the AMD answer to "what plays the role of Triton / CuTe-DSL on Instinct?" — the highest-level layer that still emits hand-tuned-quality MFMA + LDS pipelines. Cited by [lang-flydsl](../../wiki/languages/flydsl.md).
