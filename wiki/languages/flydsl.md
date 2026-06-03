---
id: lang-flydsl
title: "FlyDSL — AMD Flexible LaYout DSL"
type: language
tags: [flydsl, composable-kernel, mfma, lds, hip, preshuffle, jit-compilation]
related: [lang-hip, lang-composable-kernel, hw-mfma, hw-lds, technique-preshuffle-gemm, technique-lds-swizzling, technique-wave-specialization]
sources: [blog-amd-flydsl, blog-amd-flydsl-gemm, pr-aiter-3117, doc-amd-cdna3-isa, doc-amd-cdna4-isa]
reproducibility: snippet
architectures: [cdna3, cdna4]
confidence: source-reported
aliases: [FlyDSL, "Flexible LaYout DSL", "AMD FlyDSL", "Flexible layout python DSL"]
---

# FlyDSL — AMD Flexible LaYout DSL

## Overview

[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) is AMD's MLIR-native Python DSL for authoring
GPU tile programs. It is the AMD analogue of NVIDIA's CuTe-DSL / Triton: a Python front-end
(`@flyc.kernel` / `@flyc.jit`) over the **`fly` MLIR dialect** — a first-class layout IR with
CuTe-style layout algebra (`composition` / `product` / `divide` / coordinate mapping). Kernels
trace from Python to MLIR and lower through `Fly → ROCDL → LLVM → fatbin`.

Verified targets: **gfx942** (MI300X/MI308X), **gfx950** (MI350/MI355X), **gfx1250** (MI450),
**gfx1201** (Radeon AI PRO R9700); ROCm 6.x/7.x. Apache-2.0, experimental ("not part of the
official ROCm distribution"). The repo ships runnable examples (`01-vectorAdd`, `02-tiledCopy`,
`03-tiledMma`, `04-preshuffle_gemm`) and a production `kernels/` library (GEMM, MoE, MLA decode,
paged attention, FlashAttention, layernorm/rmsnorm/softmax). See the AMD blog
[FlyDSL: Expert GPU Kernel Development with the Ease of MLIR Python Native DSL](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html).

> **Version basis (verified 2026-06-03).** This page is verified against `pip install flydsl`
> = **0.1.8** (the latest published release) inside the dev_primus ROCm container, cross-read
> with repo `main` docs. ROCm/FlyDSL git tags top out at **v0.2.0.dev626**; there is **no
> 1.x / 1.18.0** release on PyPI or as a git tag. FlyDSL moves fast and `main` is ahead of the
> wheel — notably the `kernels/` library and `04-preshuffle_gemm` example are **repo-only**
> (`import kernels.*` is unavailable from the pip wheel; clone the repo and add it to
> `PYTHONPATH`). Re-verify symbols against the exact version you build before relying on them.

## Programming Model

The real API is `@flyc.kernel` (defines a GPU `gpu.func`) + `@flyc.jit` (host launcher that
JIT-compiles and caches on first call):

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, buffer_ops

@flyc.kernel
def vec_add_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int]):
    tid = gpu.thread_idx.x
    bid = gpu.block_idx.x
    idx = bid * 256 + tid
    # ... body uses fx.* arith, Vector, layout ops, and AMD buffer intrinsics ...

@flyc.jit
def vec_add(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, N: fx.Constexpr[int],
            stream: fx.Stream = fx.Stream(None)):
    vec_add_kernel(A, B, C, N).launch(grid=(N // 256,), block=(256,), stream=stream)
```

- `fx.Tensor` maps a PyTorch tensor to an MLIR memref via DLPack at the host boundary.
- `fx.Constexpr[int]` is compile-time (embedded in IR; distinct values → distinct cache entries).
- `range_constexpr(n)` is a compile-time-unrolled loop.
- AST rewriting turns Python `for`/`if` into `scf.for`/`scf.if`; compiled binaries cache to
  `~/.flydsl/cache/` (disable with `FLYDSL_RUNTIME_ENABLE_CACHE=0`).
- A `flydsl.autotune` module is reserved for a Triton-style autotuner, but it is **empty in the
  0.1.8 release wheel** (no public symbols) — autotuning is repo-`main`/roadmap, not yet shipped.

## Layout System

FlyDSL's core is the CuTe-style layout algebra exposed both in the `fly` dialect and in
`flydsl.expr`:

- **Shape**, **Stride**, **Layout** (`= (Shape, Stride)`), **Coord**; index = `dot(Coord, Stride)`.
- Construction: `make_shape`, `make_stride`, `make_layout`, `make_coord`.
- Mapping: `crd2idx(coord, layout)`, `idx2crd(idx, layout)`. Inspection: `size`, `cosize`, `rank`.
- Algebra: `composition(A, B)` is exposed at the `flydsl.expr` top level; `product` / `divide`
  (logical/tiled/blocked partition) are `fly`-dialect algebra ops, not top-level `flydsl.expr`
  symbols in the 0.1.8 wheel.
- Tiled copy/MMA atoms: `make_copy_atom`, `make_tiled_copy`, `CopyAtom`, MFMA atoms in
  `03-tiledMma.py`. Layout views are metadata-only; data motion is explicit.

## Compilation Pipeline

`MlirCompiler.compile()` runs three stages (`RocmBackend._pipeline_parts()`):

- **A. Fly → ROCDL**: `fly-rewrite-func-signature`, `fly-canonicalize`, `fly-layout-lowering`,
  `fly-int-swizzle-simplify`, `fly-convert-atom-call-to-ssa-form`,
  `fly-promote-regmem-to-vectorssa`, `convert-fly-to-rocdl` (copy atoms → `rocdl.buffer_load/store`,
  MMA atoms → `rocdl.mfma.*`), then `convert-gpu-to-rocdl{chipset=gfxNNN}`, `fly-rocdl-cluster-attr`.
- **B. → LLVM**: `rocdl-attach-target{chip=gfxNNN}`, scf/cf/gpu/vector/arith/func → LLVM.
- **C. binary**: `gpu-module-to-binary{format=fatbin}`.

## GEMM / MoE Optimization

FlyDSL's production GEMM is the **preshuffle** recipe — B permuted offline to MFMA-native
order, XOR16-swizzled A in LDS, ping-pong staging, K64 micro-step, optional CShuffle epilogue.
This is the same layout convention as CK / AITER (the B-layout builder docstring says it
"matches aiter/CK preshuffle"). Full treatment in
[technique-preshuffle-gemm](../techniques/preshuffle-gemm.md).

FlyDSL has reached AITER's tree at least once: [pr-aiter-3117](../../sources/prs/aiter/PR-3117.md)
merged a FlyDSL MXFP4 fused-MoE stage-2 kernel for DeepSeek-R1/V3 EP4 prefill (async-X prologue,
persistent expansion, wave-priority + scheduler-barrier tuning) — though it was **reverted the
next day** (PR #3344), so nothing FlyDSL currently ships in AITER; treat it as a technique demo.

## Status & Roadmap

- Working today: fp8/int8/int4(W4A8)/fp16/bf16 preshuffle GEMM, FP8 block-scale GEMM
  (`blockscale_preshuffle_gemm.py`), fp4/MXFP4 GEMM (`preshuffle_gemm.py` `in_dtype="fp4"`),
  2-stage MoE, MLA decode, paged-attention decode, norms/softmax; WMMA GEMM on gfx1250.
- WIP perf tuning: PagedAttention, FlashAttention.
- Open research: autotuning of block-tile / MFMA-shape / stage-count.

Treat FlyDSL as a directional bet; CK-Tile remains the default for the highest-performance
production AMD work. FlyDSL reached AITER's tree once (PR-3117) but was reverted, so it does
not currently back any shipping AITER kernel.

## See Also

- [technique-preshuffle-gemm](../techniques/preshuffle-gemm.md) — the GEMM/MoE recipe in detail
- [lang-composable-kernel](composable-kernel.md) — production C++ alternative FlyDSL ports
- [lang-hip](hip.md) — low-level escape hatch
- [blog-amd-flydsl-gemm](../../sources/blogs/amd-flydsl-gemm.md) — repo docs: pipeline + benchmark harness
- [blog-amd-flydsl](../../sources/blogs/amd-flydsl.md) — design overview
