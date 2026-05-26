---
id: lang-flydsl
title: "FlyDSL — AMD Flexible LaYout DSL"
type: language
tags: [flydsl, composable-kernel, mfma, lds, hip]
related: [lang-hip, lang-composable-kernel, hw-mfma, hw-lds, technique-wave-specialization]
sources: [blog-amd-flydsl, doc-amd-cdna3-isa, doc-amd-cdna4-isa]
reproducibility: snippet
architectures: [cdna3, cdna4]
confidence: source-reported
aliases: [FlyDSL, "Flexible LaYout DSL", "AMD FlyDSL"]
---

# FlyDSL — AMD Flexible LaYout DSL

## Overview

[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) is AMD's MLIR-native Python DSL for authoring GPU tile programs. It exposes the `flir` / `fly` dialect on top of MLIR and targets gfx942 (CDNA 3), gfx950 (CDNA 4), gfx1250 (MI450 / future), and gfx1201 (RDNA 4). Roughly the AMD analogue of NVIDIA's CuTe-DSL or Triton: Python frontend, layout-aware tile abstractions, MLIR lowering through `flir` / `fly` / `amdgpu` / `rocdl` / `gpu` dialects, and finally AMDGCN.

The repository ships runnable examples (the README lists `vectorAdd`, `tiledCopy`, `tiledMma`, and `preshuffle_gemm`) so it is more than a paper sketch. See also the AMD blog [FlyDSL: Expert GPU Kernel Development with the Ease of MLIR Python Native DSL on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/flydsl-python-native/README.html).

Status: pre-1.0 research project. Most production AMD kernels are still authored in CK-Tile C++; FlyDSL is the AMD bet on a high-level tile DSL that can match CK-Tile performance with less code.

## Programming Model

```python
import flydsl as fd

@fd.kernel(arch="gfx950", grid=lambda M, N: (M // 256, N // 256))
def gemm_mxfp4_bf16(
    A: fd.TileTensor[fd.mxfp4, "M K", layout=fd.RowMajor],
    B: fd.TileTensor[fd.mxfp4, "K N", layout=fd.ColumnMajor],
    C: fd.TileTensor[fd.bf16,  "M N", layout=fd.RowMajor],
    M: fd.i32, N: fd.i32, K: fd.i32,
):
    block = fd.BlockTile(M=256, N=256, K=128)
    pipe  = fd.Pipeline.wave_specialized(producers=2, consumers=2, stages=3)

    with pipe:
        a = fd.load_to_lds(A.tile(block.m, block.k), swizzle=fd.XorSwizzle())
        b = fd.load_to_lds(B.tile(block.k, block.n), swizzle=fd.XorSwizzle())
        acc = fd.mfma_scale(
            a, b, shape=(32, 32, 64),
            scale_a=A.scale_tile(block.m, block.k),
            scale_b=B.scale_tile(block.k, block.n),
            cbsz=fd.MXFP4, blgp=fd.MXFP4,
        )
    fd.store(C.tile(block.m, block.n), acc.to_dtype(fd.bf16))
```

The compiler:

- Lowers `fd.load_to_lds` to `buffer_load_dwordx4_lds` on CDNA 4 (128-bit) or `buffer_load_dword_lds` on CDNA 3 (32-bit).
- Lowers `fd.mfma_scale` to `v_mfma_scale_f32_32x32x64_f8f6f4` on CDNA 4 and falls back to software-scaled `v_mfma_f32_*_fp8_fp8` on CDNA 3.
- Inserts `s_waitcnt` and `s_barrier` for the producer/consumer pipeline.
- Chooses XOR-swizzle parameters from the tile shape and dtype.

## Layout System

FlyDSL's central abstraction is `TileTensor`, which carries:

- Element dtype (`bf16`, `fp8_e4m3`, `mxfp4`, …).
- Shape and stride (typed via TVM-style `tir.Var`).
- Memory space (`global`, `lds`, `vgpr`, `agpr`).
- Layout descriptor (row-major / column-major / blocked / XOR-swizzled / Morton).

Tile operations (`.tile()`, `.split()`, `.permute()`) produce new TileTensor views without data motion. Data motion is explicit (`load_to_lds`, `load_to_vgpr`, `store`).

## Status & Roadmap

- Working today: BF16/FP16/FP8 GEMM, vector add, basic FMHA forward.
- In progress: MXFP4/MXFP6 GEMM, FMHA backward, MoE.
- Open research: auto-tuning of block-tile / MFMA-shape / stage-count.

Treat FlyDSL as a directional bet, not a production tool. CK-Tile remains the default for high-performance AMD work.

## See Also

- [lang-composable-kernel](composable-kernel.md) — production C++ alternative
- [lang-hip](hip.md) — low-level escape hatch
- [blog-amd-flydsl](../../sources/blogs/amd-flydsl.md) — upstream README and design notes
