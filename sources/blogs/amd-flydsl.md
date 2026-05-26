---
id: blog-amd-flydsl
title: "FlyDSL — AMD's Flexible Layout DSL for GPU Tile Programs"
author: AMD Research
url: https://github.com/AMD-AIG-AIMA/flydsl
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

[AMD-AIG-AIMA/flydsl](https://github.com/AMD-AIG-AIMA/flydsl) is AMD's research Python+MLIR DSL for authoring GPU tile programs targeting CDNA 3/4. Roughly the AMD analogue of NVIDIA's CuTe-DSL or Triton — Python front-end, layout-aware tile abstractions, lowers through MLIR (`amdgpu`, `rocdl`, `gpu` dialects) to AMDGCN.

## Programming Model

```python
import flydsl as fd

@fd.kernel
def matmul_bf16(A: fd.TileTensor[fd.bf16, "M K"],
                 B: fd.TileTensor[fd.bf16, "K N"],
                 C: fd.TileTensor[fd.f32,  "M N"]):
    block = fd.BlockTile(M=128, N=128, K=64)
    a = fd.load_to_lds(A.tile(block.m, block.k))
    b = fd.load_to_lds(B.tile(block.k, block.n))
    acc = fd.mfma(a, b, shape=(16, 16, 32))
    fd.store(C.tile(block.m, block.n), acc)
```

The compiler chooses LDS swizzle, wave specialization, and MFMA-shape based on the declared layouts. Lowering inspects `block.m`/`block.n` against the target ISA's MFMA catalogue (cf. [doc-amd-cdna3-isa](../docs/amd-cdna3-isa.md), [doc-amd-cdna4-isa](../docs/amd-cdna4-isa.md)) and emits the appropriate intrinsic chain.

## Status

Pre-1.0 research project. Most production AMD kernels are still authored in CK-Tile C++; FlyDSL is the AMD bet on a high-level tile DSL that can match CK-Tile performance with less code. Read for design ideas; not yet production-ready.

## Why It Matters

FlyDSL is the AMD answer to "what plays the role of Triton / CuTe-DSL on Instinct?" — the highest-level layer that still emits hand-tuned-quality MFMA + LDS pipelines. Cited by [lang-flydsl](../../wiki/languages/flydsl.md).
