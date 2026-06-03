---
id: blog-amd-flydsl-gemm
title: "FlyDSL Preshuffle GEMM & MoE — Pipeline, LDS Swizzle, and Benchmark Harness"
author: AMD ROCm (FlyDSL repo docs)
url: https://github.com/ROCm/FlyDSL/blob/main/docs/prebuilt_kernels_guide.md
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- flydsl
- preshuffle
- gemm
- moe
- lds-swizzling
- mfma
- ping-pong-scheduling
- epilogue-fusion
- jit-compilation
retrieved_at: 2026-06-03
---

## Summary

This captures the concrete GEMM / MoE optimization machinery documented in the
[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) repository (README + `docs/`), as opposed
to the high-level design pitch in [blog-amd-flydsl](amd-flydsl.md). FlyDSL is a Python
front-end (`@flyc.kernel` / `@flyc.jit`) over the `fly` MLIR layout dialect; kernels trace
to MLIR and compile through a three-stage pass pipeline (`Fly → ROCDL → LLVM → fatbin`).
Verified targets: gfx942 (MI300X/MI308X), gfx950 (MI350/MI355X), gfx1250 (MI450),
gfx1201 (Radeon AI PRO R9700); ROCm 6.x/7.x.

The production GEMM is `kernels/preshuffle_gemm.py`, exposed via
`compile_preshuffle_gemm_a8(...)`. Source: `docs/prebuilt_kernels_guide.md`,
`docs/architecture_guide.md`, `kernels/mfma_preshuffle_pipeline.py`,
`scripts/run_benchmark.sh`.

> **Packaging caveat (verified vs pip `flydsl`=0.1.8, 2026-06-03):** the `kernels/` library is
> repo-only — it is **not** in the published wheel. Clone ROCm/FlyDSL and put the repo root on
> `PYTHONPATH`. Latest release is 0.1.8 (tags max `v0.2.0.dev626`); no `1.18.0` exists.

## Preshuffle GEMM Builder

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

launch_fn = compile_preshuffle_gemm_a8(
    M=16, N=5120, K=8192,
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",          # fp8 | int8 | int4 | fp16 | bf16 | fp4
    lds_stage=2,             # 2 = ping-pong LDS (tuned), 1 = single buffer
    use_cshuffle_epilog=False,
    # waves_per_eu=None,     # occupancy hint (1-4 caps occupancy)
    # use_async_copy=False,  # async DMA for A tile global→LDS
)
```

Computes `C[M,N] = A[M,K] @ B[N,K]^T` with B pre-permuted to MFMA-native layout.

## Documented Optimization Knobs (`docs/prebuilt_kernels_guide.md` §3.1)

- **B-preshuffle layout**: N-major (default) shape `(N0,K0,KLane,NLane,KPackElems) =
  (N/16, K/64, 4, 16, kpack_elems)`; K-major uses `(N/16, K/32, 2, 16, kpack_elems)`
  (`kpack_elems = kpack_bytes` for 1-byte dtypes, `kpack_bytes/2` for fp16/bf16).
  Built by `make_preshuffle_b_layout(...)`, whose docstring states it "matches aiter/CK
  preshuffle for A8 MFMA kernels": N-major order = `(0,1,3,4,2,5)`, K-major = `(0,3,1,4,2,5)`.
- **lds_stage=2 (ping-pong)**: two LDS buffers for A tiles; cross-tile A0 prefetch overlaps
  VMEM with LDS reads. `lds_stage=1` is CK-style single-buffer intrawave schedule.
- **K64-byte micro-step**: each step issues 2× K32 MFMA; constraint `tile_k * elem_bytes % 64 == 0`.
- **XOR16 swizzle**: byte-level LDS swizzle to avoid bank conflicts. `swizzle_xor16(row, col, k_blocks16)`
  computes `col XOR ((row & (k_blocks16-1)) * 16)`; `k_blocks16` is power-of-2 (`tile_k_bytes/16`)
  so the AND replaces `remui`, "saving ~10 VALU cycles on CDNA."
- **CShuffle epilogue** (`use_cshuffle_epilog=True`, `mfma_epilogues.py`): write C tile to LDS
  row-major, barrier, remap threads to a `(MLane,NLane)=(8,32)` grid and re-read for half2
  (`e_vec=2`) packed stores — a CK CShuffle port via an LDS round-trip + thread remap (the
  FlyDSL implementation uses **no `ds_bpermute`**; AMD's `prebuilt_kernels_guide.md` §3.1 says
  "ds_bpermute" but the `mfma_epilogues.py` source contains none).
- **W4A8 int4**: A is int8, B is packed int4 (2 values/byte), unpacked to int8 in-kernel.
- **Autotune**: a `flydsl.autotune` module is reserved for a "Triton-style autotune module,"
  but it is empty (no public symbols) in the 0.1.8 release wheel — roadmap, not yet shipped.
- **JIT disk cache**: `~/.flydsl/cache/`, auto-invalidates on source/closure change;
  disable with `FLYDSL_RUNTIME_ENABLE_CACHE=0`.

## Other Pre-built Kernels

`blockscale_preshuffle_gemm.py` (MXFP4 block-scale), `moe_gemm_2stage.py` /
`moe_blockscale_2stage.py` / `mixed_moe_gemm_2stage.py` (2-stage gate-up + reduce MoE),
`mla_fwd_decode*.py`, `pa_decode_fp8.py`, `flash_attn_func.py`, `hgemm_splitk.py`,
plus `layernorm/rmsnorm/softmax` (LDS-cached, XOR-shuffle wave reductions). gfx1250 uses
WMMA variants (`wmma_gemm_gfx1250.py`, `gemm_fp8fp4_gfx1250.py`).

## Benchmark Harness (`scripts/run_benchmark.sh`)

`GEMM_SHAPES` (`dtype,M,N,K,tile_m,tile_n,tile_k`) — 10 rows:

```
fp8 ,16   ,40960,5120,16 ,128,256
fp8 ,16   ,77824,5120,16 ,128,256
fp8 ,256  ,2112 ,7168,64 ,64 ,256
fp8 ,512  ,2112 ,7168,64 ,64 ,256
fp8 ,5120 ,5120 ,8320,64 ,256,128
fp8 ,9728 ,8192 ,8320,64 ,256,128
fp8 ,8192 ,8192 ,8192,128,256,128
int8,9728 ,8192 ,8320,64 ,256,128
int4,9728 ,8192 ,8320,64 ,256,128
bf16,5120 ,5120 ,8320,64 ,256,128
```

FP4 is a separate `GEMM_FP4_SHAPES` list (gfx950 only), all `8192,8192,8192` over tiles
`(64,128,256)/(64,256,256)/(128,256,256)/(128,256,128)`. There are also `*_ASYNC` variants
(trailing `lds_stage=2`) and `HGEMM_SHAPES_GFX950/_CDNA3` for fp16/bf16.

Output is tabular `TB/s` + `TFLOPS` per shape (absolute numbers are not published in the
repo; the harness must be run on-device). The perf harness itself —
`perftest()` / `checkAllclose()` in `tests/test_common.py` — is, per the file header,
"adapted from AIter."

## Why It Matters

This is the AMD answer to "what does a Triton/CuTe-DSL-class tile DSL emit for a tuned MFMA
GEMM?" The preshuffle layout, XOR16 swizzle, ping-pong LDS, K64 micro-step, and CShuffle
epilogue are the same ingredients hand-tuned CK-Tile and AITER kernels use — FlyDSL just
expresses them through layout algebra. Backs [technique-preshuffle-gemm](../../wiki/techniques/preshuffle-gemm.md)
and [lang-flydsl](../../wiki/languages/flydsl.md). The AITER lineage is concrete: the
preshuffle B layout matches aiter/CK, and AITER ships a FlyDSL MoE kernel ([pr-aiter-3117](../prs/aiter/PR-3117.md)).
