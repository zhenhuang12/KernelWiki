---
id: technique-preshuffle-gemm
title: "Preshuffle GEMM Pipeline on AMD CDNA (FlyDSL / AITER / CK)"
type: technique
architectures: [cdna3, cdna4]
tags: [preshuffle, gemm, flydsl, mfma, lds, lds-swizzling, ping-pong-scheduling, epilogue-fusion, moe]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-mfma, hw-lds, lang-flydsl]
related: [lang-flydsl, technique-lds-swizzling, technique-direct-to-lds, technique-mfma-pipelining, technique-epilogue-fusion, technique-ping-pong-scheduling, kernel-grouped-gemm, kernel-fp8-block-scale-gemm]
sources: [blog-amd-flydsl-gemm, blog-amd-flydsl, pr-aiter-3117, doc-amd-aiter-readme, doc-amd-cdna4-isa]
aliases: ["B-preshuffle", "preshuffle layout", "shuffled-B GEMM", "weight preshuffle"]
symptoms: ["ds_read stalls in MFMA loop", "B-tile global load not coalesced", "MFMA throughput below peak on narrow-M GEMM", "epilogue store bank conflicts"]
---

# Preshuffle GEMM Pipeline on AMD CDNA (FlyDSL / AITER / CK)

## Overview

*Preshuffle GEMM* is a standard high-performance recipe (used by CK / AITER / FlyDSL) for
low-precision matrix multiply on CDNA 3/4 (`C[M,N] = A[M,K] @ B[N,K]^T`). The **weight matrix B is permuted
offline** into the exact element order the MFMA instruction consumes, so the in-kernel
B load is a flat coalesced copy with **no in-kernel transpose and no per-lane gather**.
The remaining ingredients — XOR-swizzled LDS staging of A, a ping-pong (double-buffered)
LDS pipeline, a K64-byte MFMA micro-step, and an optional CShuffle epilogue — together
keep the matrix cores fed.

This page documents the recipe as it appears in three interlocking AMD codebases that
share the *same* layout convention:

- **[ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)** — `kernels/preshuffle_gemm.py`,
  `compile_preshuffle_gemm_a8(...)` (the reference implementation in the repo).
- **ROCm/composable_kernel (CK / CK-Tile)** — the original C++ implementation FlyDSL ports.
- **ROCm/aiter** — the production target FlyDSL's B layout is documented to match (the builder
  docstring says it "matches aiter/CK preshuffle"). AITER once merged a FlyDSL MoE kernel
  ([pr-aiter-3117](../../sources/prs/aiter/PR-3117.md)), but it was reverted (PR #3344) and does
  not currently ship — see the MoE section below.

## Why Preshuffle At All

A wave64 MFMA reads B in a lane/byte order dictated by the instruction (e.g.
`v_mfma_f32_16x16x32_fp8_fp8`). If B sits in plain `[N][K]` row-major, feeding the MFMA
requires either a transposing LDS round-trip or a strided per-lane global gather — both
cost bandwidth and bank conflicts. By **permuting B once, offline**, into MFMA-native
order, the hot loop's B path becomes a contiguous `buffer_load_dwordx4` straight to
registers (B usually skips LDS entirely; only A is staged through LDS). The cost is a
one-time host-side reshape of the weights — free for inference where weights are static.

## The B-Preshuffle Layout

FlyDSL's `make_preshuffle_b_layout()` (in `kernels/mfma_preshuffle_pipeline.py`) builds a
5-D blocked layout:

```
# N-major (k_major=False, default):
(N0, K0, KLane, NLane, KPackElems) = (N/16, K/64, 4, 16, kpack_elems)
# K-major (k_major=True):
(N0, K0, KLane, NLane, KPackElems) = (N/16, K/32, 2, 16, kpack_elems)
```

- `NLane = 16` (and `KLane = 4` N-major / `2` K-major) map onto the 16×16 MFMA lane geometry.
- The trailing dim is `kpack_elems` — the per-lane contiguous run an MFMA micro-step consumes
  (`kpack_bytes` is 8 or 16; for 1-byte dtypes `kpack_elems == kpack_bytes`, for fp16/bf16
  it is `kpack_bytes/2`).
- Two block-level orderings are selected by `k_major`:
  - **N-major** (`k_major=False`, default): permutation `(0,1,3,4,2,5)`.
  - **K-major** (`k_major=True`): permutation `(0,3,1,4,2,5)`.

`make_preshuffle_b_layout()`'s docstring states this layout "matches aiter/CK preshuffle for
A8 MFMA kernels" — so FlyDSL-, CK-, and AITER-preshuffled weights are intended to be
interchangeable. (This page quotes the docstring; it has not independently byte-verified the
CK source ordering.)

## Builder API (FlyDSL)

> **Packaging note (verified vs `pip install flydsl`=0.1.8, 2026-06-03).** The `kernels/`
> library shown below is **not shipped in the pip wheel** — `import kernels.preshuffle_gemm`
> fails after a plain `pip install flydsl`. Clone [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL)
> and add the repo root to `PYTHONPATH`. The DSL primitives the recipe rests on
> (`flydsl.expr` layout algebra, `buffer_ops`, `gpu`, copy/MMA atoms) *are* in the wheel.
> No `1.x`/`1.18.0` release exists; tags top out at `v0.2.0.dev626`.

```python
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

launch_fn = compile_preshuffle_gemm_a8(   # signature abridged — see source for
    M=16, N=5120, K=8192,                 # out_dtype/epilogue/xcd_swizzle/preload knobs
    tile_m=16, tile_n=128, tile_k=256,
    in_dtype="fp8",          # fp8 | int8 | int4(=W4A8) | fp16 | bf16 | fp4
    lds_stage=2,             # 2 = ping-pong LDS (tuned); 1 = single buffer
    use_cshuffle_epilog=False,
    waves_per_eu=None,       # occupancy cap (1-4) when register pressure hurts
    use_async_copy=False,    # async DMA (buffer_load_lds) for A global→LDS
)
# launch_fn(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, arg_bias, M_val, N_val, stream)
```

Constraint: `tile_k * elem_bytes` must be divisible by 64 (the K64-byte micro-step). The real
launcher takes an `arg_bias` tensor (6th positional) and the builder exposes additional knobs
(`out_dtype`, `epilogue`, `xcd_swizzle`, prologue-preload flags) elided here.

## The Five Pipeline Ingredients

### 1. K64-byte micro-step (2× K32 MFMA per step)

Each K iteration issues **two** K32 MFMA ops back to back, consuming a 64-byte K window.
This amortizes the `s_waitcnt`/issue overhead and matches the 64-byte natural granularity
of `buffer_load_dwordx4`. `load_b_pack_k32()` returns an `i64` (8 B) feeding one K32 op.

### 2. XOR16 LDS swizzle for A

A is staged through LDS; to keep `ds_read` bank-conflict-free, the K dimension is XOR-swizzled
at 16-byte granularity. The producer (`lds_store_16b_xor16`) and consumer (`lds_load_pack_k32`)
**must compute the same swizzle**:

```python
def swizzle_xor16(row, col, k_blocks16):
    # col XOR ((row & (k_blocks16 - 1)) * 16)
    # k_blocks16 = tile_k_bytes / 16 is always a power of 2, so use bitwise AND
    # instead of remui -> saves ~10 VALU cycles per address on CDNA.
    mask = k_blocks16 - 1
    return col ^ ((row & mask) * 16)
```

See [technique-lds-swizzling](lds-swizzling.md) for the bank model behind this.

### 3. Ping-pong LDS (`lds_stage=2`)

Two LDS buffers for the A tile. While the MFMA loop reads buffer *ping*, the next A tile
streams into buffer *pong*; a **cross-tile A0 prefetch** issues the first global load of the
*next* N/M tile before the current tile's epilogue, overlapping VMEM latency with LDS reads.
`lds_stage=1` falls back to CK's single-buffer intrawave schedule (lower LDS, less overlap).
Related: [technique-direct-to-lds](direct-to-lds.md), [technique-mfma-pipelining](mfma-pipelining.md),
[technique-ping-pong-scheduling](ping-pong-scheduling.md).

### 4. CShuffle epilogue (optional)

With `use_cshuffle_epilog=True`, the accumulator is written to LDS in row-major, a barrier,
then threads are **remapped to a `(MLane, NLane)=(8, 32)` grid and re-read** so output stores
become coalesced `half2` (`e_vec=2`) writes — a port of CK's CShuffle (an LDS round-trip +
thread remap; no `ds_bpermute` in the FlyDSL `mfma_epilogues.py` source, though AMD's
`prebuilt_kernels_guide.md` §3.1 describes it as `ds_bpermute`). Without it, `default_epilog`
uses the raw MFMA row iterator
`row = bx_m + mi*16 + lane_div_16*4 + ii`. See [technique-epilogue-fusion](epilogue-fusion.md).

### 5. Dtype handling

- **W4A8 (`in_dtype="int4"`)**: A is int8; B is packed int4 (2 values/byte), unpacked to
  int8 in-kernel before MFMA.
- **fp4 / MXFP4**: handled inside `preshuffle_gemm.py` itself (`in_dtype="fp4"`), which emits
  CDNA 4's scaled MFMA `mfma_scale_f32_16x16x128_f8f6f4`; gfx950 only. (The separate
  `blockscale_preshuffle_gemm.py` is the **FP8** per-block-scale variant, not MXFP4.)

## Tuned Configurations (FlyDSL `scripts/run_benchmark.sh`)

`GEMM_SHAPES` (`dtype,M,N,K,tile_m,tile_n,tile_k`) — 10 rows:

| dtype | M | N | K | tile_m | tile_n | tile_k |
|-------|---|---|---|--------|--------|--------|
| fp8  | 16   | 40960 | 5120 | 16 | 128 | 256 |
| fp8  | 16   | 77824 | 5120 | 16 | 128 | 256 |
| fp8  | 256  | 2112  | 7168 | 64 | 64  | 256 |
| fp8  | 512  | 2112  | 7168 | 64 | 64  | 256 |
| fp8  | 5120 | 5120  | 8320 | 64 | 256 | 128 |
| fp8  | 9728 | 8192  | 8320 | 64 | 256 | 128 |
| fp8  | 8192 | 8192  | 8192 | 128| 256 | 128 |
| int8 | 9728 | 8192  | 8320 | 64 | 256 | 128 |
| int4 | 9728 | 8192  | 8320 | 64 | 256 | 128 |
| bf16 | 5120 | 5120  | 8320 | 64 | 256 | 128 |

FP4 lives in a **separate** `GEMM_FP4_SHAPES` list (gfx950 only), all `M=N=K=8192` sweeping
the tile: `(64,128,256)`, `(64,256,256)`, `(128,256,256)`, `(128,256,128)`. There are also
`*_ASYNC` variants — same fields plus an optional trailing `waves_per_eu` (the async path is
enabled by the harness passing `--use_async_copy`, not by a shape-string field) — and
`HGEMM_SHAPES_GFX950/_CDNA3` for fp16/bf16.

Pattern *inferred* from the table (not stated in FlyDSL docs): **narrow-M (decode-like,
M=16)** favors `tile_m=16, tile_n=128, tile_k=256`; **large-M (prefill)** favors larger
`tile_m` (64–128) with `tile_n=256`. (Absolute TFLOPS are not published in the repo — the
harness reports `TB/s` and `TFLOPS` per-shape and must be run on-device.)

## Extending To MoE: AITER PR-3117 (FlyDSL MoE example — merged then reverted)

[pr-aiter-3117](../../sources/prs/aiter/PR-3117.md) — *"perf(flydsl): MXFP4 fused-MoE
stage2 optimization for EP prefill"* — was a FlyDSL production variant merged into AITER on
2026-05-25 and **reverted the next day by [PR #3344](https://github.com/ROCm/aiter/pull/3344)**,
so it does not currently ship. It remains a useful worked example of the techniques. It
adds `_t64x128x256_atomic_persist_async_w4_cumul3`, an fp4×fp4 (MXFP4) stage-2 fused-MoE
variant for DeepSeek-R1/V3 EP4 prefill on MI355X (CDNA 4), layering on top of the preshuffle
base. Measured result (PR body, MI355X, 3 reps): `fused_moe` **3711.8 → 3384.9 µs, −8.81%**
end-to-end, 12/12 correctness. The stacked changes:

- **Async X DMA prologue** — overlaps prologue VMEM with the K-loop (the base
  `_atomic_persist` kernel did not).
- **Drop redundant `out.fill_(0)`** — a *separate* host-wrapper change: the standard
  `fused_moe` path already zeros `moe_buf` (`moe_buf_set_zero_kernel_2d`), so the defensive
  zero-fill is removed, saving an estimated **~130 µs per call** of redundant HBM writes
  (PR estimate, not independently measured). This is orthogonal to the async prologue.
- **`cu_num_mul=3` persistent expansion** — persistent-kernel CU oversubscription (3× CU count)
  ([technique-mfma-pipelining](mfma-pipelining.md), persistent scheduling).
- **Asymmetric `b_lo` / `b_hi` 3:1 split** — `b_lo` loads 3/4 of K, `b_hi` 1/4.
- **LDS-staged `sorted_weights`** — topk routing weights cached in LDS (gated on `doweight_stage2`).
- **`disable_xdl_arb_stall` + `s_setprio(1)`** — scheduler-barrier + wave-priority bias to
  keep MFMA arbitration from stalling.
- **"Scales-before-B" K-loop ordering** — issue the block scales ahead of the B pack in the
  K loop so the scaled MFMA never waits on the scale operand.

This is a worked example of preshuffle GEMM composed with persistent scheduling,
xcd-aware tile expansion, scheduler barriers, and wave priority — see
[technique-mfma-pipelining](mfma-pipelining.md) and [technique-lds-swizzling](lds-swizzling.md).

## Diagnostics

- `rocprofv3 --pmc SQ_LDS_BANK_CONFLICT,SQ_INSTS_LDS` — ratio > 0.1 means the XOR16 swizzle
  is wrong or the producer/consumer addresses disagree.
- `MFMA busy %` below peak with B-load `vmcnt` waits dominating → B not in preshuffle order,
  or `tile_k` too small to hide global latency (raise to keep the K64 micro-step pipelined).
- Narrow-M GEMM under-occupied → try `tile_m=16` with a wider `tile_n` (inferred from the
  tuned-config table above, not from FlyDSL docs — confirm by sweeping on-device).

## CDNA 3 vs CDNA 4

- **LDS budget**: 64 KB (gfx942) → 160 KB (gfx950) — enables `lds_stage=2` ping-pong at the
  larger tiles without spilling.
- **Scaled MFMA**: the fp4/MXFP4 scaled path needs `v_mfma_scale_f32_*_f8f6f4`
  (gfx950 only); on gfx942 the same builder uses software-scaled fp8 MFMA.
- **Wide direct-to-LDS**: `use_async_copy` lowers to `buffer_load_dwordx4_lds` (128-bit) on
  CDNA 4 vs the 32-bit variant on CDNA 3.

## Caveats

- Preshuffle assumes **static weights** (inference). For training, the per-step reshape cost
  usually erases the win.
- FlyDSL is an experimental research project ("not part of the official ROCm distribution");
  treat absolute perf as on-device-measured, and pin the commit you build against.
- The B layout, XOR16 mask, and epilogue thread remap must agree exactly between host
  preshuffle and kernel — a mismatch produces silently wrong results, the worst CDNA debug class.

## See Also

- [lang-flydsl](../languages/flydsl.md) — the DSL these kernels are written in
- [technique-lds-swizzling](lds-swizzling.md) — the XOR16 bank-conflict model
- [technique-direct-to-lds](direct-to-lds.md) — async A global→LDS staging
- [kernel-grouped-gemm](../kernels/grouped-gemm.md) — MoE grouped-GEMM case study
- [pr-aiter-3117](../../sources/prs/aiter/PR-3117.md) — FlyDSL MXFP4 MoE in AITER
