---
id: technique-lds-swizzling
title: "LDS XOR Swizzling on AMD CDNA"
type: technique
architectures: [cdna3, cdna4]
tags: [lds-swizzling, lds, mfma, ds-read, ds-write]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-lds, hw-mfma]
related: [hw-lds, technique-direct-to-lds, technique-wave-specialization, technique-mfma-pipelining, lang-composable-kernel]
sources: [doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-buffer-load-lds, blog-rocm-mfma-tutorial, pr-composable-kernel-1384]
aliases: ["LDS swizzle", "XOR swizzle", "LDS bank-conflict avoidance"]
symptoms: ["ds_read stalls", "LDS bank conflicts in profiler", "MFMA throughput dropping when increasing block size"]
---

# LDS XOR Swizzling on AMD CDNA

## Overview

CDNA's LDS is organized as 32 banks of 4 B each. A `ds_read_b128` issued by a wave reads 4 dwords per lane; if two lanes in the same half-wave hit the same bank in the same cycle, the access serializes — a *bank conflict* — and stalls the wave. The fix is to interleave the LDS layout so that an MFMA-friendly read pattern naturally maps to distinct banks. The canonical interleaver is a row-keyed XOR swizzle.

## Bank Model

```
LDS dword index 0 1 2 3 4 ... 31 32 33 ... 63 64 ...
LDS bank        0 1 2 3 4 ... 31  0  1 ... 31  0 ...
```

A 4 KB LDS region has 4096 / 4 = 1024 dwords spread across 32 banks, 32 dwords per bank. Two lanes whose dword indices differ by a multiple of 32 collide.

For an MFMA `ds_read_b128` (4 dwords / lane × 64 lanes = 256 dwords per issue), the issue is split into 4 half-wave cycles of 32 lanes × 4 dwords each (this is an illustrative model of a wave64 DS access issued over 4 SIMD16 cycles; the exact bank-scheduling sequence depends on the access pattern and the LDS arbiter's per-cycle conflict resolution). Within one cycle, the 32 lanes must hit 32 distinct banks per dword position.

## The Conflict Pattern

A naive row-major `[BLOCK_M][BLOCK_K]` BF16 layout where lane `i` reads element `(row, i)`:

```
lane 0  reads (row, 0..7)   -> dwords 0..3   -> banks  0  1  2  3
lane 1  reads (row, 8..15)  -> dwords 4..7   -> banks  4  5  6  7
...
lane 8  reads (row, 64..71) -> dwords 32..35 -> banks  0  1  2  3   <-- collides with lane 0
```

(For this illustration, treat the row stride as ≥512 B so that column offsets 64..71 stay in-row; with a narrower `BLOCK_K=32` the lane-8 offset wraps into the next row's storage region, but the bank-mapping arithmetic — and therefore the conflict — is identical.)

Eight lanes hit each bank → 8× serialization → MFMA loop stalls for ~8 cycles per `ds_read`.

## The Fix: XOR Swizzle by Row Index

Replace `lds_off = row * BLOCK_K + col` with:

```cpp
// `BLOCK_K * sizeof(bf16) == 64 bytes == 16 dwords` for BLOCK_K=32.
// We want consecutive rows to skew their column position by a row-dependent XOR.
constexpr int SWIZZLE_GROUP = 8;          // # rows per swizzle period
constexpr int DWORD_PER_ROW = (BLOCK_K * sizeof(bf16)) / 4;

uint32_t lds_off_dwords =
      row * DWORD_PER_ROW
    + (col_dword ^ (row & (SWIZZLE_GROUP - 1)));   // <-- XOR

uint32_t lds_off_bytes = lds_off_dwords * 4;
```

Row 0 reads columns `[0,1,2,3,4,5,6,7]`; row 1 reads `[1,0,3,2,5,4,7,6]`; row 2 reads `[2,3,0,1,...]`. The MFMA-shaped `ds_read` now maps each lane to a distinct bank.

The XOR width (3 bits here, masking by 7) is tuned to the row × column dword stride so the lane→bank map is a permutation. CK-Tile's `tile_distribution.hpp` derives the XOR mask automatically from the MFMA shape and tile dtype.

## Standard XOR Patterns

| MFMA shape | Dtype | Tile (BLOCK_K) | XOR mask | Period |
|------------|-------|----------------|----------|--------|
| 16×16×16 | BF16/FP16 | 16 | 0x3 (2 bits) | 4 rows |
| 16×16×32 | FP8 | 32 | 0x3 (2 bits) | 4 rows |
| 32×32×8  | BF16/FP16 | 8 | 0x7 (3 bits) | 8 rows |
| 32×32×16 | BF16/FP16 | 16 | 0x7 (3 bits) | 8 rows |
| 32×32×16 | FP8 | 32 | 0x7 (3 bits) | 8 rows |
| 32×32×64 | f8f6f4 (CDNA 4) | 64 | 0xF (4 bits) | 16 rows |

The rule of thumb: XOR mask width = log2(lanes-per-half-wave / dwords-per-row). When the dwords-per-row matches or exceeds 32, no swizzle is needed (each row already covers all banks).

## Producer-Side: Write Address Must Match

`buffer_load_*_lds` writes to a flat LDS byte offset — there is no in-instruction swizzle. The producer must compute the *same* XOR'd address that the consumer's `ds_read` will read from:

```cpp
// Producer (direct-to-LDS)
uint32_t write_off_bytes =
      (row * DWORD_PER_ROW
       + (col_dword ^ (row & 0x7))) * 4;

__builtin_amdgcn_raw_buffer_load_lds(
    rsrc,
    /*lds_ptr=*/ smem_a + write_off_bytes,
    /*size=*/    4,                    // dword variant
    /*voffset=*/ thread_gmem_off,
    /*soffset=*/ 0,
    /*offset=*/  0,
    /*aux=*/     0);

// Consumer (ds_read in the MFMA loop)
auto a_frag = ds_read_b128(smem_a + write_off_bytes);
```

If write and read addresses disagree, the kernel silently produces wrong outputs — the most painful debug class on CDNA.

## Diagnostic

Use `rocprofv3 --pmc SQ_INSTS_LDS,SQ_LDS_BANK_CONFLICT`:

- ratio `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS > 0.1` → swizzle is wrong or missing.
- Healthy CK-Tile MFMA inner loop: ratio ≈ 0.02-0.04 (a few unavoidable conflicts on edge tiles).

## CDNA 3 vs CDNA 4

The bank model is unchanged (32 banks × 4 B). What changed:

- **LDS size**: 64 KB → 160 KB enables 3-4 pipeline stages, so swizzle-period × stage-count can grow without overflowing.
- **`ds_read_b128`**: same width and same conflict rules on both arches.
- **`buffer_load_dwordx4_lds` on CDNA 4**: the wider writer makes XOR-address miscomputation more painful because one bad write trashes 16 B instead of 4 B.

## Caveats

- The XOR swizzle assumes the consumer reads in the *MFMA-native* pattern. A different consumer pattern (e.g., a softmax pass scanning rows) needs its own analysis — pick the layout for whichever consumer is on the critical path.
- Padding `[BLOCK_M][BLOCK_K + PAD]` (the CDNA 2 / NVIDIA SM80 trick) also avoids conflicts but wastes LDS. XOR is strictly better when MFMA dictates the pattern.
- CK-Tile encodes the swizzle in its `tile_distribution_encoding`. Hand-rolling the same XOR pattern outside CK requires care to match CK's address computation if you want to interop.

## See Also

- [hw-lds](../hardware/lds.md) — bank organization
- [technique-direct-to-lds](direct-to-lds.md) — producer side
- [technique-mfma-pipelining](mfma-pipelining.md) — what bank-conflict-free reads unlock
- [lang-composable-kernel](../languages/composable-kernel.md) — tile_distribution_encoding API
