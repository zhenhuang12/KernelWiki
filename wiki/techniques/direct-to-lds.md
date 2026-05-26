---
id: technique-direct-to-lds
title: "Direct-to-LDS Global Loads on AMD CDNA"
type: technique
architectures: [cdna3, cdna4]
tags: [direct-to-lds, buffer-load-lds, lds, mfma, wave-specialization]
confidence: source-reported
reproducibility: snippet
prerequisites: [hw-buffer-load-lds, hw-lds]
related: [hw-buffer-load-lds, hw-lds, technique-wave-specialization, technique-mfma-pipelining, technique-lds-swizzling, lang-amdgcn-asm]
sources: [doc-amd-cdna3-isa, doc-amd-cdna4-isa, blog-rocm-buffer-load-lds, pr-composable-kernel-1384, pr-composable-kernel-2110]
aliases: ["direct-to-LDS", "buffer_load_lds technique", "VGPR-bypass loads"]
symptoms: ["VGPR pressure from staging loads", "low MFMA utilization due to load/MFMA serialization", "ds_write congestion on producer waves"]
---

# Direct-to-LDS Global Loads on AMD CDNA

## Overview

`buffer_load_*_lds` writes HBM data straight into LDS without first landing in a VGPR. This is the AMD equivalent of NVIDIA's TMA / `cp.async.bulk` and is the foundation of every modern CDNA GEMM/FMHA kernel: it frees the producer wave's VGPRs for control flow and removes the `global → VGPR → LDS` round-trip that costs both bandwidth and registers.

## Why It Matters

The conventional CDNA 2 (MI250) pattern was:

```
buffer_load_dwordx4 v[8:11], v_off, s_rsrc, 0   ; HBM -> VGPR
ds_write_b128 v_lds_off, v[8:11]                ; VGPR -> LDS
```

This consumes 4 VGPRs per active load *per lane*, which directly competes with MFMA operand / accumulator registers. For a 256-thread workgroup staging a 128×128 BF16 A-tile per K-step, the VGPR cost can erase the occupancy headroom needed to hide HBM latency.

With direct-to-LDS:

```
; CDNA 4 (gfx950) only — 128-bit direct-to-LDS:
buffer_load_dwordx4 v_lds_off, v_g_off, s_rsrc, 0 lds   ; HBM -> LDS, no VGPR

; CDNA 3 (gfx942) fallback — widest direct-to-LDS is 32-bit:
buffer_load_dword   v_lds_off, v_g_off, s_rsrc, 0 lds   ; HBM -> LDS, no VGPR
```

The producer wave needs only its address-computation registers; the operand registers are free for the consumer's MFMA inner loop. Note that the `dwordx4` LDS variant is **CDNA 4 only** — CDNA 3 tops out at `buffer_load_dword` for direct-to-LDS and must issue 4× as many loads to move the same bytes.

## Width per Architecture

| Arch | Widest direct-to-LDS | Lanes per issue [^lanes] | Bytes per wave per issue |
|------|----------------------|---------------------------|---------------------------|
| CDNA 3 (gfx942) | `buffer_load_dword_lds` | 64 | 256 |
| CDNA 4 (gfx950) | `buffer_load_dwordx4_lds` | 64 | 1024 |

[^lanes]: "Lanes per issue" reports the architectural wave64 lane count; the actual number of lanes that participate in any given issue is governed by the wave's EXEC mask, so a partially masked wave will move proportionally fewer bytes per issue.

CDNA 4's 128-bit variant is a 4× per-issue bandwidth increase and is *the* reason CDNA 4 wave-specialized kernels can reach NVIDIA SM90-class arithmetic intensity.

## Skeleton (HIP + intrinsic)

```cpp
// Cooperative producer-wave load of a BLOCK_M x BLOCK_K bf16 tile into LDS
template <int BLOCK_M, int BLOCK_K>
__device__ void direct_to_lds_load_tile(
    bf16*        smem_dst,       // BLOCK_M * BLOCK_K bf16 in LDS
    const bf16*  gmem_src,       // HBM source (already offset to tile)
    int          gmem_stride,
    int          wave_id,
    int          lane
) {
    // Each lane handles one 8-element (16 B) chunk
    constexpr int ELEMS_PER_LANE = 8;
    constexpr int TILE_ELEMS     = BLOCK_M * BLOCK_K;
    constexpr int LANES_NEEDED   = TILE_ELEMS / ELEMS_PER_LANE;

    static_assert(LANES_NEEDED % 64 == 0, "tile must be a whole number of waves");

    int linear_id = wave_id * 64 + lane;
    if (linear_id >= LANES_NEEDED) return;

    int row = (linear_id * ELEMS_PER_LANE) / BLOCK_K;
    int col = (linear_id * ELEMS_PER_LANE) % BLOCK_K;

    // Global address (in bf16 elements)
    const bf16* g = gmem_src + row * gmem_stride + col;

    // LDS offset (in bytes) — XOR-swizzled in real code; see technique-lds-swizzling
    uint32_t lds_off = (row * BLOCK_K + col) * sizeof(bf16);

    // Direct-to-LDS: no VGPR destination, target is the LDS pointer
    __builtin_amdgcn_raw_buffer_load_lds(
        /*rsrc    =*/ make_buffer_rsrc(g),
        /*lds_ptr =*/ (__attribute__((address_space(3))) void*)(&__lds_base[lds_off]),
        /*size    =*/ 16,          // bytes per lane (1, 2, 4, 12, or 16); 16 on CDNA 4, 4 on CDNA 3
        /*voffset =*/ 0,
        /*soffset =*/ 0,
        /*offset  =*/ 0,
        /*aux     =*/ 0
    );
}
```

The canonical LLVM intrinsic is `__builtin_amdgcn_raw_buffer_load_lds(rsrc, lds_ptr, size, voffset, soffset, offset, aux)`, documented in the [AMDGPU LLVM Usage Guide](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html) under the `llvm.amdgcn.raw.buffer.load.lds` entry. The intrinsic name and signature have shifted across clang 17 / 18 / 19 / 20 (e.g., earlier releases exposed `__builtin_amdgcn_buffer_load_lds` with a different argument order, and structured-buffer variants were renamed) — always consult the headers shipped with the target compiler before relying on a specific spelling. Production code (CK-Tile, AITER) often falls back to AMDGCN inline asm — see [lang-amdgcn-asm](../languages/amdgcn-asm.md) for the exact `idxen lds offset:%N` encoding.

## Pairing with Wave Specialization

Direct-to-LDS is only useful if a *different* wave runs MFMA while the producer wave is issuing loads. Otherwise the producer wave just sits on `s_waitcnt vmcnt(0)`. The canonical pairing:

1. Producer waves issue `buffer_load_dwordx4_lds` for the next K-stage.
2. `s_waitcnt vmcnt(0)` (drain HBM).
3. `__syncthreads()` handoff.
4. Consumer waves `ds_read_b128` from LDS into VGPR and run MFMA.
5. Second `__syncthreads()` releases the stage for the next producer round.

See [technique-wave-specialization](wave-specialization.md) for the full 4-wave skeleton.

## Pairing with LDS Swizzling

The destination LDS offset is a regular byte offset — there is no `swizzle:` modifier on `buffer_load_*_lds`. To avoid LDS bank conflicts on the subsequent `ds_read`, the writer must apply an XOR swizzle in the destination address itself. See [technique-lds-swizzling](lds-swizzling.md).

## Caveats

- **No mask field**: out-of-bounds bytes are silently dropped at the buffer descriptor level; bounds-safety must come from the buffer resource fields (`stride`, `num_records`).
- **Address alignment**: `buffer_load_dwordx4_lds` requires 16 B alignment on both global address and LDS offset. Misalignment silently produces wrong data (no fault on CDNA 3, partial-fault on CDNA 4).
- **HIP intrinsic flux**: the `__builtin_amdgcn_raw_buffer_load_lds` signature (and the existence of related variants such as `struct_buffer_load_lds` / the older `buffer_load_lds`) is unstable across LLVM 17 / 18 / 19 / 20. CK-Tile pins clang to a known-good ROCm release; if you hand-roll, prefer the inline-asm form or guard on `__clang_major__`.
- **Counter accounting**: direct-to-LDS is a vector-memory operation and is tracked by `vmcnt`, not `lgkmcnt`. The producer wave must use `s_waitcnt vmcnt(0)` (not `s_waitcnt lgkmcnt(0)`) before signalling the consumer. `lgkmcnt` only covers the LDS-side completion of unrelated `ds_*` traffic, not the HBM-to-LDS write performed by `buffer_load_*_lds`.

## See Also

- [hw-buffer-load-lds](../hardware/buffer-load-lds.md) — instruction-level reference
- [technique-wave-specialization](wave-specialization.md) — the consumer-side companion
- [technique-lds-swizzling](lds-swizzling.md) — bank-conflict avoidance for the receiver
- [technique-mfma-pipelining](mfma-pipelining.md) — what the freed VGPRs buy you
