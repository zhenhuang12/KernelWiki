---
name: KernelWiki
description: Use when the user asks about optimizing NVIDIA Blackwell (SM100, B200) / Hopper (SM90, H100) or AMD Instinct MI300X (CDNA 3, gfx942) / MI355X (CDNA 4, gfx950) GPU kernels — tcgen05/TMEM/CLC/NVFP4/2-SM cooperative, warp specialization, MFMA/AGPR/LDS/buffer_load_lds/XCD/MXFP4, FlashAttention-4, DeepGEMM, FlashMLA, fused MoE, grouped GEMM, CK-Tile / CuTe-DSL / PTX / Triton / HIP, or wants concrete PR references from CUTLASS / SGLang / vLLM / FlashInfer / PyTorch / composable_kernel / aiter / mori / rccl (note: RCCL active development at ROCm/rocm-systems/projects/rccl; ROCm/rccl receives backport cherry-picks for stable branches). Do NOT use for generic CUDA/HIP Q&A unrelated to tensor-core/matrix-core programming, host-side framework integration, or distributed systems (DeepEP / EPLB / DualPipe / ROCSHMEM transport).
argument-hint: "[natural-language-question] | [--tag foo --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# KernelWiki — Blackwell / Hopper / CDNA 3 / CDNA 4 Kernel Optimization Wiki

> **Knowledge cutoff: 2026-04-27.** All upstream PR data, blog summaries, and version-claim entries reflect upstream state on or before this date (per `data/refresh-cutoff.yaml`). Re-run the refresh tooling to advance the cutoff.

Query a structured, cross-referenced knowledge base of GPU kernel optimization for NVIDIA Blackwell (SM100) / Hopper (SM90) and AMD Instinct MI300X (CDNA 3, gfx942) / MI355X (CDNA 4, gfx950).

## When To Use This Skill

Trigger this skill when the user asks about:

- **NVIDIA Blackwell/SM100 kernel programming** — tcgen05.mma, TMEM, CLC, 2-SM cooperative, NVFP4, FP8/FP4 block scaling, PDL/GDC
- **AMD CDNA 3/4 kernel programming** — MFMA / `v_mfma_scale_f32_*_f8f6f4`, AGPR pressure, LDS XOR swizzle, `buffer_load_*_lds`, XCD-aware scheduling, Infinity Cache, MXFP4/6/8 (OCP MX), wave specialization, MFMA pipelining
- **Kernel implementations** — FlashAttention-4, DeepGEMM, FlashMLA, NSA, GatedDeltaNet, NVFP4 GEMM/GEMV, fused MoE, gated dual GEMM, CK-Tile FP8/MXFP4 GEMM, AITER fused MoE / MLA decode
- **Performance patterns** — low SM utilization, memory-bound, register pressure, compute-bound, tail effects, pipeline stalls, LDS bank conflicts, low MFMA utilization
- **DSLs / languages** — CuTe DSL, CUDA C++ with PTX inline, Triton on Blackwell, HIP, Composable Kernel (CK / CK-Tile), AMDGCN inline assembly, FlyDSL
- **Cross-architecture migration** — Hopper wgmma → Blackwell tcgen05, register → TMEM accumulators, CUDA → HIP / CDNA
- **PR references** — "how did vLLM / SGLang / FlashInfer / CUTLASS / PyTorch / composable_kernel / aiter / mori / rccl implement X?" (RCCL active development at ROCm/rocm-systems/projects/rccl; ROCm/rccl receives backport cherry-picks for stable branches.)
- **Competition solutions** — GPU Mode NVFP4 hackathon, FlashInfer MLSys 2026 submissions

Do NOT use this skill for:

- Generic CUDA / HIP questions unrelated to tensor-core / matrix-core kernel programming
- Host-side framework integration (model loading, request routing, scheduling policy)
- Distributed systems topics — DeepEP, EPLB, DualPipe, ROCSHMEM transport are out of scope

## How To Query

All commands below run from the skill directory (the clone root — the directory this `SKILL.md` lives in). The scripts auto-resolve the wiki root; **no environment variable required**.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to fuse gate-up dual GEMM on Blackwell"
python3 scripts/query.py --tag nvfp4 --type kernel
python3 scripts/query.py --repo cutlass --limit 20
python3 scripts/query.py --symptom tail-effect --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--limit`, `--compact`, `--paths-only`. `--tag`
and `--architecture` accept aliases — `--tag UMMA` matches `tcgen05`,
`--architecture B200` matches `sm100`, etc.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-flash-attention-4
python3 scripts/get_page.py pr-cutlass-2472
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
python3 scripts/get_page.py kernel-flash-attention-4 --body-only
```

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "tcgen05\\.fence"
python3 scripts/grep_wiki.py "2-CTA backward" --only wiki
python3 scripts/grep_wiki.py "nvfp4" "block_scale" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — 15 techniques with architectures, confidence, reproducibility, source count
- `queries/by-hardware-feature.md` — tcgen05/tmem/clc/tma/nvfp4/etc. → related wiki + PR pages
- `queries/by-kernel-type.md` — gemm/attention/moe/mla/gated-delta-net → pages
- `queries/by-language.md` — cute-dsl/cuda-cpp/ptx/triton → guide page + related kernels/sources
- `queries/by-repo.md` — all 2179 PRs across cutlass/sglang/vllm/flashinfer/pytorch/DeepGEMM

### Path 5: Primer, schema, examples

Companion docs under `references/`:

- `references/primer.md` — topic map: hardware features, techniques, symptoms, canonical page IDs. Read this first when the question is broad.
- `references/schema.md` — condensed frontmatter schema, confidence rules, reproducibility ladder, controlled vocabulary, canonical aliases.
- `references/examples.md` — 10 worked query patterns mapping user questions → command sequences → synthesis.

## Output Pattern

When answering from this KB:

1. **Cite specific pages** with paths (e.g., `wiki/kernels/flash-attention-4.md`) and IDs (`kernel-flash-attention-4`).
2. **Follow `sources:` fields** to trace claims back to PRs/blogs/docs.
3. **Respect confidence levels** — `verified` > `source-reported` > `inferred` > `experimental`. Call out when a claim is `experimental` or `inferred`.
4. **Include code snippets** from wiki pages when they exist — technique/kernel/language pages are guaranteed `snippet`-reproducibility (validator-enforced).
5. **Report performance claims with all six fields** — `gpu`, `dtype`, `shape`, `metric`, `value`, `source_id`.

## Knowledge Base Contents (knowledge cutoff: 2026-04-27)

- **2265 total markdown pages** — 2179 PR references + 48 wiki synthesis + 20 blogs + 11 docs + 7 contests
- **6 candidate ledgers** in `candidates/` — 4,222 merged PRs classified (include/defer/exclude) Jan 2025 – Apr 2026
- **89 verbatim/extracted/derived asset bundles** in `artifacts/` (PR diffs, kernel files, blog code) — pinned to upstream SHAs via `PROVENANCE.yaml`
- **6 auto-generated query indices** in `queries/`
- **Controlled vocabulary** (80+ tags) in `data/tags.yaml`, alias map in `data/aliases.yaml`
- **Hybrid version-claim registry** — per-page `version_sensitive: <id>` pointers + `data/version-claims.yaml` central registry, validated for bidirectional consistency
- **Validator** `scripts/validate.py` — 2265 files / 89 bundles / 6 ledgers / 0 errors
- **Blackwell-first for the NVIDIA half** — SM90-only pages only appear when they carry explicit `blackwell_relevance`
- **CDNA 3 / CDNA 4 dual-architecture** — every AMD page declares both `cdna3` and `cdna4` when applicable, with explicit per-arch differences (LDS budget, MFMA shapes, scale-MFMA availability)

The knowledge cutoff date is the last day on which upstream PRs / blog snapshots were refreshed. To advance it: run `scripts/refresh_candidate_ledger.py`, regenerate PR pages, then bump `data/refresh-cutoff.yaml::cutoff_date`.

## Quality Guarantees

- Every `verified` page has official-doc + upstream-code evidence
- Every technique/kernel/language page has a compilable snippet
- Every PR page has `inclusion_reason` and `status: merged`
- All Hopper-inclusive pages have explicit `blackwell_relevance`
