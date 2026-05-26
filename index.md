# GPU Kernel Optimization Knowledge Base

> Comprehensive knowledge base for GPU kernel optimization on NVIDIA Blackwell (SM100) / Hopper (SM90) and AMD Instinct MI300X (CDNA 3) / MI355X (CDNA 4).
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and conventions.
> **For Claude Code agents**: this repository is a Claude Code skill — see [SKILL.md](SKILL.md).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|pr|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for 10 worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a specific technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a specific kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a specific language/DSL | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

### NVIDIA Blackwell / Hopper
- [hw-tcgen05-mma](wiki/hardware/tcgen05-mma.md) — Blackwell MMA instruction (replaces wgmma)
- [hw-tmem](wiki/hardware/tmem.md) — Tensor Memory (256KB dedicated accumulator storage)
- [hw-clc](wiki/hardware/clc.md) — Cluster Launch Control (dynamic tile scheduling)
- [hw-tma](wiki/hardware/tma.md) — Tensor Memory Accelerator (async bulk loads)
- [hw-2sm-cooperative](wiki/hardware/2sm-cooperative.md) — Two-SM cooperative MMA
- [hw-nvfp4](wiki/hardware/nvfp4.md) — NVFP4 and block-scaled narrow precision
- [hw-pdl-gdc](wiki/hardware/pdl-gdc.md) — Programmatic Dependent Launch / Grid Dependency Control

### AMD CDNA 3 / CDNA 4
- [hw-mfma](wiki/hardware/mfma.md) — MFMA matrix-fused-multiply-add (Matrix Cores)
- [hw-mfma-scale-f8f6f4](wiki/hardware/mfma-scale-f8f6f4.md) — Block-scaled MFMA on CDNA 4 (MXFP4/6/8)
- [hw-agpr](wiki/hardware/agpr.md) — Accumulation VGPRs (512-register shared pool)
- [hw-lds](wiki/hardware/lds.md) — Local Data Share (64 KB / 160 KB)
- [hw-buffer-load-lds](wiki/hardware/buffer-load-lds.md) — Direct-to-LDS HBM loads (AMD's TMA equivalent)
- [hw-xcd](wiki/hardware/xcd.md) — 8-chiplet topology (MI300X / MI355X)
- [hw-infinity-cache](wiki/hardware/infinity-cache.md) — 256 MB MALL last-level cache
- [hw-mxfp-cdna4](wiki/hardware/mxfp-cdna4.md) — OCP microscaling formats (MXFP4/6/8)

## Optimization Techniques

### Cross-architecture
- [technique-warp-specialization](wiki/techniques/warp-specialization.md) — Warp role assignment (NVIDIA)
- [technique-persistent-kernels](wiki/techniques/persistent-kernels.md) — Persistent kernel patterns with CLC
- [technique-swizzling](wiki/techniques/swizzling.md) — Shared memory swizzling
- [technique-pipeline-stages](wiki/techniques/pipeline-stages.md) — Software pipelining
- [technique-epilogue-fusion](wiki/techniques/epilogue-fusion.md) — Fusing epilogue with mainloop
- [technique-tile-scheduling](wiki/techniques/tile-scheduling.md) — Tile scheduling strategies
- [technique-double-buffering](wiki/techniques/double-buffering.md) — Double/multi-buffering
- [technique-software-exp](wiki/techniques/software-exp.md) — Software-emulated exponential
- [technique-fine-grained-quant](wiki/techniques/fine-grained-quantization.md) — Fine-grained FP8/FP4 quantization
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md) — Wide vectorized loads and cache policies

### AMD CDNA-specific
- [technique-wave-specialization](wiki/techniques/wave-specialization.md) — Producer/consumer waves on CDNA
- [technique-mfma-pipelining](wiki/techniques/mfma-pipelining.md) — MFMA software pipelining (3 levels)
- [technique-direct-to-lds](wiki/techniques/direct-to-lds.md) — `buffer_load_*_lds` HBM → LDS
- [technique-lds-swizzling](wiki/techniques/lds-swizzling.md) — LDS XOR swizzle for bank-conflict avoidance
- [technique-xcd-aware-scheduling](wiki/techniques/xcd-aware-scheduling.md) — `blockIdx.x` remap for 8-XCD locality

## Kernel Case Studies

### NVIDIA Blackwell / Hopper
- [kernel-flash-attention-4](wiki/kernels/flash-attention-4.md) — FlashAttention-4 (1605 TFLOPS on B200)
- [kernel-deepgemm](wiki/kernels/deepgemm.md) — DeepGEMM FP8 GEMM (1550 TFLOPS on H800)
- [kernel-flashmla](wiki/kernels/flashmla.md) — FlashMLA sparse/dense MLA decoding
- [kernel-nsa](wiki/kernels/nsa.md) — Native Sparse Attention (9x fwd speedup)
- [kernel-gated-delta-net](wiki/kernels/gated-delta-net.md) — Gated Delta Net linear attention
- [kernel-nvfp4-gemm](wiki/kernels/nvfp4-gemm.md) — NVFP4 GEMM from GPU Mode hackathon
- [kernel-nvfp4-gemv](wiki/kernels/nvfp4-gemv.md) — NVFP4 batched GEMV optimization
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Fused MoE with FP8

### AMD CDNA 3 / CDNA 4
- [kernel-ck-fp8-gemm-cdna3](wiki/kernels/ck-fp8-gemm-cdna3.md) — CK-Tile V3 FP8 GEMM (1240 TFLOPS on MI300X)
- [kernel-ck-mxfp4-gemm-cdna4](wiki/kernels/ck-mxfp4-gemm-cdna4.md) — CK-Tile MXFP4 GEMM (4720 TFLOPS on MI355X)
- [kernel-aiter-fused-moe](wiki/kernels/aiter-fused-moe.md) — AITER fused MoE (2.1× over vLLM baseline)
- [kernel-aiter-mla-decode](wiki/kernels/aiter-mla-decode.md) — AITER MLA decode (~87% HBM utilization)

## Problem → Solution Patterns

- [pattern-low-sm-utilization](wiki/patterns/low-sm-utilization.md) — SM utilization is low
- [pattern-memory-bound](wiki/patterns/memory-bound.md) — Kernel is memory bandwidth limited
- [pattern-register-pressure](wiki/patterns/register-pressure.md) — Too many registers → low occupancy
- [pattern-compute-bound](wiki/patterns/compute-bound.md) — Not reaching peak FLOPS
- [pattern-tail-effect](wiki/patterns/tail-effect.md) — Last wave underutilizes GPU
- [pattern-moe-load-imbalance](wiki/patterns/moe-load-imbalance.md) — Expert token-count skew starves SMs/CUs
- [pattern-pipeline-stalls](wiki/patterns/pipeline-stalls.md) — Producer/consumer pipeline bubbles drop throughput

## Languages & DSLs

### NVIDIA
- [lang-cute-dsl](wiki/languages/cute-dsl.md) — CuTe DSL for Blackwell
- [lang-cuda-cpp](wiki/languages/cuda-cpp.md) — CUDA C++ with PTX inline
- [lang-ptx](wiki/languages/ptx-sm100.md) — PTX instructions for SM100
- [lang-triton](wiki/languages/triton-blackwell.md) — Triton on Blackwell

### AMD
- [lang-hip](wiki/languages/hip.md) — HIP (CUDA → HIP cheat sheet)
- [lang-composable-kernel](wiki/languages/composable-kernel.md) — CK / CK-Tile (production C++)
- [lang-amdgcn-asm](wiki/languages/amdgcn-asm.md) — AMDGCN inline assembly
- [lang-flydsl](wiki/languages/flydsl.md) — FlyDSL Python+MLIR research DSL

## Migration Guides

- [migration-wgmma-to-tcgen05](wiki/migration/wgmma-to-tcgen05.md) — Hopper wgmma → Blackwell tcgen05
- [migration-register-to-tmem](wiki/migration/register-to-tmem.md) — Register accumulators → TMEM
- [migration-cuda-to-hip](wiki/migration/cuda-to-hip.md) — CUDA / SM90 → HIP / CDNA 3

## Source Repositories

### NVIDIA stack
| Repository | Focus |
|---|---|
| [NVIDIA/cutlass](queries/by-repo.md#nvidiacutlass) | CUTLASS 4.x Blackwell support |
| [sgl-project/sglang](queries/by-repo.md#sgl-projectsglang) | SGLang Blackwell integration |
| [vllm-project/vllm](queries/by-repo.md#vllm-projectvllm) | vLLM Blackwell support |
| [flashinfer-ai/flashinfer](queries/by-repo.md#flashinfer-aiflashinfer) | FlashInfer Blackwell kernels |
| [pytorch/pytorch](queries/by-repo.md#pytorchpytorch) | PyTorch/Inductor Blackwell |
| [deepseek-ai/DeepGEMM](queries/by-repo.md#deepseek-aideepgemm) | DeepGEMM FP8/FP4 GEMM + Mega MoE on Hopper/Blackwell |

### AMD stack
| Repository | Focus |
|---|---|
| [ROCm/composable_kernel](queries/by-repo.md#rocmcomposable_kernel) | CK / CK-Tile production GEMM/FMHA on CDNA 3/4 (note: ROCm/composable_kernel is deprecated; active development moved to ROCm/rocm-libraries) |
| [ROCm/aiter](queries/by-repo.md#rocmaiter) | AITER fused MoE / MLA / quantized inference kernels |
| [ROCm/rccl](queries/by-repo.md#rocmrccl) | RCCL collectives (NCCL counterpart). RCCL active development at [ROCm/rocm-systems/projects/rccl](https://github.com/ROCm/rocm-systems/tree/main/projects/rccl); ROCm/rccl receives backport cherry-picks for stable branches. |
| [ROCm/rocm-systems](queries/by-repo.md#rocmrocm-systems) | Active RCCL development (projects/rccl); ROCm/rccl receives backport cherry-picks for stable branches. |
| ROCm/FlyDSL | FlyDSL research DSL (Python + MLIR) — no captured PRs yet. |
| ROCm/mori | MoE Expert-Parallel dispatch/combine library + shmem/RDMA primitives — GitHub tagline "Modular RDMA Interface" undersells the EP role. No captured PRs yet. |

## Competitions

- [GPU Mode NVFP4 Hackathon](sources/contests/gpu-mode-nvfp4/) — 4 NVFP4 kernel challenges on B200
- [FlashInfer MLSys 2026](sources/contests/flashinfer-mlsys26/) — MoE, Sparse Attention, GatedDeltaNet
