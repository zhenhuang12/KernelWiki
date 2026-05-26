---
id: blog-amd-mori-readme
title: "MoRI — Modular ROCm Inference Kernels (Research Preview)"
author: AMD Research
url: https://github.com/AMD-AIG-AIMA/mori
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- moe
- attention
- mla
- hip
- composable-kernel
- ep-dispatch-combine
- all-to-all
retrieved_at: 2026-04-27
---

## Summary

[AMD-AIG-AIMA/mori](https://github.com/AMD-AIG-AIMA/mori) is AMD's research-oriented modular inference kernel library, focused on rapid iteration of MoE / MLA / sparse-attention patterns on Instinct. Roughly the AMD equivalent of FlashInfer's research staging area before kernels graduate to production AITER.

## Kernel Catalogue

- **MLA decode** (DeepSeek V3-style) — multi-head latent attention with KV-cache compression, FP8 KV.
- **Sparse MoE dispatch** — variant of EP dispatch/combine using RCCL `AllToAll` with in-kernel reshape.
- **Long-context attention** — sliding-window + global-token attention for >32K contexts on MI300X.
- **GatedDeltaNet** — port of the linear-attention variant from the SGLang/FlashInfer side.

## Why It Matters

MoRI is the canonical "what's AMD prototyping next?" repository — kernels here usually appear in AITER 2-3 months later. Treat it as advance signal for what optimizations matter on CDNA 3/4. Listed alongside AITER in the AMD source corpus.
