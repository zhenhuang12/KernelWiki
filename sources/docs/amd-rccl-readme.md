---
id: doc-amd-rccl-readme
title: "AMD RCCL — ROCm Collective Communications Library"
url: https://github.com/ROCm/rccl
source_category: official-doc
architectures:
- cdna3
- cdna4
tags:
- all-reduce
- all-to-all
- xgmi
- infinity-fabric
- ep-dispatch-combine
retrieved_at: 2026-04-27
---

# AMD RCCL — ROCm Collective Communications Library

## Overview

[ROCm/rccl](https://github.com/ROCm/rccl) is the AMD analogue of NVIDIA NCCL — a multi-GPU/multi-node collective library implementing `AllReduce`, `AllGather`, `ReduceScatter`, `Broadcast`, `AllToAll`, `Send`/`Recv`, and grouped variants. RCCL is API-compatible with NCCL (same `ncclXxx` symbols in many transitional builds; native `rcclXxx` aliases also exported), so most CUDA distributed code ports cleanly.

## Key Differences vs NCCL

- **Transport**: xGMI (Infinity Fabric) for intra-node, RDMA over RoCE / Slingshot for inter-node. No NVLink/NVSwitch.
- **Topology**: 8-GPU OAM systems use a fully connected xGMI mesh (no switch); large clusters use HSN networks.
- **Algorithm tuning**: RCCL ships its own per-topology algorithm tuner (`RCCL_ENABLE_TOPO_TUNER=1`) — chunk sizes and tree-vs-ring splits differ from NCCL defaults.
- **MSCCL backend**: RCCL bundles MSCCL-style algorithm specification, allowing custom collective DAGs (key for MoE EP dispatch/combine on Instinct).

## Relevance to Kernel Work

For pure single-GPU kernel work, RCCL is out of scope. But for MoE expert parallelism and KV-cache sharding, RCCL `AllToAll` and `AllReduce` performance directly bounds end-to-end inference throughput. Kernel-level work increasingly fuses small reductions into RCCL post/pre-processing (e.g. AITER's `fused_moe` exposes a `topk_weights_reduce` hook that RCCL `AllReduce` consumes). See [kernel-aiter-fused-moe](../../wiki/kernels/aiter-fused-moe.md).

## Why It Matters

RCCL is the AMD collective. Any cross-GPU work on Instinct (EP dispatch/combine, TP all-reduce, DP allgather) routes through it. Treat as out-of-scope for single-CTA kernel optimization but in-scope when reasoning about end-to-end MoE / TP latency.
