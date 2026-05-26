---
id: blog-rocm-mlperf-inference-v51
title: "Technical Dive into AMD MLPerf Inference v5.1"
author: ROCm Engineering
url: https://rocm.blogs.amd.com/artificial-intelligence/mlperf-inference-v5.1/README.html
source_category: benchmark-blog
architectures:
- cdna3
- cdna4
tags:
- gemm
- moe
- fused-moe
- mxfp4
- fp8
- mfma
- mfma-scale-f8f6f4
- block-scale
retrieved_at: 2026-05-26
---

## Summary

AMD's deep-dive on the MLPerf Inference v5.1 submission covering MI325X (CDNA 3) and MI355X (CDNA 4) results. Breaks down end-to-end inference cost by kernel family for Llama2-70B, DeepSeek-R1, and Mixtral, with kernel-by-kernel attribution and the AITER tile-shape selection logic.

## Key Numbers Cited

- **MXFP4 GEMMs account for ~62% of end-to-end cost** on Llama2-70B on MI355X.
- **MI355X delivers ~2.7× over MI325X** on Llama2-70B inference, attributed almost entirely to native MXFP4 block-scaled MFMA replacing FP8 dense MFMA.
- AITER MXFP4 grouped-MoE GEMM on Mixtral-8×22B reaches measured throughput tied to the scaled-MFMA peak.
- Tile-shape selection (block_m/n) is per-shape, picked by an offline tuner; the post explains the heuristic.

## Why It Matters

This is the most credible AMD-published source for *what fraction of inference time is spent in which kernel family* on Instinct hardware. It motivates wiki coverage priority — MoE grouped-GEMM and MXFP4 block-scaled GEMM are the highest-leverage kernels on MI355X, and FP8 KV / paged attention dominate on MI325X.

Cited by [kernel-aiter-fused-moe](../../wiki/kernels/aiter-fused-moe.md), [kernel-ck-mxfp4-gemm-cdna4](../../wiki/kernels/ck-mxfp4-gemm-cdna4.md).
