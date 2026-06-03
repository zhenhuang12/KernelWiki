---
id: blog-amd-flydsl-nightly-wheels
title: "Getting Started with FlyDSL Nightly Wheels on ROCm"
author: "Felix Li, Shijie Feng, Carlus Huang, Dewei Wang, Hongxia Yang, Kiran Thumma, Satya Ramji Ainapurapu, Peng Sun, Emad Barsoum (AMD)"
url: https://rocm.blogs.amd.com/software-tools-optimization/flydsl-nightly-wheel/README.html
source_category: community-note
architectures:
- cdna3
- cdna4
tags:
- flydsl
- jit-compilation
retrieved_at: 2026-06-03
---

## Summary

AMD ROCm blog (2026-04-20) on installing FlyDSL's prebuilt **nightly wheels** instead of building
from source. Documents FlyDSL's *second* distribution channel and pins exact environment
requirements — useful for version alignment. Backs [lang-flydsl](../../wiki/languages/flydsl.md).

## Distribution & Requirements

- **AMD-hosted nightly index** (NOT public PyPI): `https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/`
  install with `uv` (recommended) or `pip`.
- **Python 3.12 or 3.13** (the public-PyPI `flydsl` wheels span cp310–cp314; the two channels
  differ by index host and version scheme, not by Python tag).
- **ROCm 7.1 or 7.2** (ROCm 7.13 not yet supported); PyTorch-with-ROCm; MI300X/MI325X (gfx942) or
  MI350X/MI355X (gfx950).
- Bare-metal or ROCm PyTorch containers.

## Version Scheme

Nightly versions follow `<base_version>+<date>.<commit_hash>` — e.g. `flydsl==0.1.1+20260323.77e1352`.
Base versions seen: 0.1.0 / 0.1.1. (Cross-check: public PyPI tops out at 0.1.8 / 0.2.0.dev626,
repo `main` is at 0.2.0; **no 1.x / 1.18.0 exists on any channel**.)

## Why It Matters

Two coexisting FlyDSL distribution channels (public PyPI 0.1.x cp310–cp314, and the AMD nightly
index `0.1.x+date.commit`, py3.12/3.13) plus a fast-moving `main` mean "the FlyDSL version" is ambiguous —
pin the exact wheel/commit when reproducing. See [lang-flydsl](../../wiki/languages/flydsl.md)
version-basis note.
