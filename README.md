# XT-Luna-Zero

**Architecture of a 194M-Parameter Sparse Decoder**
Technical report draft · November 6, 2026 · XtrComSu

📄 **[Read the paper (PDF)](paper/XT-Luna-Zero-Architecture.pdf)**

> **This README was updated on 2026-11-08.** The paper describes the *original* 194M
> design and its numerical checks, and has not been rewritten. Real-text training has
> since happened on a larger model, and this README now covers both. Where the two
> differ, the sections are labelled. The paper is unchanged; treat it as the record of
> the design phase it documents.

## Abstract

XT-Luna-Zero is a decoder-only language model built to train from scratch on a
single GPU. Sixteen blocks pair causal attention with a feed-forward mixture of
experts: one shared expert and four routed experts in every block. Top-1
routing puts the model at a reported 194,048,640 stored parameters and
99,676,800 active parameters per token, counted the way this project counts
them.

The report follows the forward path, works through the sparse parameter
budget, and separates the numerical checks that were actually run from the
work that was not.

**That abstract describes the design phase.** At the time the paper was written there
was no real-text training checkpoint; language quality, reasoning and long-context
recall were unmeasured. They have since been measured on a larger model, in
[Models](#models) below.

## Models

Two model definitions exist in this project. Both are verified by instantiating them
and summing parameters, not by reading a config.

### XT-Luna-Zero (v3), the 194M design this report documents

| | |
|---|---|
| Stored parameters | **194,048,640** |
| Active per token | 99,676,800 |
| Layers | 16 |
| Width | 640 |
| Heads | 10 query / 2 KV (GQA) |
| Experts per block | 1 shared + 4 routed, top-1 |
| FFN width | 1024 |
| Vocabulary | 32,768 (tied embedding) |
| Attention | sliding window 2048, every 4th layer global |
| Positional | RoPE θ=500,000, max 131,072 |
| Definition | `molab/model_v3.py` |

This is the architecture the paper's parameter accounting describes. Its engineering
checks are the attention output against an explicit causal-window reference in FP32 at
lengths from 1,024 to 6,144.

### XT-Luna Core, the model that was actually trained

| | |
|---|---|
| Stored parameters | **374,989,440** |
| Active per token | **139,059,840** |
| Layers | 24 |
| Width | 640 |
| Heads | 10 query / 2 KV (GQA), head dim 64 |
| Experts per block | **1 shared + 6 routed, top-1** |
| FFN width | 1024 |
| Vocabulary | 32,768 (tied embedding) |
| Attention | sliding window 2048, every 4th layer global |
| Positional | RoPE θ=500,000, max 131,072 |
| Optimizer | Muon (Kimi-style) on hidden matrices, AdamW elsewhere |
| Definition | `molab/model_core.py` |

The active-parameter count is checkable by hand. Each routed expert holds

```
3 · d · d_ff = 3 · 640 · 1024 = 1,966,080
```

so skipping five of six per layer saves `5 · 1,966,080 = 9,830,400` per layer, and over
24 layers `235,929,600`. And `374,989,440 - 235,929,600 = 139,059,840`, exactly the
active figure. That arithmetic only closes for **top-1 of six**; it is how the count was
verified rather than assumed.

## Measured results

Held-out cross-entropy, byte-level BPE, 8 windows of 2048 tokens per domain (16,384
tokens per point). Lower is better.

| Tokens | Mean | General | Science | Math | Code | Top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03B | 4.1230 | 5.4140 | 4.4630 | 3.6560 | 2.9590 | 0.3310 |
| 0.20B | 2.4930 | 3.9960 | 2.9390 | 1.6770 | 1.3590 | 0.5210 |
| 0.40B | 2.3400 | 3.7860 | 2.7110 | 1.5850 | 1.2780 | 0.5380 |
| 0.60B | 2.2502 | 3.7061 | 2.5972 | 1.5023 | 1.1950 | 0.5515 |
| 0.80B | 2.1845 | 3.6548 | 2.5364 | 1.4449 | 1.1017 | 0.5654 |
| 1.00B | 2.1578 | 3.6312 | 2.4608 | 1.3964 | 1.1427 | 0.5650 |
| 1.20B | 2.1424 | 3.6009 | 2.4223 | 1.4379 | 1.1085 | 0.5693 |
| 1.40B | 2.1203 | 3.5817 | 2.3800 | 1.4193 | 1.1001 | 0.5690 |
| 1.60B | 2.0937 | 3.5677 | 2.3753 | 1.3660 | 1.0657 | 0.5746 |
| 1.80B | 2.0724 | 3.5296 | 2.3432 | 1.3668 | 1.0498 | 0.5770 |
| 2.00B | 2.0511 | 3.5338 | 2.3077 | 1.3540 | 1.0088 | 0.5815 |
| 2.20B | 2.0289 | 3.5002 | 2.2884 | 1.3181 | 1.0088 | 0.5860 |
| 2.40B | 2.0119 | 3.4884 | 2.2674 | 1.3035 | 0.9886 | 0.5884 |
| 2.60B † | 2.2275 | 3.4527 | 2.2387 | 1.2930 | 1.9255 | 0.5462 |
| 2.80B † | 2.2259 | 3.4289 | 2.2180 | 1.3646 | 1.8920 | 0.5435 |

† From 2.6B the training sequence length changed from 2048 to 8192. Evaluation stays at
2048 throughout, so these rows are comparable in measurement but not in training
condition. The mean rises and code more than doubles while general, science and math all
continue to improve. The model is adapting to longer training sequences, and the mean
is being carried by code. These rows are **not** a quality regression and should not be
read as one.

Observed trend before the switch was `approximately -0.11 nats per billion tokens`,
stable from 1.2B to 2.4B.

### What the curve does not say

- It is a **single** held-out set at 8 windows per domain. The standard error is about
  0.08 to 0.10 nats, so differences below roughly 0.2 nats between adjacent points are
  not individually meaningful; the trend across many points is.
- It measures next-token prediction on held-out text of the same distribution. It says
  nothing about instruction following, reasoning chains, or long-context recall.
- Code crossed below math at 2.0B and is the only domain that changed discontinuously at
  the sequence-length switch. Both facts are unexplained by this curve alone.

## Corpus

Trained on 5.5B tokens from a byte-level BPE vocabulary of 32,768, drawn from fineweb-edu
(general), open-web-math (science), OpenMathInstruct-2 and NuminaMath (math), and
OpenCodeInstruct and OpenCodeReasoning (code), plus dialogue and long-document data in
the later mixture.

A corpus rebuild for a 20B-token run is in progress, sized from measured dataset capacity
rather than dataset cards. One finding worth recording: the original mixture sampled math
at **5.35× repetition** against a 51M-token bucket, and math is the only domain showing no
improvement in paired per-window comparisons. Repetition moves loss without adding
capability.

## Status

| Stage | State |
|---|---|
| Architecture and parameter accounting | Complete and verified by instantiation |
| Attention reference checks (FP32, 1,024 to 6,144) | Complete |
| BF16 training step | Complete |
| Real-text training, 194M design | **Not run** |
| Real-text training, 375M core | **Complete to 2.8B tokens**, continuing |
| Held-out evaluation from 0.03B to 2.8B | Complete, table above |
| Long-context recall | **Unmeasured**. The memory probe is a synthetic proxy, not a benchmark |
| Instruction following / reasoning | **Unmeasured** |

## Planned experiments

The paper's §6 names these as future work. **None has been started.** They are listed
here so the roadmap is visible, not to imply progress:

- a larger flat control
- phase-specific expert pools
- chained reasoning
- Grassmann mixing
- rhythm conditioning
- multi-token prediction
- a DeepSeek-inspired reading/writing split

They are **not** numbered V1-V5 in any document in this project, and no results exist for
any of them. Any ablation claim about this architecture is currently unsupported: without
a matched control run, an architectural change cannot be separated from seed variance.

## Citation

```bibtex
@techreport{xtlunazero2026,
  title  = {XT-Luna-Zero: Architecture of a 194M-Parameter Sparse Decoder},
  author = {XtrComSu},
  year   = {2026},
  month  = {November},
  type   = {Technical report draft},
  url    = {https://github.com/XtrComSu/XT-Luna-Zero}
}
```

The paper cites the 194M design. If you are citing the measured results, cite them as
the 375M core model, not as the paper's architecture.

Machine-readable metadata sits in [`CITATION.cff`](CITATION.cff).

## Lineage

Architectural ideas are drawn from [Kimi Linear](https://arxiv.org/abs/2510.26692)
(Kimi Delta Attention), [Kimi K2](https://arxiv.org/abs/2507.20534)
(MuonClip and QK-clip), [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
(auxiliary-loss-free MoE balancing), [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
(shared expert isolation), and [SmolLM2](https://arxiv.org/abs/2502.02737)
(small-model data curation).

## License

© 2026 XtrComSu. Released under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).

Share it, remix it, use it commercially. Just give credit, link the license,
and say whether you changed anything.

Works cited in the report's reference list belong to their original authors
and stay under their own terms. This license does not extend to them.
