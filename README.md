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
| Optimizer | **AdamW** (β 0.9/0.95, eps 1e-8, wd 0.1 on ndim≥2). A HybridMuonAdamW exists behind `--optimizer muon` but is **not** in use: see note below |
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

| Tokens | General | Science | Math | Code | Convo | Long | Top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.03B | 5.4140 | 4.4630 | 3.6560 | 2.9590 | – | – | 0.3310 |
| 0.20B | 3.9960 | 2.9390 | 1.6770 | 1.3590 | – | – | 0.5210 |
| 0.40B | 3.7860 | 2.7110 | 1.5850 | 1.2780 | – | – | 0.5380 |
| 0.80B | 3.6548 | 2.5364 | 1.4449 | 1.1017 | 2.2360 | 3.2150 | 0.5654 |
| 1.20B | 3.6009 | 2.4223 | 1.4379 | 1.1085 | 2.1970 | 3.1950 | 0.5693 |
| 1.60B | 3.5677 | 2.3753 | 1.3660 | 1.0657 | 2.1720 | 3.1630 | 0.5746 |
| 2.00B | 3.5338 | 2.3077 | 1.3540 | 1.0088 | 2.1100 | 3.1240 | 0.5815 |
| 2.20B | 3.5002 | 2.2884 | 1.3181 | 1.0088 | 2.0960 | 3.1390 | 0.5860 |
| 2.40B | 3.4884 | 2.2674 | 1.3035 | 0.9886 | – | – | 0.5884 |
| 2.60B † | 3.4527 | 2.2387 | 1.2930 | 1.9255 | – | – | 0.5462 |
| 2.80B † | 3.4289 | 2.2180 | 1.3646 | 1.8920 | – | – | 0.5435 |
| **5.23B** | **3.3613** | **2.0502** | **1.0149** | **1.6088** | **1.6333** | **3.0388** | **0.5777** |

The 2.4B → 5.23B rows are **not** a clean continuation, and the gap is documented rather
than smoothed over. Three changes landed between them: the sequence length went to 8192
and back to 2048, the data mixture was rebuilt (the `arith` mix, which raised general,
science, math, code, convo, long and arithmetic shares), and the corpus was regenerated.
The 5.23B row is measured on 32 windows per domain (65,536 tokens), the earlier rows on
8 windows; a larger measurement is a *different* estimate, not a strictly better one.

Reading the movement honestly:
- **science 2.2674 → 2.0502** and **math 1.3035 → 1.0149** are large, real gains.
- **general 3.4884 → 3.3613** continues the pre-existing trend.
- **code 0.9886 → 1.6088 is a genuine regression**, not noise. It is 0.62 nats, far above
  the 0.066 standard error. The trajectory locates it: code was 0.9886 at 2.40B and
  1.9255 at 2.60B † — it more than doubled in the single step where the training sequence
  went from 2048 to 8192, and no other domain moved that way. It has since recovered to
  1.6088 without returning to its 2.4B level. The honest reading is that code is the
  domain most sensitive to the sequence-length change, and that the 8192 experiment was
  not undone cleanly for it. The corpus rebuild (code is 160M tokens, the thinnest bucket
  at 8 shards) is a plausible contributing factor but is **not** established by this
  data — attributing the regression to it would be a guess.
- **mean is intentionally omitted** for the 5.23B row: averaging six domains that changed
  mixture weights mid-run produces a number whose movement cannot be attributed.

Observed trend before the corpus rebuild was `approximately -0.11 nats per billion
tokens`, stable from 1.2B to 2.4B, on general/science/math/code.

### Why there is no Muon here

The table above previously listed Muon as the optimizer. It is not. Training runs
**AdamW**. `HybridMuonAdamW` exists in `molab/muon.py` and is selectable with
`--optimizer muon`, but it was measured and rejected for this run on two grounds:

1. **Cost:** +7.7% step time (0.648 → 0.698 s/step at batch 16), with 353M of 375M
   parameters routed to Muon.
2. **Break-even:** the published absolute gap is roughly 2B tokens of progress. Switching
   optimizer alone therefore costs about **+1.02 days** on this run before it pays back;
   bundled with a full tokenizer-v2 restart the estimate turns marginally negative
   (−0.14 days), which is inside the noise of the estimate itself.

There is a harder blocker: `HybridMuonAdamW.load_state_dict` raises on a non-hybrid
checkpoint, so Muon cannot be adopted by resuming — it requires restarting from step 0.
At 5.2B tokens in, that trade is not close.

**QK-Clip** (Kimi K2) is implemented and verified but **disabled**. Attention logits were
measured exactly on the real checkpoint: max **167.0** at layer 16, with 9 of 24 layers
above τ=100. Enabling the clip mid-run would change numerics discontinuously, so it is
armed for the next run rather than this one.

### What the curve does not say

- It is a **single** held-out set at 8 windows per domain. The standard error is about
  0.08 to 0.10 nats, so differences below roughly 0.2 nats between adjacent points are
  not individually meaningful; the trend across many points is.
- It measures next-token prediction on held-out text of the same distribution. It says
  nothing about instruction following, reasoning chains, or long-context recall.
- Code crossed below math at 2.0B and is the only domain that changed discontinuously at
  the sequence-length switch. Both facts are unexplained by this curve alone.

## Corpus

Trained on 5.23B tokens from a byte-level BPE vocabulary of 32,768. The corpus was
rebuilt for a 20B-token run, sized from **measured** dataset capacity rather than dataset
cards.

Current token counts per bucket, as shards on disk:

| Bucket | Tokens | Source |
|---|---:|---|
| general | 2.48B | fineweb-edu |
| science | 1.32B | open-web-math |
| math | 51M | OpenMathInstruct-2, NuminaMath, FineMath |
| code | 160M | OpenCodeInstruct, OpenCodeReasoning |
| convo | 600M | WildChat and multi-turn dialogue |
| long | 2.03B | long-document web text |
| arithmetic | 350M | synthesised, 3.8M problems |
| indic | in progress | English Wikipedia, India-related articles |

Bytes per token are **not** uniform, which matters more than it looks for interpreting the
loss table: general 4.56, science 4.98, code 2.50, math 2.21. A nats improvement on math is
therefore not comparable to the same number on general — math packs roughly twice as much
text into each token, so each token carries less information.

Two findings worth recording:

- The original mixture sampled math at **5.35× repetition** against a 51M-token bucket,
  and math was the only domain showing no improvement in paired per-window comparisons.
  Repetition moves loss without adding capability.
- **Indic history and culture is English-only by decision.** Native-script corpora such as
  ai4bharat/sangraha would add ~22 scripts to a 32k vocabulary that was never budgeted for
  them. The bucket is English Wikipedia articles selected as India-related, which adds
  knowledge without diluting the tokenizer.
- There is **no shuffling** in the training loop: batches are consumed strictly
  sequentially and position advances monotonically. Duplication was measured independently
  (0 duplicates in a 400-document sample), so this is not currently harmful, but it is a
  known limitation rather than a design choice.

## Status

| Stage | State |
|---|---|
| Architecture and parameter accounting | Complete and verified by instantiation |
| Attention reference checks (FP32, 1,024 to 6,144) | Complete |
| BF16 training step | Complete |
| Real-text training, 194M design | **Not run** |
| Real-text training, 375M core | **Complete to 5.23B tokens**, running toward 20B |
| Held-out evaluation from 0.03B to 5.23B | Complete, table above |
| Long-context recall | **Unmeasured and currently blocked.** Sequence length is 2048, and `WINDOW=2048` means every layer has taken the full-attention branch: the sliding window has never been active and the global layers have never seen a position beyond 2047. Corpus supply is also thin (12.6% of general tokens live in documents ≥8K). |
| Laptop inference | **Working.** A verified CPU runner ships with the public checkpoint; measured 19–23 tok/s |
| Instruction following / reasoning | Partially measured: 20-turn chat probe, refusal rate 0.40, turn-1 relevance 0.53 falling to 0.38 on later turns |

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
