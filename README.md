# XT-Luna-Zero

**Architecture of a 194M-Parameter Sparse Decoder**
Technical report draft · November 6, 2026 · XtrComSu

📄 **[Read the paper (PDF)](paper/XT-Luna-Zero-Architecture.pdf)**

> **This README was updated on 2026-11-17.** The paper describes the *original* 194M
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

Held-out cross-entropy, byte-level BPE, 2048-token windows. Lower is better.

**The window count is not constant across this table, and that matters.** Rows up to
5.23B use 8–32 windows per domain (16,384–65,536 tokens) with standard error 0.08–0.10
nats. From 5.80B the evaluator uses **48 windows (98,304 tokens)** and stays there, with
standard error 0.003–0.012. A larger measurement is a *different* estimate, not a
strictly better one, so **only the 5.80B–6.40B rows can be differenced against each
other**; comparisons spanning the boundary carry the 8-window noise on one side.

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

<!-- EVAL-TABLE:START -->

_Auto-generated from `eval_curve.jsonl` at 10 comparable milestones. Do not edit between the markers._

Held-out cross-entropy, 48 windows of 2048 tokens per domain (98,304 tokens; standard error 0.003-0.012). Lower is better.

| Tokens | General | Science | Math | Code | Convo | Long | Top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.80B | 3.2455 | 1.9917 | 1.0674 | 1.6918 | 1.6678 | 2.8613 | 0.5761 |
| 6.00B | 3.2375 | 1.9903 | 1.1204 | 1.7273 | 1.6606 | 2.8240 | 0.5708 |
| 6.20B | 3.2113 | 1.9556 | 1.1086 | **1.6427** | **1.6392** | 2.8488 | 0.5785 |
| 6.40B | 3.2048 | 1.9485 | 0.9487 | 1.6518 | 1.6578 | 2.7840 | 0.5858 |
| 6.60B | 3.1813 | **1.9314** | 0.9200 | 1.6719 | 1.6485 | 2.7572 | 0.5888 |
| 6.80B | **3.1777** | 1.9350 | **0.9077** | 1.6670 | 1.6630 | **2.7308** | 0.5892 |
| 7.60B | 3.1500 | 1.9050 | 1.0290 | 1.4700 | 1.6630 | 2.6760 | 0.5920 |
| 7.80B | 3.1520 | 1.9080 | 1.0600 | 1.4530 | 1.6670 | 2.6580 | 0.5910 |
| 8.00B | 3.1590 | 1.9130 | 1.1190 | 1.3290 | 1.7030 | 2.6960 | 0.5900 |
| 8.20B | 3.1370 | 1.8905 | 1.1494 | 1.2939 | 1.5301 | 2.6560 | 0.5918 |

Latest comparable milestone: **8.20B**, mean 1.8677, top-1 0.5918. Code and convo both improved sharply at 8.20B (1.3290 -> 1.2939 and 1.7030 -> 1.5301). Bold marks each domain's best value in this series.

<!-- EVAL-TABLE:END -->

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

### The code regression at 5.8B → 6.0B, and how it was fixed

From 5.80B the evaluation switched to 48 windows per domain (98,304 tokens) and has
stayed there, so **5.80B onward is a paired series**: identical windows at every
milestone, standard error 0.003–0.012 instead of 0.08–0.10. Only those four rows can be
differenced.

| Domain | 5.80B | 6.00B | Δ | t | Verdict |
|---|---:|---:|---:|---:|---|
| math | 1.0674 | 1.1204 | **+0.0530** | 13.27 | worse |
| code | 1.6918 | 1.7273 | **+0.0354** | 5.01 | worse |
| long | 2.8613 | 2.8240 | −0.0374 | −5.28 | better |
| general | 3.2455 | 3.2375 | −0.0080 | −1.89 | flat |
| science | 1.9917 | 1.9903 | −0.0014 | −0.46 | flat |
| convo | 1.6678 | 1.6606 | −0.0072 | −0.22 | flat |

Code and math got worse while everything else held. That pattern is the finding, and it
was not noise: across the six domains, **Pearson r(epochs consumed, loss delta) = +0.82**.
The two regressions were exactly the two most-repeated buckets — math at 9.13 epochs and
code at 2.03 — while every under-repeated bucket (long 0.08×, convo 0.47×, general 0.76×)
held or improved.

The cause was supply, not capacity. `nvidia/OpenCodeReasoning` yields about 0.16B tokens
in total, so the code bucket was at its **source** limit at 0.220B, not a conservative
cap. The mixture allocator (`_supply_matched`) therefore cut code to 1.81% and moved the
freed budget into long (25.8%).

The fix was to add real data rather than redistribute the shortage:

| Bucket | Before | After | Added from |
|---|---:|---:|---|
| code | 0.220B | 1.223B | open-r1/codeforces-cots, KodCode-V1, smollm-corpus `python-edu` |
| math | 0.051B | 0.952B | OpenMathReasoning `tir`, OpenR1-Math-220k, NuminaMath-CoT, MathInstruct |
| general | 3.307B | 4.208B | fineweb-edu sample-10BT / sample-100BT |
| science | 1.653B | 2.153B | cosmopedia-v2, finemath-4plus |

1.003B new code tokens were built with **190,692 exact duplicate rows filtered**; the
math build filtered 72,192. Mixtures then moved to a **pool-proportional** rule, and
later to an **equal-final-epochs** rule (see the mixture section).

Outcome — every domain better than the 5.8B alarm point:

| Domain | 5.8→6.0 | 6.0→6.2 | 6.2→6.4 | Net |
|---|---:|---:|---:|---:|
| general | −0.0080 | −0.0263 | −0.0065 | **−0.0408** |
| science | −0.0014 | −0.0347 | −0.0070 | **−0.0431** |
| math | +0.0530 | −0.0118 | −0.1599 | **−0.1187** |
| code | +0.0354 | −0.0846 | +0.0091 | **−0.0401** |
| convo | −0.0072 | −0.0214 | +0.0186 | **−0.0100** |
| long | −0.0374 | +0.0248 | −0.0648 | **−0.0774** |

Code went 1.6918 → 1.7273 (worst) → 1.6427 → 1.6518, i.e. it peaked at the point of
maximum restriction and recovered once the bucket was 5.6× larger.

The math move at 6.4B (−0.1599, t = −30.47) is the largest single milestone in the run,
so it was checked for contamination rather than accepted:

- the `heldout_*.bin` shards were not touched by the builders (their mtimes predate both
  builds), so the eval windows are unchanged;
- the heldout text is **old-format** LaTeX ("Example 11 … $f(x,y)=(x-y)^2+…$") while the
  new training shards are `think`-prefixed chain-of-thought, i.e. different distributions;
- a 40-token heldout probe appears **zero** times in the newest 20 training shards.

So the gain is transfer to a different format, not memorisation of the eval. One
milestone is still one milestone, and the 6.6B point is the confirmation to watch.

### Why repetition is the thing to control

Two independent observations in this run say the same thing:

1. At 5.35× repetition the original mixture's math bucket was the only domain showing no
   improvement in paired per-window comparisons.
2. At 9.13× (math) and 2.03× (code) both domains regressed, at r = +0.82 between epochs
   and loss delta.

The mixture now targets an **equal final epoch count per bucket** rather than equal
shares, because after the builds the buckets were at very different points in their
pools (general 0.70× consumed, long 0.22×). A pure pool split would have ended the run
with general and science near 1.44× and long near 0.94×. Solving for equal *final*
epochs gives **1.28–1.29× for all eight buckets** — a 0.009-epoch spread.

### Code is the one domain still moving the wrong way

At 6.60B the paired deltas were general −0.0235 (t=−5.4), science −0.0172 (t=−5.2),
math −0.0287 (t=−11.1), long −0.0267 (t=−4.4), convo −0.0092 (t=−0.9, flat), and
**code +0.0201 (t=+3.0)**. Overall mean fell 1.9385 → 1.9261 and top-1 rose
0.5858 → 0.5888, so the run is improving — but code has now risen at three of the last
four points.

The cause was not found in the mixture, and is worth recording because the obvious
hypothesis was wrong. Code is *not* being over-repeated: at v7 it gets 1.29×, the same
as every other bucket. Two measurements point elsewhere.

**1. A small tail of unusable training data.** Sampling every one of the 212 code
train shards (5 windows each, decoding on EOT boundaries) found **11 shards where over
half the documents are Git blob hashes and repository paths** rather than code:

```
dc44aed22f1ce2ff83aae4/Leetcode/Trees/Medium/865_smallest_subtree...py
0529d754f9929c6bfadabd3bfea2156f8114ee59/Leetcode/Arrays/Easy/665_non_dec...
```

That is 0.055B of the 1.223B pool — **4.5%** — introduced by the `KodCode-V1` source,
which went in third and last during the code build. An earlier single-sample-per-shard
probe suggested the problem was 57.7% of the pool; that measurement was wrong (five
samples per shard is enough to see the difference), and it is recorded here rather than
quietly deleted because the first number nearly triggered a much larger intervention
than the evidence supported.

**2. A distribution mismatch at the evaluation.** The code held-out set is
competitive-programming style — stdin parsing, `map(int, input().split())`,
`print("Yes")` — while the newest training shards are repository snapshots with almost
no stdin idioms. Measured on decoded documents: the held-out shards are **80–100%
stdin-style**, the noisy train shards are **0%**. So part of the code number reflects the
model being asked for a style the newest data does not teach, which is a data-selection
problem rather than a model or mixture problem.

**Root cause found (2026-11-17).** This is not a hash-sequence problem. The bucket is
92% reasoning prose because the builder's field-priority list put `output`/`solution`
ahead of `problem`/`input`/`instruction`, so on every source that stores a task and its
response separately the *task was discarded*. On `nvidia/OpenCodeReasoning` one row holds
`input` (882 chars, the problem), `output` (9,617 chars, the CoT) and `solution` (360
chars, the code); `output` won, so the bucket trained on mid-thought continuations with
no problem and no code. Measured: only **1.6%** of the bucket's characters are inside
fenced code, and **60.9%** of its 2,048-token windows contain no document start at all —
i.e. no instruction, just continuation.

A full audit of all eight buckets, two independent model reviews, and a corrected metric
for this class of defect are in `CORPUS-AUDIT.md` in the data repo (`code/CORPUS-AUDIT.md`).
The short version, as a fraction of windows that contain no task at all:

| bucket | windows with no document start |
|---|---:|
| science, arithmetic | 0.0% |
| general | 17.1% |
| convo | 24.4% |
| indic | 44.3% |
| code | **60.9%** |
| math | **62.8%** |
| long | **98.7%** |

`code` and `math` share the defect; `long` is mismatched to a 2,048-token window; `indic`
is English text about India rather than Indic-language text.

**The fix is written and tested but not applied.** `build_fixed.py` names the task and
response fields per source instead of searching a priority list, caps document length so
one pair fits inside one window, and asserts the pairing before writing. Measured on the
capped rebuild: **0.0% mid-document windows** (from 63.4% for code, 62.9% for math, 46.7%
for indic), median document 1,006 / 1,793 / 1,793 tokens, ending in real code and real
boxed answers. `swap.py` installs clean shards ahead of each bucket's read head, moves the
old pool to `_old/` rather than deleting it, refuses to touch any `heldout_*.bin`
(`fixed_windows` derives starts from `arr.size`, so changing a heldout size silently
invalidates the whole eval curve), and asserts shard counts and token ids afterwards.

### Code ability, measured rather than inferred

Loss on code text is not program synthesis, so a probe was built that **executes** what
the model writes. It takes CodeContests problems with public tests, generates a solution
greedily (one token per forward pass, no sampling, so checkpoints stay comparable), runs
it under a 5-second timeout, and requires every public test to pass.

**Result at step ~102,400 (6.71B tokens): 0 of 30 solved.** One problem re-run with a
900-token budget produced a complete 1,389-character program — not a truncated one — and
also failed.

What the model writes is structured and plausible, and fails at the algorithm:

```python
import sys

def main():
    T = int(sys.stdin.readline())
    for _ in range(T):
        R = int(sys.stdin.readline())
        coords = []
        for _ in range(3):
            x, y = map(int, sys.stdin.readline().split())
            coords.append((x, y))
```

Correct stdin parsing, correct loop structure, correct types — then no working algorithm,
and occasionally open derailment (one sample nested `for a in range(0, 50000)` inside
another to compute a gcd).

This reframes the regression honestly. The rising code loss is **not** evidence of lost
ability, because there is no measurable ability on this benchmark to lose: a
139M-active-parameter model scoring pass@1 ≈ 0 on CodeContests is the expected result, and
greedy decoding with no sample filtering gives it the worst possible shot. The instrument
is too insensitive at this scale to track progress, and the correct conclusion is that
**the mixture should not be changed on the strength of it** — two changes have already
been made to code, and v7 puts it at 1.29×, at parity with every other bucket.

The same limitation applies to the loss curve itself: it can rank checkpoints, but it
cannot currently demonstrate that any of them can write a working program.

### Code turned the corner at 6.80B

The 6.60B → 6.80B paired comparison is the first point where code stops rising:

| Domain | 6.60B | 6.80B | Δ | t | Verdict |
|---|---:|---:|---:|---:|---|
| math | 0.9200 | 0.9077 | −0.0123 | −5.75 | better |
| long | 2.7572 | 2.7308 | −0.0265 | −2.65 | better |
| **code** | **1.6719** | **1.6670** | **−0.0050** | −1.21 | **flat (was +0.0201)** |
| general | 3.1813 | 3.1777 | −0.0036 | −1.02 | flat |
| science | 1.9314 | 1.9350 | +0.0037 | +1.43 | flat |
| convo | 1.6485 | 1.6630 | +0.0145 | +1.26 | flat |

No domain regressed. Code's three-milestone rise ends here, which — combined with the
probe result — supports the reading that it was a distribution/measurement effect rather
than a loss of capability. Against the 5.80B alarm point every domain is now better:
general −0.0678, science −0.0567, math −0.1598, code −0.0248, convo −0.0048,
long −0.1306.

Mean 1.9261 → 1.9219 and top-1 0.5888 → 0.5892, both still improving. The rate has
flattened (mean fell 0.0123 over the previous interval, 0.0043 over this one), which is
what a cosine decay is supposed to look like as the learning rate drops to 1.16e-04 —
not a plateau, since the paired tests still resolve movement in math and long.

A related trap, recorded because it nearly caused a wasted intervention: 10.3M tokens of
genuine competitive-programming data (13,328 CodeContests problems) were built and
appended as shards 212–216, and then found to be **unreachable**. Code sits at batch
position 0.072B of a 1.233B pool; at a 12.36% share it reaches only ~1.098B by the 15B
target, while those shards begin at 1.223B. Appending data to a bucket that is read
strictly forward from a saved position does not make that data train; it has to land
ahead of the read head. Nothing was disturbed to fix this — 10.3M tokens is 0.8% of the
pool and no measurement available here can detect its effect.

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

## Switched to 32K native context (2026-11-18, step 125,600)

Training now runs at **seq 32,768 × batch 2 × accum 1** instead of 2048 × 16 × 2. Both are
exactly **65,536 tokens/step**, which is load-bearing: `lr_at()` is indexed by *step* and
`total_steps = tokens // tok_per_step`, so any other batch size would silently rescale the
cosine schedule and move the 228,881-step target.

### Why

The `long` bucket is **2.01B tokens with median document 93,231 tokens**, and **97.1% of its
tokens live in documents ≥32k** (89.5% ≥64k, 61.0% ≥128k). At a 2048 window the model saw
2% of each document and never the rest — the largest unused resource in the corpus. This is
also the direct fix for the speaker/context errors observed in the greedy samples, where the
model answered as the wrong participant.

### Verified safe to change mid-run, not assumed

| claim | how it was checked |
|---|---|
| no data skipped or repeated | `take()` reads a flat `(batch*seq+1)` span, so 16×2048 and 2×32768 consume the **same tokens in the same order** |
| read positions survive | `batch_pos` is stored in **tokens**, not windows |
| no spurious bucket resets | `bucket_fp` hashes `(name, total, nfiles, samples)` only — **not** seq |
| curve stays comparable | the trainer invokes eval with **`--seq 2048`** unchanged |
| RoPE covers it | `MAX_POS = 131072`; the trainer asserts `seq > MAX_POS` and warns |

### Measured cost

| seq | batch | tokens/step | tok/s | peak GiB |
|---|---|---|---|---|
| 2,048 | 16 ×2 accum | 65,536 | 53,257 | 53.4 |
| **32,768** | **2** | **65,536** | **23,005 live** (31,451 isolated) | **18.5** |
| 16,384 | 4 | 65,536 | 27,875 | 12.6 |
| 32,768 | 6 | 196,608 | 17,618 | 33.2 |

Memory *fell* from 53.4 to 18.5 GiB, so the card is no longer the constraint. `--grad-checkpoint`
is required: 37.8 GiB without it at 32k versus 11.3 GiB with.

Note the curve is inverted from expectation — larger batches are **slower per token** (B=6 →
17.6k vs B=2 → 31.5k) because the local-layer `chunk_loop` dominates and extra rows add
memory pressure without parallelism. The 23,005 tok/s live figure is below the 31,451
isolated probe because three `tok2fix` rebuilds and the shard pusher share the 20 vCPUs.

Two things this does **not** fix: the fast flex-attention path is still unusable (uncompiled
it asks for a 40 GiB scores matrix at 32k; compiled it throws an `InductorError` in Triton's
sparse kernel), so 18 of 24 local layers remain on the slow path. And the convo speaker bug is
a **data** defect — 22.7% of `User:`/`Assistant:` markers are glued to the previous word
(`documents.User:`) — which no context length repairs.

## Run resumed again (2026-11-18, step 122,071 → )

A third box, restored from the Hub in **324 s** with zero failures:

| item | verification |
|---|---|
| Corpus | 1,217 shards, **11.730B tokens** (general 4.224B, science 2.161B, long 2.045B, code 1.241B, math 0.953B, convo 0.603B, arithmetic 0.362B, indic 0.141B) |
| Tokenizer | sha256 `e096c2ef…c690b4` — exact match |
| Checkpoint | **step 122,071**, 4,500,277,543 bytes |
| Health | `healthcheck.py` → `"ok": true` |

The previous box had pushed a checkpoint at **122,071** before it was reaped, so the restore
recovered 7,071 steps (0.46B tokens) that would otherwise have been lost — the 200M-token
backup cadence earning its keep. Training resumed at ~53,400 tok/s.

Two things were learned about recovery itself and are now fixed in the repository rather
than patched per box:

- `launch.sh` and `watchdog.sh` did not survive the box loss, because they lived only on the
  box. Both are now stored in `code/` alongside `restore3.py`, so a fresh box restores its
  own launcher.
- the clean `tok2fix` rebuilds also did not survive — only the scripts did. Rebuilds must be
  pushed as they are produced, or they cost the same 45 minutes again on every box.

## Run resumed (2026-11-18, step 114,400 → )

The box was replaced and the run is live again. Everything was restored from the Hub and
verified rather than assumed:

| item | verification |
|---|---|
| Corpus | 1,217 shards, **11.730B tokens**, 23.46 GB — byte sizes checked against the Hub, 0 failures |
| Tokenizer | sha256 `e096c2ef…c690b4`, 32,768 vocab — **exact** match to the laptop's original |
| Checkpoint | `train_state.pt` **4,500,277,543 bytes**, loads with 363 tensors, step 114,400 |
| Relaunch config | read from the checkpoint's own recorded `args`, not from memory |
| Backups | **verified by pushing**: 4.50 GB state + 750 MB weights in 26.4 s, `latest.json` advanced |

One flag needed deriving: the checkpoint records everything except `accum`, and 7,497,318,400
tokens ÷ 114,400 steps = exactly 65,536/step = 2048 × 16 × **2**. That also reproduces the
run's target, 15e9 / 65,536 = 228,881 steps.

Two silent-failure bugs were found and fixed on the way, both of the same shape — a line
that appears to work and does nothing:

- the data restore set a base URL in the parent shell and used it inside an `xargs sh -c`
  subshell, where it expanded to empty; 1,217 iterations created 1,217 empty directories and
  downloaded nothing. Counting directories would have said "done". Counting files said 0.
- `backup.py` resolves the token from `$HOME`, which on this box is `/home/marimo` and not
  `/root`. The token was installed in `/root` first and `whoami` still failed.

Watchdog, 30-minute backups and `healthcheck.py` are all running; `healthcheck` reports
`"ok": true`. Rebuilds of `code`, `math` and `indic` with task–response pairing are running
alongside the trainer.

## Run interruption (2026-11-17, step 114,400)

The training box was terminated by the host (HTTP 410 on all six known box URLs), not by a
fault in the run. State at that moment:

| item | status |
|---|---|
| Checkpoint | **step 114,400, resumable**, `train_state.pt` 4.5 GB on `XtrCom/xt-luna-ckpt-public`; `latest.json` points at it |
| Weights | `weights/last/model.safetensors` present and loadable (362 tensors; tied embedding stored once) |
| Tokenized corpus | **fully intact** — 1,217 shards, 11.73B tokens, all 8 buckets plus heldout |
| `arithmetic` heldout | **recovered** — 3 shards, 12.00M tokens (the bucket previously had none) |
| Progress | 114,400 / 228,881 steps = 7.50B of 15B tokens |
| Docs and scripts | all rebuild, audit and verification scripts backed up to `code/` in the data repo |

Nothing is lost. Resuming needs one GPU box: restore the data repo, download
`ckpt/step_114400/train_state.pt`, and launch with `--mix v7`. The rebuild described above
can then be applied before restarting, since a restart re-globs the data directory anyway.

A note on why the corpus mattered more than the loss curve here: at 7.5B tokens the model
predicts real code to a mean of **0.713** nats (median ~0.006) inside the same windows
where the aggregate reads 1.667, because 92% of those tokens are reasoning prose. The
capability was always better than the headline number, and the data defect was in the task
pairing rather than in the mixture weights — which is why two earlier mixture changes
failed to move it.

## Corpus

Training on a byte-level BPE vocabulary of 32,768. The corpus holds **11.652B measured
tokens**, sized from actual dataset capacity rather than dataset cards, against a run
target of 15B tokens — so every bucket repeats once, at 1.28–1.29× each.

Current token counts per bucket, as shards on disk:

| Bucket | Tokens | Source |
|---|---:|---|
| general | 4.208B | fineweb-edu (sample-10BT, sample-100BT) |
| long | 2.025B | long-document web text |
| science | 2.153B | cosmopedia-v2, finemath-4plus |
| code | 1.223B | OpenCodeReasoning, codeforces-cots, KodCode-V1, smollm python-edu (+10.3M CodeContests, unreachable — see above) |
| math | 0.952B | OpenMathReasoning (`cot`, `tir`), OpenR1-Math-220k, NuminaMath-CoT, MathInstruct |
| convo | 600M | WildChat and multi-turn dialogue |
| arithmetic | 350M | synthesised, 3.8M problems |
| indic | 140M | English Wikipedia, India-related articles |

Total **11.652B measured tokens**, up from 8.35B, after the code/math/general/science
builds described above. Growth in a bucket is append-only: new shards are numbered above
the existing ones because the loader globs and sorts `train_*.bin`, and a lower-numbered
shard would interleave new text into the middle of the stream and rewind every bucket
position.

One rule that is easy to get wrong: **only `train_*.bin` may grow.** The evaluator reads
`heldout_*.bin` and draws window starts from `rng.integers(0, len - seq)`, so enlarging a
heldout array would move every window and silently break comparability with earlier rows.

### Mixture, retargeted to 15B

The run was originally planned to 20B tokens and has been **retargeted to 15B** to reduce
plateau risk. That changes the cosine horizon: total steps go 305,175 → 228,881, so the
learning rate at step 99,000 steps **down** from 1.424e-04 to 1.173e-04. The direction
matters — a one-off drop cannot cause the divergence a raise could. The prediction was
checked against the trainer's own logged value (1.42e-04) before the restart and matched
to three significant figures.

Current shares, solved for an equal **final** epoch count rather than equal shares:

| Bucket | Share | Final epochs |
|---|---:|---:|
| general | 29.34% | 1.289× |
| long | 25.50% | 1.283× |
| science | 14.58% | 1.289× |
| code | 12.36% | 1.286× |
| math | 8.54% | 1.287× |
| convo | 4.53% | 1.288× |
| arithmetic | 3.18% | 1.287× |
| indic | 1.96% | 1.282× |

No bucket exceeds 1.3 passes to 15B.

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
| Real-text training, 375M core | **Running, 6.8B of a 15B target** (retargeted from 20B) |
| Held-out evaluation from 0.03B to 6.80B | Complete, table above. 5.80B onward is a paired 48-window series |
| Long-context recall | **Unmeasured and currently blocked.** Sequence length is 2048, and `WINDOW=2048` means every layer has taken the full-attention branch: the sliding window has never been active and the global layers have never seen a position beyond 2047. Corpus supply is also thin (12.6% of general tokens live in documents ≥8K). |
| Laptop inference | **Working.** A verified CPU runner ships with the public checkpoint; measured 19–23 tok/s |
| Instruction following / reasoning | Partially measured: 20-turn chat probe, refusal rate 0.40, turn-1 relevance 0.53 falling to 0.38 on later turns |

## Operational findings

Infrastructure lessons from the 5.2B → 6.5B stretch, each costing either hours or disk.
They are recorded because every one of them failed *silently*.

- **Deleting files from a Hub repo does not free storage.** `delete_folder` removes tree
  entries and `super_squash_history` rewrites commits, but neither purges LFS blobs — the
  objects stay downloadable and stay billed. Only deleting the repo reclaims them.
  Measured on this account: a repo showing 15.1 GB of files was billed **155.35 GB**.
- **That leak had a rate.** Each checkpoint push uploads a fresh 4.5 GB `train_state.pt`
  and prunes the previous one, orphaning a 4.5 GB blob per push. At `--ckpt-every 200`
  (~60 s) the backup interval was 240 s, i.e. 15 pushes/hour = **67.5 GB/hour ≈ 1.3 TB**
  over the remaining run. Pushing every 30 min instead makes it ~9 GB/hour. Verified: the
  billed figure held flat at 155.35 GB across 87 s where the old rate added ~1.6 GB.
  The durable fix is to stop shipping the full optimizer state every push.
- **`usedStorage` is not returned by `HfApi.model_info(expand=[...])`** on the training
  box — it silently returns 0. Query `https://huggingface.co/api/models/<repo>` with a
  bearer token instead.
- **The evaluator's window set is frozen by design.** `fixed_windows` seeds from
  `crc32(domain)` and draws from `rng.integers(0, len - seq)`. The determinism is what
  makes milestones comparable, but it also means growing a heldout array shifts every
  window. `len` is part of the seed inputs, so this is not a theoretical hazard.
- **`build_arithmetic.py` has no resume path.** It restarts at shard 0 and overwrites.
  Extending that bucket by re-running it destroys the existing shards; change the mixture
  share instead, or add a resume mode first.
- **Gated datasets fail late.** `bigcode/the-stack-dedup`, `starcoderdata` and
  `the-stack-v2-dedup` are all `gated=auto` and need manual approval, so a builder that
  lists them as sources produces nothing on this account. `codeparrot/github-code` and
  `BAAI/TACO` fail differently: "Dataset scripts are no longer supported."
- **The box had no supervisor.** `launch.sh` was `exec`'d from a bare `nohup setsid`, so
  any crash stopped training until a human noticed — which happened once. A `watchdog.sh`
  now relaunches on death, caps at 6 tries/hour, and writes a halt file rather than
  looping on a deterministic crash. It also runs a health probe each cycle, because a
  process that is alive is not necessarily healthy: a hang inside a CUDA call leaves the
  pid present and the log frozen.

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
