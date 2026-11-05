# XT-Luna-Zero

**Architecture of a 194M-Parameter Sparse Decoder**
Technical report draft · November 6, 2026 · XtrComSu

📄 **[Read the paper (PDF)](paper/XT-Luna-Zero-Architecture.pdf)**

## Abstract

XT-Luna-Zero is a decoder-only language model built to train from scratch on a
single GPU. Sixteen blocks pair causal attention with a feed-forward mixture of
experts: one shared expert and four routed experts in every block. Top-1
routing puts the model at a reported 194,048,640 stored parameters and
99,676,800 active parameters per token, counted the way this project counts
them.

The report follows the forward path, works through the sparse parameter
budget, and separates the numerical checks that were actually run from the
work that was not. At the last reported milestone there was no real-text
training checkpoint. Language quality, reasoning, and effective long-context
recall are still unmeasured.

## Contents of the report

| § | Topic |
|---|---|
| 1 | Purpose and scope |
| 2 | Decoder architecture: 16 blocks, width 640, GQA head layout, tied 32K embeddings |
| 3 | Attention and expert routing: local and global causal schedule, shared and routed SwiGLU experts, parameter accounting |
| 4 | Reported engineering evidence: attention reference checks, BF16 integration, data and checkpoints |
| 5 | Pretraining and evaluation protocol: 60/30/10 corpus allocation, 0.5B-token pilot, held-out metrics |
| 6 | Scope of future experiments |

## Key figures

| Quantity | Parameters |
|---|---:|
| Stored model | 194,048,640 |
| Routed weights skipped per token | 94,371,840 |
| Active per token (project convention) | 99,676,800 |

That active count describes selection for one token. A batch can wake every
routed expert at once, and the stored weights still have to fit in memory
either way. It is not a measured FLOP count, and it says nothing about
wall-clock throughput.

## Status

What the paper does contain is an engineering design and a parameter
accounting. The numerical checks cover attention output against an explicit
causal-window reference in FP32 at lengths from 1,024 to 6,144. A production
BF16 training step is still unverified. The only checkpoint reported so far
came from a repeated numeric pattern built for a smoke test.

So any claim about language modeling quality, or about a particular
architectural choice helping, needs held-out results from real-text runs
before it means anything.

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
