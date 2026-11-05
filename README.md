# XT-Luna-Zero

**XT-Luna-Zero: Architecture of a 194M-Parameter Sparse Decoder**
Technical report draft · November 6, 2026 · XtrComSu

📄 **[Read the paper (PDF)](paper/XT-Luna-Zero-Architecture.pdf)**

---

## Abstract

XT-Luna-Zero is a decoder-only language model designed for training from
scratch on a single GPU. Its 16 blocks combine causal attention with a
feed-forward mixture of experts containing one shared expert and four routed
experts per block. Top-1 routing yields a reported 194,048,640 stored
parameters and 99,676,800 active parameters per token under the project's
counting convention. We describe the forward path, account for the sparse
parameter budget, and distinguish the reported numerical checks from
unresolved training and evaluation work. At the latest reported milestone,
the project had no real-text training checkpoint. Language quality, reasoning
performance, and effective long-context recall remain unmeasured.

## Contents of the report

| § | Topic |
|---|---|
| 1 | Purpose and scope |
| 2 | Decoder architecture — 16 blocks, width 640, GQA head layout, tied 32K embeddings |
| 3 | Attention and expert routing — local/global causal schedule, shared + routed SwiGLU experts, parameter accounting |
| 4 | Reported engineering evidence — attention reference checks, BF16 integration, data and checkpoints |
| 5 | Pretraining and evaluation protocol — 60/30/10 corpus allocation, 0.5B-token pilot, held-out metrics |
| 6 | Scope of future experiments |

## Key figures

| Quantity | Parameters |
|---|---:|
| Stored model | 194,048,640 |
| Routed weights skipped per token | 94,371,840 |
| Active per token (project convention) | 99,676,800 |

The active count is a per-token selection figure. A batch can activate every
routed expert, and stored weights still count against memory. It is not a
measured FLOP or wall-clock number.

## Status

The paper reports an engineering design and a parameter accounting. Numerical
checks cover attention output against an explicit causal-window reference in
FP32 at lengths 1,024–6,144. A production BF16 training step remains
unverified, and the only reported checkpoint was a repeated-numeric-pattern
smoke test. Claims about language modeling quality or architectural
improvements require held-out results from real-text runs.

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

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## Lineage

Architectural ideas are drawn from prior work: [Kimi Linear](https://arxiv.org/abs/2510.26692)
(Kimi Delta Attention), [Kimi K2](https://arxiv.org/abs/2507.20534)
(MuonClip/QK-clip), [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
(auxiliary-loss-free MoE balancing), [DeepSeekMoE](https://arxiv.org/abs/2401.06066)
(shared expert isolation), and [SmolLM2](https://arxiv.org/abs/2502.02737)
(small-model data curation).

## License

© 2026 XtrComSu. This report is licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](LICENSE).

You are free to share and adapt the material for any purpose, including
commercially, provided you give appropriate credit, provide a link to the
license, and indicate whether changes were made.

Third-party works cited in the report's reference list remain under their
original copyright and are not covered by this license.
