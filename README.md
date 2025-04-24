# Differential-Attention Vision Transformer (Diff-ViT) on CIFAR-10

A course-project repo that upgrades the Vision Transformer by replacing vanilla self-attention with **Differential Attention**—a twin-query, noise-cancelling mechanism—and benchmarks it end-to-end on CIFAR-10.

---

## ⭐ Highlights
| ✔ | Component |
|---|-----------|
| **Differential Attention layer** (`DifferentialAttention`) with two Q/K streams and learnable λ-scaling |
| **ViT encoder refactor** (`TransformerEncoder`): pre-norm, Diff-Attn + MLP, residuals |
| PatchConv embedding, CLS token, **four positional-embedding options** (none / 1-D learned / 2-D learned / sinusoidal) |
| Training pipeline: AdamW + cosine-anneal LR, warm-up, tqdm bars |
| Experiment suites: patch-size, depth/width, augmentation, positional embeddings |
| Visual tools: per-layer CLS→patch attention maps & rollout heat-maps |

---

## 📈 Key Results (@ 50 epochs)

| Model | Patch | Params | Augmentation | Test Acc |
|-------|------:|-------:|--------------|---------:|
| Vanilla ViT | 4 × 4 | 14.2 M | flip + crop | **73.0 %** |
| Diff-ViT | 2 × 2 | 6.4 M | flip + crop | 66.7 % |
| **Diff-ViT** | 4 × 4 | 14.3 M | **flip + crop** | **79.7 %** |
| Diff-ViT | 8 × 8 | 25.3 M | flip + crop | 69.2 % |

**+6.7 pp over size-matched vanilla ViT** with < 1 % extra parameters.

More sweeps:

| Sweep | Best Acc |
|-------|---------:|
| Depth 12 / LR 2e-4 | 75.4 % |
| RandAugment (N = 2, M = 9) | 79.6 % |
| 1-D learned positional embedding (default) | **79.7 %** |

