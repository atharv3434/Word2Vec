# Word2Vec (SGNS) — Evaluation Report

Skip-gram with Negative Sampling, implemented from scratch in numpy (see `src/model.py`), trained on a small synthetic corpus with deliberately structured semantics (see `data/generate_corpus.py`).

Vocabulary size: 65

## Nearest neighbors

- **king** → prince (0.93), princess (0.78), queen (0.72), throne (0.57), man (0.55)
- **queen** → princess (0.94), prince (0.79), king (0.72), girl (0.51), throne (0.50)
- **lion** → dog (0.88), lioness (0.81), cat (0.81), grass (0.64), tree (0.60)
- **dog** → lion (0.88), lioness (0.84), cat (0.80), tree (0.56), grass (0.50)

## Category separation

Average cosine similarity between words sharing a category label vs. words that don't:

| Axis | Same-category sim | Different-category sim |
|---|---|---|
| Domain (royal/common/animal) | 0.833 | 0.357 |
| Gender (male/female) | 0.526 | 0.454 |

## Analogy test

**king - man + woman = ?** (expected: queen)

- princess: 0.895
- queen: 0.858 **← expected**
- prince: 0.775
- throne: 0.543
- guard: 0.488

## Embedding space (PCA)

![Embedding PCA](figures/embedding_pca.png)

## Training loss

![Loss curve](figures/loss_curve.png)