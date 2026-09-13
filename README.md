# Word2Vec (Skip-Gram with Negative Sampling), From Scratch

A from-scratch numpy implementation of the Word2Vec skip-gram model with
negative sampling, replicating the core method of:

- Mikolov, Chen, Corrado, Dean (2013). *Efficient Estimation of Word
  Representations in Vector Space.* ICLR Workshop. (introduces the
  skip-gram and CBOW architectures)
- Mikolov, Sutskever, Chen, Corrado, Dean (2013). *Distributed
  Representations of Words and Phrases and their Compositionality.*
  NeurIPS. (introduces negative sampling and frequent-word subsampling,
  and the famous `king - man + woman ≈ queen` analogy result)

No deep learning framework is used — the forward pass, loss, and gradients
are all derived and implemented by hand with numpy, and checked against a
numerical (finite-difference) gradient computation to confirm correctness
before training (see `tests/test_gradients.py`).

## Why a synthetic corpus?

Word embeddings need a lot of repeated, structured co-occurrence to learn
meaningful geometry, and real-world text corpora aren't something this
project can download. Instead, `data/generate_corpus.py` builds a small
corpus from templates designed so that:
- words sharing a **domain** (royal / common / animal) share many contexts
  ("the king **rules the kingdom**", "the queen **rules the kingdom**")
- words sharing a **gender** (male / female) share other contexts
  ("**he** is the king", "**he** is the man")

This is the same kind of structure that makes the classic word2vec
analogy demos work, just deliberately engineered into a tiny corpus so the
result is reproducible without a huge dataset or long training time.

## Project structure

```
word2vec-from-scratch/
├── config.yaml                   # all hyperparameters
├── requirements.txt
├── data/
│   ├── generate_corpus.py        # builds the structured synthetic corpus
│   ├── corpus.txt                # pre-generated sample corpus (4000 sentences)
│   └── word_categories.json      # domain/gender labels, used only for evaluation
├── src/
│   ├── utils.py                  # config/JSON loading
│   ├── preprocessing.py          # tokenizing, vocab, subsampling, pair generation
│   ├── model.py                  # SGNS model: forward pass + hand-derived gradients
│   ├── train.py                  # training loop
│   ├── query.py                  # nearest-neighbor lookup for a word
│   └── evaluate.py               # category separation, analogy test, PCA plot
├── tests/
│   └── test_gradients.py         # numerical gradient check
├── models/
│   └── embeddings.joblib          # trained word vectors + vocab (after training)
├── output/
│   ├── figures/                   # loss curve, PCA scatter plot
│   └── evaluation_report.md       # full write-up of results
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. (Optional) Verify the gradients

```bash
python tests/test_gradients.py
```

Confirms the hand-derived backpropagation gradients in `model.py` match a
finite-difference approximation (max error ~1e-10 in testing — i.e. correct
up to floating point precision).

## 2. Train

```bash
python src/train.py
```

Builds the vocabulary, generates skip-gram training pairs (dynamic window
size + frequent-word subsampling, both as in the original paper), and
trains for 40 epochs with a linearly-decaying learning rate. Saves the
learned embeddings to `models/embeddings.joblib` and a loss curve to
`output/figures/loss_curve.png`. Training on the bundled corpus (65-word
vocabulary, ~7,000 pairs) takes well under a minute on CPU.

## 3. Query

```bash
python src/query.py --word king --top-k 5
```

```
Nearest neighbors of 'king':
  prince       cosine_sim=0.929
  princess     cosine_sim=0.778
  queen        cosine_sim=0.715
  throne       cosine_sim=0.572
  man          cosine_sim=0.553
```

## 4. Evaluate

```bash
python src/evaluate.py
```

Produces:
- **Nearest neighbors** for several query words
- **Category separation**: same-category words average 0.833 cosine
  similarity vs. 0.357 for different-category words (domain axis) —
  confirming the embedding space actually clusters by meaning, not just
  by chance.
- **Analogy test**: `king - man + woman = ?` returns **princess** (0.895)
  narrowly ahead of **queen** (0.858). This is an honest, informative
  near-miss rather than a perfect hit: the synthetic corpus never
  distinguishes "queen" from "princess" by age, so the model correctly
  learns "royal + female" but has no signal to separate those two further
  — a good illustration of how an embedding model can only be as precise
  as the distinctions present in its training data.
- **PCA visualization** (`output/figures/embedding_pca.png`): the 12
  labeled words separate into three tight, clearly-defined clusters
  (royal / common / animal) in just two principal components, capturing
  ~74% of the variance.

Full write-up with all figures: `output/evaluation_report.md`.

## How the model works (short version)

For every observed (center word, nearby context word) pair in the corpus,
the model is trained as binary logistic regression: predict "yes, these
co-occur" for the true pair, and "no" for `k` randomly sampled negative
(center, random word) pairs. Two embedding matrices are learned — `W_in`
(used as the final word vectors) and `W_out` (a separate context-word
representation, discarded after training). See the derivation and
implementation notes at the top of `src/model.py`.

## Extending this project

- **Real text**: point `corpus_path` at any plain-text file (one sentence
  per line works best) to train on real data instead of the synthetic
  corpus — everything else stays the same.
- **CBOW**: add a variant of `model.py` that predicts the center word from
  the (averaged) context words instead of the reverse — the other
  architecture from the original paper.
- **Larger vocabulary**: for a much larger vocabulary, replace
  `np.random.choice` negative sampling with an alias-method sampler, and
  consider hierarchical softmax as an alternative to negative sampling
  (also proposed in the original papers).
