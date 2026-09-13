"""Corpus preprocessing: tokenization, vocabulary building, frequent-word
subsampling, the negative-sampling noise distribution, and skip-gram
training pair generation — following Mikolov et al. (2013).
"""

import numpy as np


def tokenize(corpus_text):
    """Lowercase, whitespace-split tokenization, one list of tokens per
    sentence (sentence boundaries are respected — context windows never
    cross them).
    """
    sentences = []
    for line in corpus_text.strip().split("\n"):
        tokens = line.strip().lower().split()
        if tokens:
            sentences.append(tokens)
    return sentences


def build_vocab(sentences, min_count=1):
    """Build word2idx/idx2word mappings and raw counts, keeping only words
    that appear at least `min_count` times.
    """
    counts = {}
    for sent in sentences:
        for tok in sent:
            counts[tok] = counts.get(tok, 0) + 1

    vocab_words = [w for w, c in counts.items() if c >= min_count]
    vocab_words.sort()  # deterministic ordering
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_counts = np.array([counts[w] for w in vocab_words], dtype=np.float64)

    return word2idx, idx2word, vocab_counts


def build_noise_distribution(vocab_counts, power=0.75):
    """The unigram^0.75 noise distribution used for negative sampling —
    raising counts to the 0.75 power (rather than using raw frequency)
    boosts the relative sampling probability of rare words, which Mikolov
    et al. found worked better empirically than sampling from the raw
    unigram distribution.
    """
    weighted = vocab_counts ** power
    return weighted / weighted.sum()


def subsample_probabilities(vocab_counts, threshold):
    """Compute the keep-probability for each vocabulary word, using the
    subsampling formula from Mikolov et al. (2013), section on frequent
    words: P(keep) = (sqrt(freq/t) + 1) * (t/freq), clipped to [0, 1].
    Frequent words like "the" are then randomly dropped from training,
    which speeds up training and improves the quality of rarer words'
    vectors.
    """
    total = vocab_counts.sum()
    freq = vocab_counts / total
    keep_prob = (np.sqrt(freq / threshold) + 1) * (threshold / freq)
    return np.clip(keep_prob, 0, 1)


def build_training_pairs(sentences, word2idx, max_window_size, keep_prob, rng):
    """Generate (center_idx, context_idx) skip-gram pairs.

    For each token, a window size is sampled uniformly from
    [1, max_window_size] independently (the "dynamic window" trick from
    the original word2vec implementation, which effectively weights
    closer context words more heavily on average). Subsampling is applied
    to decide whether each token participates at all, following Mikolov
    et al.'s frequent-word subsampling.
    """
    pairs = []
    for sent in sentences:
        idxs = [word2idx[w] for w in sent if w in word2idx]
        n = len(idxs)
        for i, center in enumerate(idxs):
            if rng.random() > keep_prob[center]:
                continue  # subsampled away

            window = rng.integers(1, max_window_size + 1)
            start, end = max(0, i - window), min(n, i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context = idxs[j]
                if rng.random() > keep_prob[context]:
                    continue
                pairs.append((center, context))

    return np.array(pairs, dtype=np.int64)
