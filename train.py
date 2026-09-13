"""
Train Word2Vec (Skip-gram with Negative Sampling) on the corpus.

Usage:
    python src/train.py [--config config.yaml]

Loads the corpus, builds the vocabulary and training pairs, trains the
model batch-by-batch with a linearly-decaying learning rate (as in the
original word2vec implementation), and saves the resulting word vectors.
"""

import argparse
import os
import sys

import joblib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from utils import load_config
from preprocessing import (
    tokenize, build_vocab, build_noise_distribution,
    subsample_probabilities, build_training_pairs,
)
from model import SkipGramNegativeSampling


def main():
    parser = argparse.ArgumentParser(description="Train word2vec (SGNS) from scratch.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(config.get("random_state", 42))

    print(f"Reading corpus from {config['corpus_path']} ...")
    with open(config["corpus_path"], "r", encoding="utf-8") as f:
        corpus_text = f.read()
    sentences = tokenize(corpus_text)
    n_tokens = sum(len(s) for s in sentences)
    print(f"{len(sentences)} sentences, {n_tokens} tokens.")

    word2idx, idx2word, vocab_counts = build_vocab(sentences, min_count=config.get("min_count", 1))
    vocab_size = len(word2idx)
    print(f"Vocabulary size: {vocab_size}")

    noise_dist = build_noise_distribution(vocab_counts)
    keep_prob = subsample_probabilities(vocab_counts, config.get("subsample_threshold", 1e-3))

    print("Building skip-gram training pairs (dynamic window + subsampling)...")
    pairs = build_training_pairs(sentences, word2idx, config.get("max_window_size", 4), keep_prob, rng)
    print(f"{len(pairs)} training pairs generated.")

    model = SkipGramNegativeSampling(
        vocab_size=vocab_size,
        embedding_dim=config.get("embedding_dim", 30),
        random_state=config.get("random_state", 42),
    )

    epochs = config.get("epochs", 40)
    batch_size = config.get("batch_size", 64)
    initial_lr = config.get("initial_lr", 0.05)
    min_lr = config.get("min_lr", 0.0001)
    k = config.get("negative_samples", 5)

    n_pairs = len(pairs)
    n_batches_per_epoch = max(1, n_pairs // batch_size)
    total_steps = epochs * n_batches_per_epoch
    step = 0

    loss_history = []
    print(f"\nTraining for {epochs} epochs ({n_batches_per_epoch} batches/epoch)...")
    for epoch in range(epochs):
        rng.shuffle(pairs)
        epoch_losses = []

        for b in range(n_batches_per_epoch):
            batch = pairs[b * batch_size:(b + 1) * batch_size]
            if len(batch) == 0:
                continue
            center_ids = batch[:, 0]
            context_ids = batch[:, 1]
            neg_ids = rng.choice(vocab_size, size=(len(batch), k), p=noise_dist)

            lr = initial_lr - (initial_lr - min_lr) * (step / total_steps)
            lr = max(lr, min_lr)

            loss = model.train_batch(center_ids, context_ids, neg_ids, lr)
            epoch_losses.append(loss)
            step += 1

        avg_loss = float(np.mean(epoch_losses))
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  loss={avg_loss:.4f}  lr={lr:.4f}")

    os.makedirs(os.path.dirname(config["embeddings_path"]), exist_ok=True)
    joblib.dump({
        "embeddings": model.word_vectors(),
        "word2idx": word2idx,
        "idx2word": idx2word,
        "config": config,
    }, config["embeddings_path"])
    print(f"\nEmbeddings saved to {config['embeddings_path']}")

    os.makedirs(config["figures_dir"], exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, epochs + 1), loss_history, color="#3B6E8F")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average training loss")
    ax.set_title("SGNS training loss")
    fig.tight_layout()
    loss_plot_path = os.path.join(config["figures_dir"], "loss_curve.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)
    print(f"Loss curve saved to {loss_plot_path}")


if __name__ == "__main__":
    main()
