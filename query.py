"""Look up the nearest neighbors of a word by cosine similarity in the
trained embedding space.

Usage:
    python src/query.py --word king --top-k 5
"""

import argparse
import os
import sys

import joblib
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils import load_config


def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No embeddings found at '{path}'. Run `python src/train.py` first.")
    return joblib.load(path)


def nearest_neighbors(word, bundle, top_k=5):
    word2idx = bundle["word2idx"]
    idx2word = bundle["idx2word"]
    embeddings = bundle["embeddings"]

    if word not in word2idx:
        raise ValueError(f"'{word}' is not in the vocabulary.")

    vec = embeddings[word2idx[word]]
    norms = np.linalg.norm(embeddings, axis=1)
    vec_norm = np.linalg.norm(vec)
    sims = (embeddings @ vec) / (norms * vec_norm + 1e-10)

    order = np.argsort(sims)[::-1]
    results = []
    for idx in order:
        candidate = idx2word[idx]
        if candidate == word:
            continue
        results.append((candidate, float(sims[idx])))
        if len(results) == top_k:
            break
    return results


def main():
    parser = argparse.ArgumentParser(description="Find nearest neighbors of a word.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--word", required=True, help="Query word")
    parser.add_argument("--top-k", type=int, default=5, help="Number of neighbors to show")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = load_embeddings(config["embeddings_path"])

    results = nearest_neighbors(args.word, bundle, args.top_k)
    print(f"Nearest neighbors of '{args.word}':")
    for word, sim in results:
        print(f"  {word:12s} cosine_sim={sim:.3f}")


if __name__ == "__main__":
    main()
