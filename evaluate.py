"""Evaluate the trained embeddings: nearest-neighbor examples, a
same-category vs. different-category separation metric, a 2D PCA
visualization, and the classic word2vec analogy test
(king - man + woman ~= queen).

Usage:
    python src/evaluate.py [--config config.yaml]
"""

import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from utils import load_config, load_json
from query import nearest_neighbors


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def category_separation(embeddings, word2idx, categories, axis):
    """Compare average cosine similarity between words that share a
    category label (on the given axis, e.g. "domain" or "gender") against
    words that don't. A well-structured embedding space should show
    same-category similarity notably higher than different-category
    similarity.
    """
    words = [w for w in categories if w in word2idx and categories[w].get(axis)]
    same, diff = [], []

    for i, w1 in enumerate(words):
        for w2 in words[i + 1:]:
            sim = cosine_sim(embeddings[word2idx[w1]], embeddings[word2idx[w2]])
            if categories[w1][axis] == categories[w2][axis]:
                same.append(sim)
            else:
                diff.append(sim)

    return {
        "same_category_mean_sim": round(float(np.mean(same)), 3) if same else None,
        "diff_category_mean_sim": round(float(np.mean(diff)), 3) if diff else None,
        "n_same_pairs": len(same),
        "n_diff_pairs": len(diff),
    }


def run_analogy(embeddings, word2idx, idx2word, a, b, c, top_k=5):
    """Classic word2vec vector-arithmetic analogy: a - b + c ~= ?
    e.g. king - man + woman ~= queen.
    """
    vec = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]
    norms = np.linalg.norm(embeddings, axis=1)
    sims = (embeddings @ vec) / (norms * np.linalg.norm(vec) + 1e-10)

    order = np.argsort(sims)[::-1]
    exclude = {a, b, c}
    results = []
    for idx in order:
        word = idx2word[idx]
        if word in exclude:
            continue
        results.append((word, float(sims[idx])))
        if len(results) == top_k:
            break
    return results


def plot_pca(embeddings, word2idx, categories, figures_dir):
    words = [w for w in categories if w in word2idx]
    vectors = np.array([embeddings[word2idx[w]] for w in words])
    domains = [categories[w]["domain"] for w in words]

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vectors)

    color_map = {"royal": "#8C2F39", "common": "#3B6E8F", "animal": "#4C7A3D"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for domain, color in color_map.items():
        mask = [d == domain for d in domains]
        ax.scatter(coords[mask, 0], coords[mask, 1], color=color, label=domain, s=60)

    for word, (x, y) in zip(words, coords):
        ax.annotate(word, (x, y), textcoords="offset points", xytext=(5, 3), fontsize=9)

    ax.set_title("Word embeddings, projected to 2D (PCA)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(figures_dir, "embedding_pca.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained word embeddings.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = joblib.load(config["embeddings_path"])
    embeddings, word2idx, idx2word = bundle["embeddings"], bundle["word2idx"], bundle["idx2word"]
    categories = load_json(config["categories_path"])

    print(f"Vocabulary size: {len(word2idx)}\n")

    print("--- Nearest neighbors ---")
    neighbor_results = {}
    for word in config.get("query_words", []):
        results = nearest_neighbors(word, bundle, config.get("n_neighbors", 5))
        neighbor_results[word] = results
        neighbor_str = ", ".join(f"{w} ({s:.2f})" for w, s in results)
        print(f"  {word:10s} -> {neighbor_str}")

    print("\n--- Category separation ---")
    domain_sep = category_separation(embeddings, word2idx, categories, "domain")
    gender_sep = category_separation(embeddings, word2idx, categories, "gender")
    print(f"  Domain axis: same-category sim={domain_sep['same_category_mean_sim']}, "
          f"diff-category sim={domain_sep['diff_category_mean_sim']}")
    print(f"  Gender axis: same-category sim={gender_sep['same_category_mean_sim']}, "
          f"diff-category sim={gender_sep['diff_category_mean_sim']}")

    print("\n--- Analogy test ---")
    analogy_cfg = config.get("analogy", {})
    analogy_results = None
    if analogy_cfg:
        a, b, c, expected = analogy_cfg["a"], analogy_cfg["b"], analogy_cfg["c"], analogy_cfg["expected"]
        analogy_results = run_analogy(embeddings, word2idx, idx2word, a, b, c)
        top_word = analogy_results[0][0]
        print(f"  {a} - {b} + {c} =")
        for word, sim in analogy_results:
            marker = "  <-- expected" if word == expected else ""
            print(f"    {word:12s} {sim:.3f}{marker}")
        print(f"  Result: {'PASS' if top_word == expected else 'did not exactly match'} "
              f"(top result: '{top_word}', expected: '{expected}')")

    print("\n--- PCA visualization ---")
    os.makedirs(config["figures_dir"], exist_ok=True)
    pca_path = plot_pca(embeddings, word2idx, categories, config["figures_dir"])
    print(f"  Saved to {pca_path}")

    # --- Write markdown report ---
    lines = ["# Word2Vec (SGNS) — Evaluation Report", ""]
    lines.append(
        "Skip-gram with Negative Sampling, implemented from scratch in numpy "
        "(see `src/model.py`), trained on a small synthetic corpus with "
        "deliberately structured semantics (see `data/generate_corpus.py`)."
    )
    lines.append("")
    lines.append(f"Vocabulary size: {len(word2idx)}")
    lines.append("")

    lines.append("## Nearest neighbors")
    lines.append("")
    for word, results in neighbor_results.items():
        neighbor_str = ", ".join(f"{w} ({s:.2f})" for w, s in results)
        lines.append(f"- **{word}** → {neighbor_str}")
    lines.append("")

    lines.append("## Category separation")
    lines.append("")
    lines.append("Average cosine similarity between words sharing a category label vs. words that don't:")
    lines.append("")
    lines.append("| Axis | Same-category sim | Different-category sim |")
    lines.append("|---|---|---|")
    lines.append(f"| Domain (royal/common/animal) | {domain_sep['same_category_mean_sim']} | {domain_sep['diff_category_mean_sim']} |")
    lines.append(f"| Gender (male/female) | {gender_sep['same_category_mean_sim']} | {gender_sep['diff_category_mean_sim']} |")
    lines.append("")

    if analogy_results:
        lines.append("## Analogy test")
        lines.append("")
        lines.append(f"**{a} - {b} + {c} = ?** (expected: {expected})")
        lines.append("")
        for word, sim in analogy_results:
            marker = " **← expected**" if word == expected else ""
            lines.append(f"- {word}: {sim:.3f}{marker}")
        lines.append("")

    lines.append("## Embedding space (PCA)")
    lines.append("")
    lines.append("![Embedding PCA](figures/embedding_pca.png)")
    lines.append("")
    lines.append("## Training loss")
    lines.append("")
    lines.append("![Loss curve](figures/loss_curve.png)")

    with open(config["report_path"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nFull report saved to {config['report_path']}")


if __name__ == "__main__":
    main()
