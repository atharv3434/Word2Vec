"""Generate a small synthetic corpus with deliberately structured semantics,
so a word embedding model has a fair chance of recovering meaningful
structure (semantic clusters, and a "royal - male + female" style analogy)
from a tiny amount of text.

This project ships with a pre-generated corpus already in place
(data/corpus.txt, data/word_categories.json), so you don't need to run this
to try the project out. Run it again for a different size or seed.

Design: words are assigned to categories along two independent axes
(royal/common/animal "domain", and male/female "gender" for people words).
Template sentences are built so that words sharing a domain share many
contexts (e.g. all royals "rule", "wear a crown", "live in a castle"), and
words sharing a gender share other contexts (e.g. all male words pair with
"he", all female words with "she"). This mirrors the structure that made
the classic word2vec "king - man + woman = queen" result possible.

Usage:
    python data/generate_corpus.py [--n-sentences 4000] [--seed 42]
    
"""

import argparse
import json
import os

import numpy as np

WORDS = {
    # word: (domain, gender)   gender is None where not applicable
    "king": ("royal", "male"),
    "queen": ("royal", "female"),
    "prince": ("royal", "male"),
    "princess": ("royal", "female"),
    "man": ("common", "male"),
    "woman": ("common", "female"),
    "boy": ("common", "male"),
    "girl": ("common", "female"),
    "lion": ("animal", "male"),
    "lioness": ("animal", "female"),
    "dog": ("animal", "male"),
    "cat": ("animal", "female"),
}

DOMAIN_TEMPLATES = {
    "royal": [
        "the {w} rules the kingdom",
        "the {w} lives in the castle",
        "the {w} wears a golden crown",
        "the {w} sits on the throne",
        "the {w} commands the royal guard",
    ],
    "common": [
        "the {w} works in the field",
        "the {w} lives in the village",
        "the {w} walks to the market",
        "the {w} carries water from the well",
        "the {w} tends the small farm",
    ],
    "animal": [
        "the {w} runs through the forest",
        "the {w} sleeps near the fire",
        "the {w} hunts in the tall grass",
        "the {w} rests under the old tree",
        "the {w} watches the quiet valley",
    ],
}

GENDER_TEMPLATES = {
    "male": [
        "he is the {w}",
        "the {w} is strong and he knows it",
    ],
    "female": [
        "she is the {w}",
        "the {w} is graceful and she knows it",
    ],
}


def build_sentences(rng, n_sentences):
    words = list(WORDS.keys())
    sentences = []
    for _ in range(n_sentences):
        w = words[rng.integers(0, len(words))]
        domain, gender = WORDS[w]

        use_gender_template = gender is not None and rng.random() < 0.35
        if use_gender_template:
            template = GENDER_TEMPLATES[gender][rng.integers(0, len(GENDER_TEMPLATES[gender]))]
        else:
            template = DOMAIN_TEMPLATES[domain][rng.integers(0, len(DOMAIN_TEMPLATES[domain]))]

        sentences.append(template.format(w=w))
    return sentences


def main():
    parser = argparse.ArgumentParser(description="Generate the synthetic word2vec training corpus.")
    parser.add_argument("--n-sentences", type=int, default=4000, help="Number of sentences to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sentences = build_sentences(rng, args.n_sentences)

    os.makedirs(args.out_dir, exist_ok=True)
    corpus_path = os.path.join(args.out_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    categories_path = os.path.join(args.out_dir, "word_categories.json")
    with open(categories_path, "w", encoding="utf-8") as f:
        json.dump(
            {w: {"domain": d, "gender": g} for w, (d, g) in WORDS.items()},
            f, indent=2,
        )

    print(f"Wrote {len(sentences)} sentences to {corpus_path}")
    print(f"Word category labels saved to {categories_path}")


if __name__ == "__main__":
    main()
