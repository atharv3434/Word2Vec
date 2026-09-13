"""Skip-gram with Negative Sampling (SGNS), implemented from scratch with
plain numpy — forward pass, loss, and gradients are all derived and coded
by hand (no autograd), following Mikolov et al. (2013).

Model: two embedding matrices, W_in (the "center word" / target
embeddings, which become the final word vectors) and W_out (the "context
word" embeddings, used only during training). For a center word c and
context word o, the model treats this as binary logistic regression:
predict 1 for true (c, o) pairs seen in the corpus, and 0 for k sampled
negative pairs (c, negative) where `negative` is drawn from the noise
distribution rather than truly being a neighbor of c.

Derivation of the gradients used below:
    score   = W_in[c] . W_out[o]
    pred    = sigmoid(score)
    loss    = -[label * log(pred) + (1-label) * log(1-pred)]
    d(loss)/d(score) = pred - label

which is the standard logistic-regression gradient. From there:
    d(loss)/d(W_out[o]) = (pred - label) * W_in[c]
    d(loss)/d(W_in[c])  = (pred - label) * W_out[o]

Because the same word can appear more than once as a center/context/
negative word within one batch, updates are accumulated with `np.add.at`
rather than plain fancy-index assignment, which would silently drop all
but the last update to a repeated index.
"""

import numpy as np


def sigmoid(x):
    # Numerically stable sigmoid.
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


class SkipGramNegativeSampling:
    def __init__(self, vocab_size, embedding_dim, random_state=42):
        rng = np.random.default_rng(random_state)
        bound = 0.5 / embedding_dim
        self.W_in = rng.uniform(-bound, bound, size=(vocab_size, embedding_dim))
        self.W_out = np.zeros((vocab_size, embedding_dim))
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def train_batch(self, center_ids, context_ids, neg_ids, lr):
        """One SGD step on a batch of positive pairs + their negative samples.

        center_ids: (batch,)
        context_ids: (batch,)          — the true, positive context word for each center
        neg_ids: (batch, k)            — k sampled negative words for each center
        Returns the average loss over the batch (for monitoring only).
        """
        v_c = self.W_in[center_ids]              # (batch, dim)
        v_o = self.W_out[context_ids]             # (batch, dim)
        v_neg = self.W_out[neg_ids]               # (batch, k, dim)

        pos_score = np.sum(v_c * v_o, axis=1)                       # (batch,)
        pos_pred = sigmoid(pos_score)
        neg_score = np.einsum("bd,bkd->bk", v_c, v_neg)              # (batch, k)
        neg_pred = sigmoid(neg_score)

        eps = 1e-10
        pos_loss = -np.log(pos_pred + eps)
        neg_loss = -np.log(1 - neg_pred + eps).sum(axis=1)
        avg_loss = float(np.mean(pos_loss + neg_loss))

        # Gradient coefficients: (pred - label), label=1 for positives, 0 for negatives.
        g_pos = pos_pred - 1.0          # (batch,)
        g_neg = neg_pred                # (batch, k)

        # --- Gradients w.r.t. W_out (context/negative embeddings) ---
        grad_context = g_pos[:, None] * v_c                          # (batch, dim)
        grad_neg = g_neg[:, :, None] * v_c[:, None, :]                # (batch, k, dim)

        np.add.at(self.W_out, context_ids, -lr * grad_context)
        np.add.at(self.W_out, neg_ids.reshape(-1), -lr * grad_neg.reshape(-1, self.embedding_dim))

        # --- Gradient w.r.t. W_in (center embeddings) ---
        grad_center = g_pos[:, None] * v_o + np.sum(g_neg[:, :, None] * v_neg, axis=1)  # (batch, dim)
        np.add.at(self.W_in, center_ids, -lr * grad_center)

        return avg_loss

    def word_vectors(self):
        """The final learned word embeddings (the W_in matrix)."""
        return self.W_in
