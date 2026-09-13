"""
Numerical gradient check for the hand-derived SGNS gradients in model.py.

Since the training update in SkipGramNegativeSampling.train_batch() is
implemented by hand (no autograd), this test verifies it against a
finite-difference approximation of the true gradient — the standard way to
catch a sign error or mis-derived gradient before trusting hours of
training.

Usage:
    python tests/test_gradients.py

"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import sigmoid


def loss_fn(W_in, W_out, center, context, negs):
    """Recompute the SGNS loss for a single example, given explicit
    embedding matrices — used both directly and for finite differences.
    """
    v_c = W_in[center]
    v_o = W_out[context]
    v_neg = W_out[negs]

    pos_score = np.dot(v_c, v_o)
    pos_pred = sigmoid(pos_score)
    neg_scores = v_neg @ v_c
    neg_preds = sigmoid(neg_scores)

    eps = 1e-10
    loss = -np.log(pos_pred + eps) - np.sum(np.log(1 - neg_preds + eps))
    return loss


def analytic_gradients(W_in, W_out, center, context, negs):
    v_c = W_in[center]
    v_o = W_out[context]
    v_neg = W_out[negs]

    pos_pred = sigmoid(np.dot(v_c, v_o))
    neg_preds = sigmoid(v_neg @ v_c)

    g_pos = pos_pred - 1.0
    g_neg = neg_preds

    grad_context = g_pos * v_c
    grad_neg = g_neg[:, None] * v_c[None, :]
    grad_center = g_pos * v_o + np.sum(g_neg[:, None] * v_neg, axis=0)

    return grad_center, grad_context, grad_neg


def numerical_gradient(W_in, W_out, center, context, negs, target, index_path, h=1e-5):
    """Central-difference gradient of the loss w.r.t. a single scalar
    parameter, identified by `target` ("W_in" or "W_out") and `index_path`
    (row, col) into that matrix.
    """
    matrices = {"W_in": W_in, "W_out": W_out}
    matrix = matrices[target]
    row, col = index_path

    original = matrix[row, col]

    matrix[row, col] = original + h
    loss_plus = loss_fn(W_in, W_out, center, context, negs)

    matrix[row, col] = original - h
    loss_minus = loss_fn(W_in, W_out, center, context, negs)

    matrix[row, col] = original  # restore
    return (loss_plus - loss_minus) / (2 * h)


def run_check():
    rng = np.random.default_rng(0)
    vocab_size, dim, k = 12, 6, 3
    W_in = rng.normal(scale=0.3, size=(vocab_size, dim))
    W_out = rng.normal(scale=0.3, size=(vocab_size, dim))

    center, context = 2, 5
    negs = np.array([0, 3, 9])

    grad_center, grad_context, grad_neg = analytic_gradients(W_in, W_out, center, context, negs)

    max_abs_error = 0.0

    for d in range(dim):
        numeric = numerical_gradient(W_in, W_out, center, context, negs, "W_in", (center, d))
        error = abs(numeric - grad_center[d])
        max_abs_error = max(max_abs_error, error)

    for d in range(dim):
        numeric = numerical_gradient(W_in, W_out, center, context, negs, "W_out", (context, d))
        error = abs(numeric - grad_context[d])
        max_abs_error = max(max_abs_error, error)

    for i, neg in enumerate(negs):
        for d in range(dim):
            numeric = numerical_gradient(W_in, W_out, center, context, negs, "W_out", (neg, d))
            error = abs(numeric - grad_neg[i, d])
            max_abs_error = max(max_abs_error, error)

    print(f"Max absolute error between analytic and numerical gradients: {max_abs_error:.2e}")
    tolerance = 1e-4
    if max_abs_error < tolerance:
        print(f"PASS — gradients match within tolerance ({tolerance:.0e}).")
        return True
    else:
        print(f"FAIL — gradients differ by more than tolerance ({tolerance:.0e}).")
        return False


if __name__ == "__main__":
    ok = run_check()
    sys.exit(0 if ok else 1)
