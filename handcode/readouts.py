"""Better unembeddings for the hand-coded embedding, without gradient descent.

Diagnosis this is built on: in the authors' construction the correct label's
logit is *exactly* zero by design, and every error is a tie -- some wrong label's
neurons all happen to be silent too, and argmax breaks the tie by index. Nothing
is ever ranked wrong; it is undecided. Measured at their acc=1 capacity points,
100% of errors are such ties.

Their unembedding cannot break those ties, and not by accident. With weight -2
on a label's own neurons and a uniform w elsewhere,

    logit_c = w * H - (2 + w) * sum_{i in S_c} h_i ,   H = sum_i h_i

and `w * H` does not depend on c, so the ranking is identical for every w -- the
authors note this themselves. Breaking ties needs a *non-uniform* readout.

The obvious candidate, a Hebbian/prototype term eps * <h, mu_c> with mu_c the
mean hidden vector of label c's facts, does NOT work, for the same reason one
order up. Labels have no shared structure here (the facts are random pairs), so
mu_c is essentially a global mean activation m masked by label c's own neurons,
mu_c ~ m * (1 - 1_{S_c}). On a tied fact h is already zero on every tied label's
neurons, so

    <h, mu_c> = sum_{i not in S_c} h_i m_i = <h, m> - 0 = <h, m>

-- identical for every tied label. Measured: the correct label wins such ties at
chance rate. A prototype readout is structurally incapable of separating them.

What does work is whitening. Ridge regression is (H^T H + lam I)^-1 H^T Y, and
the Hebbian rule is exactly its lam -> infinity limit (the unwhitened H^T Y).
The (H^T H)^-1 factor is what makes the readout fact-aware rather than
prototype-aware, and fact-aware is the only thing that can help here.

So the two readouts offered are:

  ridge_down     solve for the unembedding directly, discarding the silence code;
  tiebreak_down  keep the silence code as primary and add a *scaled* ridge
                 correction, small enough to adjudicate ties and nothing else.

Both are closed-form; the post's rules explicitly allow ridge regression.
"""

import torch

from .handcoded import build_down_matrix


def label_means(hidden: torch.Tensor, targets: torch.Tensor, n_labels: int) -> torch.Tensor:
    """Unit-norm mean hidden vector per label -- (n_labels, d_mlp).

    mu_c points along "what facts labeled c look like". Labels with no facts
    get a zero row.
    """
    totals = torch.zeros(n_labels, hidden.shape[1])
    totals.index_add_(0, targets, hidden)
    counts = torch.bincount(targets, minlength=n_labels).clamp(min=1).unsqueeze(1)
    means = totals / counts
    return means / means.norm(dim=1, keepdim=True).clamp(min=1e-12)


def tiebreak_down(
    base: torch.Tensor,
    correction: torch.Tensor,
    hidden: torch.Tensor,
    rho: float = 0.5,
) -> torch.Tensor:
    """A discrete primary readout plus a correction that can only break its ties.

        logit_c = <h, base_c>  +  eps * <h, correction_c>

    Both constructions here have integer-valued activations and integer-valued
    primary readouts, so `base` separates any two labels it genuinely ranks by
    at least 2 (the silence code's logits are even and <= 0; the coincidence
    code's are even and >= 0). Choosing

        eps = rho / max_{f,c} |<h_f, correction_c>|   with rho < 1

    bounds each label's correction by rho, so the largest possible swing between
    two labels is 2 * rho < 2 and no gap of 2 can ever be closed. The primary
    code therefore decides exactly what it decided before, and the correction is
    consulted only where it had abstained.
    """
    swing = (hidden @ correction.T).abs().max().clamp(min=1e-12)
    return base + (rho / swing) * correction


def silence_base(conn) -> torch.Tensor:
    """The authors' primary readout, for use as `base` above."""
    return build_down_matrix(conn)


def ridge_down(
    hidden: torch.Tensor,
    targets: torch.Tensor,
    n_labels: int,
    alpha: float,
) -> torch.Tensor:
    """Closed-form ridge regression of one-hot labels on the hidden features.

    W = argmin ||H W^T - Y||^2 + lambda ||W||^2, i.e. W^T = (H^T H + lambda I)^-1 H^T Y.
    `alpha` sets lambda relative to the mean eigenvalue of H^T H so one grid of
    alphas works across model sizes. H^T Y is accumulated directly rather than
    materializing the one-hot Y.
    """
    h = hidden.double()
    d_mlp = h.shape[1]

    gram = h.T @ h  # (d_mlp, d_mlp)
    rhs = torch.zeros(n_labels, d_mlp, dtype=torch.float64)
    rhs.index_add_(0, targets, h)  # (n_labels, d_mlp) == (H^T Y)^T

    lam = alpha * gram.diagonal().mean().clamp(min=1e-12)
    gram = gram + lam * torch.eye(d_mlp, dtype=torch.float64)
    return torch.linalg.solve(gram, rhs.T).T.float()


# Ridge strengths tried for each (S, top_fraction) cell; the solve is a d x d
# system, so sweeping these costs far less than rebuilding the embedding.
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 1e-1, 1.0)
