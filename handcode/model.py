"""The Figure-4 toy model, plus the trained baselines.

Architecture (bias-free, no norms, no residuals):

    hidden = relu(one_hot_pair(x) @ up.T)     up   : (d_mlp, 2 * n_input_vocab)
    logits = hidden @ down.T                  down : (n_output_vocab, d_mlp)

`up` is the pair of embedding matrices [E1 | E2] with W_in folded in; `down` is
the unembedding with W_out folded in. Everything in the challenge is a choice
of these two matrices.

The one-hot matmul is implemented as a gather -- `up.T[x0] + up.T[x1 + n_vocab]`
is exactly `one_hot_pair(x) @ up.T`, but costs O(n_facts * d_mlp) instead of
O(n_facts * d_mlp * n_vocab). This matters: it is the inner loop of every
training run in the binary search.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelShape:
    """The four dimensions of a toy model. The post ties them together as
    n_input_vocab = 2d, d_mlp = d, n_output_vocab = d."""

    input_vocab_size: int
    output_vocab_size: int
    d_mlp: int

    @classmethod
    def from_d(cls, d: int) -> "ModelShape":
        return cls(input_vocab_size=2 * d, output_vocab_size=d, d_mlp=d)

    @property
    def max_facts(self) -> int:
        return self.input_vocab_size**2

    @property
    def n_params(self) -> int:
        return self.d_mlp * (2 * self.input_vocab_size + self.output_vocab_size)


def hidden_activations(up: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """relu of the two embedding rows summed -- (n_facts, d_mlp)."""
    n_vocab = up.shape[1] // 2
    emb = up.T  # (2 * n_vocab, d_mlp): rows 0..V-1 are E1, rows V..2V-1 are E2
    return torch.relu(emb[inputs[:, 0]] + emb[inputs[:, 1] + n_vocab])


def forward(up: torch.Tensor, down: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return hidden_activations(up, inputs) @ down.T


def accuracy(up: torch.Tensor, down: torch.Tensor, facts: dict) -> float:
    """Fraction of facts whose argmax logit is the correct label."""
    with torch.no_grad():
        logits = forward(up, down, facts["inputs"])
        return (logits.argmax(-1) == facts["targets"]).float().mean().item()


def _uniform(shape: tuple[int, int], fan_in: int, gen: torch.Generator) -> torch.Tensor:
    """nn.Linear's default init: U(-1/sqrt(fan_in), +1/sqrt(fan_in))."""
    bound = 1.0 / fan_in**0.5
    return (torch.rand(*shape, generator=gen) * 2 - 1) * bound


def random_init(shape: ModelShape, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    up = _uniform(
        (shape.d_mlp, 2 * shape.input_vocab_size), 2 * shape.input_vocab_size, gen
    )
    down = _uniform((shape.output_vocab_size, shape.d_mlp), shape.d_mlp, gen)
    return up, down


def train(
    up: torch.Tensor,
    down: torch.Tensor,
    facts: dict,
    train_up: bool,
    n_epochs: int = 5000,
    lr: float = 1e-2,
    patience: int = 100,
) -> float:
    """Full-batch Adam on cross-entropy; returns the best accuracy seen.

    The post's recipe: Adam (no weight decay), lr=1e-2, all facts in every
    batch, up to 5000 epochs, early stop on 100% accuracy or 100 epochs without
    improvement. When `train_up` is False (hybrid / rand-emb) the up matrix is
    frozen, so the hidden activations are constant and precomputed once.
    """
    inputs, targets = facts["inputs"], facts["targets"]

    up = up.detach().clone().requires_grad_(train_up)
    down = down.detach().clone().requires_grad_(True)

    params = [down] + ([up] if train_up else [])
    optimizer = torch.optim.Adam(params, lr=lr)

    hidden_const = None
    if not train_up:
        with torch.no_grad():
            hidden_const = hidden_activations(up, inputs)

    best_accuracy = 0.0
    epochs_since_improvement = 0

    for _ in range(n_epochs):
        optimizer.zero_grad()
        hidden = hidden_activations(up, inputs) if train_up else hidden_const
        logits = hidden @ down.T
        F.cross_entropy(logits, targets).backward()
        optimizer.step()

        acc = (logits.argmax(-1) == targets).float().mean().item()
        if acc > best_accuracy:
            best_accuracy = acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if best_accuracy == 1.0 or epochs_since_improvement >= patience:
            break

    return best_accuracy
