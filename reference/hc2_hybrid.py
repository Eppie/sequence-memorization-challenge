"""Hybrid hand-coded / trained model built on HandCodedModel2.

HybridModel2 keeps the analytically constructed MLP *up* matrix of
HandCodedModel2 (built by exactly the same code path, so all settings — S,
top_n / top_fraction, tie-breaking randomness, connection matrix — apply
unchanged) but throws away the hand-coded *down* matrix. The down matrix is
instead randomly initialised (nn.Linear-style uniform) and trained with
full-batch cross-entropy while the up matrix stays frozen.

Because the up matrix is frozen, the hidden activations for the fact set are
constant throughout training. train_down_matrix therefore precomputes them
once and each epoch is just a linear layer + CE — training cost is
O(n_facts * d_ff * n_labels) per epoch regardless of input vocab size.

Training recipe (deliberate, matching the repo's conventions): plain Adam,
no weight decay, no gradient clipping, full-batch CE loss. Early stopping on
accuracy: stop at 1.0, or when it hasn't improved for `patience` epochs.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from hc2 import HandCodedModel2, HandCodedModel2Settings


class HybridModel2(HandCodedModel2):
    """HandCodedModel2 up matrix (frozen) + randomly initialised trainable down.

    The constructor first builds a full HandCodedModel2 — facts, connection
    matrix, and up matrix are produced by exactly the same code — then replaces
    down_matrix with a random trainable parameter.

    init_seed: seed for the down-matrix initialisation. If None, the global
    torch RNG is used (whatever state it is in after the up-matrix build).
    """

    def __init__(
        self,
        settings: HandCodedModel2Settings,
        precomputed_conn: Optional[np.ndarray] = None,
        init_seed: Optional[int] = None,
    ):
        super().__init__(settings, precomputed_conn=precomputed_conn)

        hidden_dim = settings.d_ff
        n_labels = settings.output_vocab_size
        device = self.up_matrix.device

        # nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for the
        # weight, with fan_in = hidden_dim.
        bound = 1.0 / (hidden_dim ** 0.5)
        if init_seed is not None:
            gen = torch.Generator(device=device).manual_seed(init_seed)
        else:
            gen = None
        self.down_matrix = (
            (torch.rand(n_labels, hidden_dim, generator=gen, device=device) * 2 - 1) * bound
        ).requires_grad_(True)

    # forward() and evaluate() are inherited from HandCodedModel2 unchanged.


def train_down_matrix(
    model: HybridModel2,
    n_epochs: int = 5000,
    lr: float = 1e-2,
    patience: int = 100,
    verbose: bool = False,
) -> tuple:
    """Train down_matrix on the model's stored facts with CE loss.

    The up matrix is frozen: hidden activations are precomputed once, so each
    epoch is a single linear layer forward/backward. Plain Adam, full batch.

    Early stopping: stop as soon as accuracy reaches 1.0, or when it has not
    improved for `patience` consecutive epochs.

    Returns (best_accuracy, epochs_run) where best_accuracy is the highest
    accuracy observed during training.
    """
    inputs = model.facts["inputs"]
    targets = model.facts["targets"]
    n_vocab = model.settings.input_vocab_size

    # Precompute the (constant) hidden activations for all facts.
    with torch.no_grad():
        first = F.one_hot(inputs[:, 0], num_classes=n_vocab).float()
        second = F.one_hot(inputs[:, 1], num_classes=n_vocab).float()
        x_enc = torch.cat([first, second], dim=-1)
        hidden = torch.relu(x_enc @ model.up_matrix.T)  # (n_facts, d_ff)

    optimizer = torch.optim.Adam([model.down_matrix], lr=lr)

    best_accuracy = 0.0
    epochs_since_improvement = 0
    epoch = 0

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        logits = hidden @ model.down_matrix.T
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if best_accuracy == 1.0:
            if verbose:
                print(f"Early stopping at epoch {epoch}: perfect accuracy.")
            break
        if epochs_since_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}: no improvement for "
                      f"{patience} epochs (best {best_accuracy:.2%}).")
            break

    if verbose and epoch == n_epochs:
        print(f"Finished {n_epochs} epochs. Best accuracy: {best_accuracy:.2%}")

    return best_accuracy, epoch
