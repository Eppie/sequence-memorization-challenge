"""Fully trained model: the all-trained counterpart of RandomUpModel2.

FullyTrainedModel2 is identical to RandomUpModel2 (same architecture, same
random nn.Linear-style init for both matrices) except that NOTHING is frozen:
the up matrix is trained together with the down matrix + bias.

Because the up matrix changes every step, the hidden activations cannot be
precomputed the way train_down_matrix does — train_full_network runs the full
forward pass each epoch (the one-hot input encoding, which IS constant, is
precomputed once). The training recipe is otherwise identical: full-batch CE,
plain Adam, early stopping on accuracy (stop at 1.0, or when it hasn't improved
for `patience` epochs).
"""

from typing import Optional

import torch
import torch.nn.functional as F

from hc2_random_up import RandomUpModel2


class FullyTrainedModel2(RandomUpModel2):
    """RandomUpModel2 with the up matrix trainable as well (see module docstring)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up_matrix.requires_grad_(True)


def train_full_network(
    model: FullyTrainedModel2,
    n_epochs: int = 5000,
    lr: float = 1e-2,
    patience: int = 100,
    verbose: bool = False,
    early_stopping: bool = True,
) -> tuple:
    """Train up_matrix + down_matrix on the model's facts with CE loss.

    Mirrors hc2_hybrid.train_down_matrix (plain Adam, full batch, same early
    stopping), but backpropagates through the whole network, so the hidden
    activations are recomputed every epoch.

    early_stopping (default True) keeps the historical behaviour: stop as soon as
    accuracy hits 1.0, or when it has not improved for `patience` epochs. Set it
    False to always run the full `n_epochs` (no stopping at all) — e.g. when you
    want the fully-converged final weights rather than the earliest passing ones.

    Returns (best_accuracy, epochs_run) where best_accuracy is the highest
    accuracy observed during training.
    """
    inputs = model.facts["inputs"]
    targets = model.facts["targets"]
    n_vocab = model.settings.input_vocab_size

    # The one-hot input encoding is constant — precompute it once.
    with torch.no_grad():
        first = F.one_hot(inputs[:, 0], num_classes=n_vocab).float()
        second = F.one_hot(inputs[:, 1], num_classes=n_vocab).float()
        x_enc = torch.cat([first, second], dim=-1)  # (n_facts, 2*n_vocab)

    optimizer = torch.optim.Adam(
        [model.up_matrix, model.down_matrix], lr=lr)

    best_accuracy = 0.0
    epochs_since_improvement = 0
    epoch = 0

    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        hidden = torch.relu(x_enc @ model.up_matrix.T)
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

        if early_stopping:
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
