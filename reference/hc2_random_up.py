"""Random-up / trained-down model: the random baseline for HybridModel2.

RandomUpModel2 has the same architecture as HandCodedModel2 / HybridModel2:

    hidden = relu(one_hot_pair @ up_matrix.T)      # (batch, d_ff)
    logits = hidden @ down_matrix.T                # (batch, n_labels)

but the up matrix is NOT hand-coded — it is random (nn.Linear-style uniform
init, U(+-1/sqrt(fan_in)) with fan_in = 2*input_vocab_size, no bias, matching
the up layer's shape in HandCodedModel2) and frozen. The down matrix is
randomly initialised and trainable, exactly as in HybridModel2; train it
with hc2_hybrid.train_down_matrix (it only touches up_matrix / down_matrix /
facts, all of which this class provides).

Because nothing about the up matrix is hand-coded, there is no S, top_n, or
top_fraction — the only randomness knob is init_seed, which draws both the up
matrix and the down init.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from models import generate_facts


class RandomUpModel2:
    """Random frozen up matrix + trainable down matrix (see module docstring)."""

    def __init__(
        self,
        input_vocab_size: int = 32,
        output_vocab_size: int = 16,
        n_facts: int = 16,
        d_ff: int = 16,
        seed: int = 42,          # facts seed (fixed across attempts, like the other searches)
        init_seed: Optional[int] = None,  # up + down init seed (varies per attempt)
    ):
        # Minimal settings object so train_down_matrix and evaluate() can read
        # the fields they need, mirroring HandCodedModel2Settings.
        class _S:
            pass
        self.settings = _S()
        self.settings.input_vocab_size = input_vocab_size
        self.settings.output_vocab_size = output_vocab_size
        self.settings.n_facts = n_facts
        self.settings.d_ff = d_ff
        self.settings.seed = seed

        self.facts = generate_facts(
            n_facts=n_facts,
            input_len=2,
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            seed=seed,
        )
        device = self.facts["targets"].device

        if init_seed is not None:
            gen = torch.Generator(device=device).manual_seed(init_seed)
        else:
            gen = None

        # Frozen random up matrix: same shape/role as HandCodedModel2's up
        # matrix, nn.Linear-style init with fan_in = the one-hot input width.
        in_dim = input_vocab_size * 2
        up_bound = 1.0 / (in_dim ** 0.5)
        self.up_matrix = (
            (torch.rand(d_ff, in_dim, generator=gen, device=device) * 2 - 1) * up_bound
        )

        # Trainable down matrix, initialised exactly as in HybridModel2.
        down_bound = 1.0 / (d_ff ** 0.5)
        self.down_matrix = (
            (torch.rand(output_vocab_size, d_ff, generator=gen, device=device) * 2 - 1) * down_bound
        ).requires_grad_(True)

    def forward(self, x):
        """Same forward pass as HandCodedModel2. Returns (logits, hidden)."""
        first = F.one_hot(x[:, 0], num_classes=self.settings.input_vocab_size).float()
        second = F.one_hot(x[:, 1], num_classes=self.settings.input_vocab_size).float()
        x_enc = torch.cat([first, second], dim=-1)
        hidden = torch.relu(x_enc @ self.up_matrix.T)
        logits = hidden @ self.down_matrix.T
        return logits, hidden

    def evaluate(self):
        """Same metric as HandCodedModel2.evaluate()."""
        logits, hidden = self.forward(self.facts["inputs"])
        accuracy = self.facts["targets"].eq(logits.argmax(dim=1)).float().mean().item()
        return accuracy, logits, hidden
