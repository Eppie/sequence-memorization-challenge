"""Fact generation for the sequence-memorization toy task.

Quoted verbatim (modulo formatting) from the "Training data" section of
Linsefors & Bushnaq, "Challenge: Hand coding weights for efficient sequence
memorisation" (LessWrong, 2026-07-23).

A "fact" is a pair of input tokens mapped to one output label. Input pairs are
drawn uniformly at random from all `input_vocab_size ** 2` possibilities and
labels are dealt round-robin, so there is no structure to exploit: the model
can only memorize.
"""

import torch


def generate_facts(
    n_facts: int,
    input_vocab_size: int,
    output_vocab_size: int,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    if n_facts > input_vocab_size**2:
        raise ValueError(
            f"Cannot generate {n_facts} unique facts with a vocabulary of size "
            f"{input_vocab_size}. Maximum unique facts: {input_vocab_size ** 2}"
        )

    device = torch.tensor(0).device  # respect default device
    generator = torch.Generator(device=device).manual_seed(seed)

    all_possible_inputs = torch.cartesian_prod(
        torch.arange(input_vocab_size), torch.arange(input_vocab_size)
    )
    inputs = all_possible_inputs[
        torch.randperm(all_possible_inputs.size(0), generator=generator)[:n_facts]
    ]

    targets = torch.arange(n_facts) % output_vocab_size
    sorted_indices = torch.argsort(targets)

    return {"inputs": inputs[sorted_indices], "targets": targets[sorted_indices]}


def one_hot_pair(inputs: torch.Tensor, input_vocab_size: int) -> torch.Tensor:
    """(n_facts, 2) token ids -> (n_facts, 2 * input_vocab_size) concatenated one-hots.

    The two halves are the two token positions, which is what makes the two
    embedding matrices E1 / E2 of Figure 4 a single (d_mlp, 2 * n_vocab) matrix.
    """
    first = torch.nn.functional.one_hot(inputs[:, 0], input_vocab_size).float()
    second = torch.nn.functional.one_hot(inputs[:, 1], input_vocab_size).float()
    return torch.cat([first, second], dim=-1)
