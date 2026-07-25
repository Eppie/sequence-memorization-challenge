import torch
import torch.nn as nn
import torch.nn.functional as F

from device import setup_default_device
setup_default_device()  # cuda if available, else cpu; override with MTM_DEVICE

import math
import copy

import wandb
import os
os.environ["WANDB_SILENT"] = "true"
WANDB_PROJECT = "Memory Toy Models"

class ModelSettings:
    def __init__(self, input_len=2, input_vocab_size=32, output_vocab_size=16, n_facts=64, seed=42,
                 d_residual=16, n_heads=1, d_ff=16,
                 attention=True, qk_is_one=False,
                 ff=True, bias=True, norms=True, ff_residual=True, ff_activation_type='GELU'):
        
        # Data dimensions
        self.input_len = input_len
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.n_facts = n_facts
        self.seed = seed  # For reproducibility

        # Internal model dimensions
        self.d_residual = d_residual
        self.n_heads = n_heads
        self.d_ff = d_ff

        # Model architectural choices
        self.attention = attention
        self.qk_is_one = qk_is_one # If True, use a fixed attention matrix of all ones 
        self.ff = ff
        self.bias = bias
        self.norms = norms
        self.ff_residual = ff_residual
        self.ff_activation_type = ff_activation_type  # Can be extended to support other activations


# Mapping strings to actual activation functions
ACTIVATION = {
    'GELU': lambda: nn.GELU(),
    'ReLU': lambda: nn.ReLU(),
}


def generate_facts(n_facts: int, # of facts to generate,
                   input_len: int, # numer of input tokens per fact   
                   input_vocab_size: int, # of unique tokens in the vocabulary
                   output_vocab_size: int, # of unique targets
                   seed: int = 42
                  ) -> dict[str, torch.Tensor]:
    
    if n_facts > input_vocab_size ** input_len:
        raise ValueError(f"Cannot generate {n_facts} unique facts with a vocabulary of size {input_vocab_size} and input length {input_len}. Maximum unique facts: {input_vocab_size ** input_len}")
    
    device = torch.tensor(0).device  # respect default device
    generator = torch.Generator(device=device).manual_seed(seed)

    targets = torch.arange(n_facts) % output_vocab_size

    if input_len == 1:
        inputs = torch.randperm(input_vocab_size, generator=generator)[:n_facts].unsqueeze(1)
    elif input_len == 2:
        all_possible_inputs = torch.cartesian_prod(torch.arange(input_vocab_size), torch.arange(input_vocab_size))
        inputs = all_possible_inputs[torch.randperm(all_possible_inputs.size(0), generator=generator)[:n_facts]]
    else:
        inputs = torch.randint(0, input_vocab_size, (n_facts, input_len), generator=generator)

    sorted_indices = torch.argsort(targets)    
    return {"inputs": inputs[sorted_indices], "targets": targets[sorted_indices]}



class MemoryToyModel(nn.Module):
    """A single-layer transformer model for sequence modeling."""

    def __init__(self, settings):
        super().__init__()
        self.settings = copy.deepcopy(settings)
        
        if settings.attention:
            # Token and positional embeddings
            self.token_emb = nn.Embedding(settings.input_vocab_size, settings.d_residual)
            self.pos_emb = nn.Embedding(settings.input_len, settings.d_residual)

            # Single transformer layer
            self.attn = CausalSelfAttention(settings.d_residual, settings.n_heads, settings.qk_is_one)
            self.ln1 = nn.RMSNorm(settings.d_residual) if settings.norms else nn.Identity()

        else:
            self.token_emb = nn.ModuleList([nn.Embedding(settings.input_vocab_size, settings.d_residual) 
                              for _ in range(settings.input_len)])

        if settings.ff:
            self.ln2 = nn.RMSNorm(settings.d_residual) if settings.norms else nn.Identity()
            self.ff = nn.Sequential(
                nn.Linear(settings.d_residual, settings.d_ff, bias=settings.bias),
                ACTIVATION[settings.ff_activation_type](),
                nn.Linear(settings.d_ff, settings.d_residual, bias=settings.bias),
            )

        # Output head
        self.ln_f = nn.RMSNorm(settings.d_residual) if settings.norms else nn.Identity()
        self.head = nn.Linear(settings.d_residual, settings.output_vocab_size, bias=settings.bias)

        # Generate and store the facts as buffers so they are part of the model's state_dict
        facts = generate_facts(settings.n_facts, settings.input_len, 
                                    settings.input_vocab_size, settings.output_vocab_size, 
                                    settings.seed)
        self.register_buffer('fact_inputs', facts['inputs'])
        self.register_buffer('fact_targets', facts['targets'])

    @property
    def facts(self):
        return {"inputs": self.fact_inputs, "targets": self.fact_targets}

    def forward(self, idx):
        settings = self.settings

        B, T = idx.shape
        assert T == settings.input_len, f"Sequence length is {T} but should be {settings.input_len}"

        # Embeddings and attention
        if settings.attention:
            tok_emb = self.token_emb(idx)
            pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
            x = tok_emb + pos_emb
            x = x[:, -1, :] + self.attn(self.ln1(x))[:, -1, :]
        else:
            x = sum(emb(idx[:, i]) for i, emb in enumerate(self.token_emb))

        # Feedforward
        if settings.ff:
            if settings.ff_residual:
                x = x + self.ff(self.ln2(x))
            else:
                x = self.ff(self.ln2(x))

        # Output
        logits = self.head(self.ln_f(x))

        return logits
    



def train_model(model, n_epochs=2000, lr=1e-2, 
                optimizer_type='Adam', grad_clip_norm=None, smoothing=None,
                log_to_wandb=True, wandb_log_every=10, wandb_finish=True,
                wandb_project=WANDB_PROJECT, wandb_group='test', wandb_name=None,
                early_stopping = False, patience = 100, verbose = True,
                target_accuracy = 'accuracy', loss_type='BCE') -> bool:
    """Train a MemoryToyModel on its stored facts.
    
    Args:
        model: A MemoryToyModel instance.
        n_epochs: Number of training epochs.
        lr: Learning rate (defaults to 1e-2).
        log_every: Print loss every this many epochs (0 to disable).
        wandb_log_every: Log to wandb every this many epochs.
        wandb_finish: Whether to call wandb.finish() at the end of training (to properly close the run).
        wandb_project: W&B project name. If provided, logs to wandb.
        wandb_group: W&B group name. If provided, logs to wandb.
        early_stopping: Whether to stop training early if accuracy plateaus.
        patience: Number of epochs to wait for accuracy improvement before stopping.
        verbose: Whether to print training progress and early stopping info.
        loss_type: Either 'BCE' or 'CE' to select binary or multiclass cross entropy.
    Returns:
        True if the model achieved perfect monitored accuracy, False otherwise.
    """
    device = next(model.parameters()).device
    inputs = model.facts["inputs"]
    targets = model.facts["targets"]

    if loss_type not in {'BCE', 'CE'}:
        raise ValueError("loss_type must be either 'BCE' or 'CE'.")

    # Initialise wandb run (reuse existing run if already active)
    if log_to_wandb and wandb.run is None:
        wandb.init(
            project=wandb_project,
            group=wandb_group,
            config=vars(model.settings),
            name=wandb_name,
        )

    # Determine epoch offset from existing wandb run so consecutive
    # calls to train_model continue the epoch count seamlessly.
    epoch_offset = 0
    if log_to_wandb and wandb.run is not None:
        epoch_offset = wandb.run.summary.get("epoch", 0)

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    one_hot_targets = F.one_hot(targets, model.settings.output_vocab_size).float()

    if loss_type == 'BCE':
        if smoothing is None:
            loss_targets = one_hot_targets
        else:
            loss_targets = torch.ones_like(one_hot_targets) * smoothing + one_hot_targets * (1 - 2*smoothing)
    else:
        loss_targets = targets

    # Early stopping state
    best_accuracy = None
    epochs_since_change = 0
    early_stopping_triggered = False

    model.train()
    for epoch in range(1, n_epochs + 1):
        global_epoch = epoch_offset + epoch

        optimizer.zero_grad()
        
        logits = model(inputs)
        if loss_type == 'BCE':
            loss = F.binary_cross_entropy_with_logits(logits, loss_targets)
        else:
            loss = F.cross_entropy(logits, loss_targets, label_smoothing=0.0 if smoothing is None else smoothing)

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        loss_val = loss.item()
        if loss_type == 'BCE':
            accuracy = (one_hot_targets.bool() == (logits > 0)).float().mean().item()
        else:
            accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        best_guess_accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        if target_accuracy == 'accuracy':
            monitored_accuracy = accuracy
        elif target_accuracy == 'best_guess_accuracy':
            monitored_accuracy = best_guess_accuracy

        if log_to_wandb and epoch % wandb_log_every == 0:
            wandb.log({"loss": loss_val, "epoch": global_epoch, "lr": lr, "accuracy": accuracy, "best_guess_accuracy": best_guess_accuracy})

        if early_stopping:

            if monitored_accuracy == 1.0:
                if verbose:
                    print(f"Early stopping at epoch {epoch}: perfect accuracy reached.")
                early_stopping_triggered = True
                break
            if best_accuracy is not None and monitored_accuracy <= best_accuracy:
                epochs_since_change += 1
                if epochs_since_change >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}: accuracy stable at {best_accuracy:.2%} for {patience} epochs.")
                    early_stopping_triggered = True
                    break
            else:
                best_accuracy = monitored_accuracy
                epochs_since_change = 0

    if early_stopping_triggered and log_to_wandb and epoch % wandb_log_every != 0:
        wandb.log({"loss": loss_val, "epoch": global_epoch, "lr": lr, "accuracy": accuracy, "best_guess_accuracy": best_guess_accuracy})

    if not early_stopping_triggered and verbose:
        print(f"Finished {n_epochs} epochs. Final accuracy: {accuracy:.2%}")

    if log_to_wandb:
        wandb.log({"learned_all_facts": monitored_accuracy==1.0, 
                   "early_stopping_triggered": early_stopping_triggered})
    if wandb_finish:
        wandb.finish()

    return monitored_accuracy == 1.0, best_accuracy


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, d_model, n_heads, qk_is_one=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qk_is_one = qk_is_one

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.qk_is_one:
            attn = torch.ones(T, T, device=x.device, dtype=v.dtype) / T

        else:
            # Scaled dot-product attention with causal mask
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float('-inf'))
            attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

