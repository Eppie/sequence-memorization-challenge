import copy


import torch
import torch.nn.functional as F

import wandb

from models import ModelSettings, MemoryToyModel, train_model

import modal
app = modal.App("capacity-search")

# The image the Modal containers run in. The default image has none of our deps,
# so we install torch + wandb and ship our local modules so `from models import ...`
# works inside the container. (CPU torch here; for GPU add a CUDA build + gpu=... .)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "wandb", "numpy")
    .add_local_python_source("models", "device", "log")
)

MODAL_GPU = "T4"  # GPU type used when device == "gpu"


def name_function_n_facts(settings: ModelSettings) -> str:
    """Generate a descriptive name for a run based on the number of facts."""
    return f"n_facts={settings.n_facts}"


def evaluate_model(model, target_accuracy='accuracy', loss_type: str = 'BCE') -> bool:
    """Check whether a trained model has learned all its facts perfectly."""
    model.eval()
    inputs = model.facts["inputs"]
    targets = model.facts["targets"]

    if loss_type not in {'BCE', 'CE'}:
        raise ValueError("loss_type must be either 'BCE' or 'CE'.")

    with torch.no_grad():
        logits = model(inputs)
        best_guess_accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
        if loss_type == 'BCE':
            one_hot = F.one_hot(targets, model.settings.output_vocab_size)
            accuracy = (one_hot.bool() == (logits > 0)).float().mean().item()
        else:
            accuracy = best_guess_accuracy

    if target_accuracy == 'accuracy':
        return accuracy == 1.0
    elif target_accuracy == 'best_guess_accuracy':
        return best_guess_accuracy == 1.0


def find_max_facts(settings: ModelSettings,
                   precision: int = 1,
                   n_epochs: int = 2000,
                   lr: list[float] = [1e-2],
                   optimizer_type: str = 'Adam',
                   grad_clip_norm=None,
                   patience: int = 100,
                   number_of_attempts: int = 1,
                   log_to_wandb: bool = True, wandb_log_every: int = 10,
                   wandb_group: str = 'capacity_search',
                   verbose: bool = True,
                   name_function = name_function_n_facts,
                   target_accuracy: str = 'accuracy',
                   threshold_to_continue: float = 0.99,
                   loss_type: str = 'BCE',
                   use_modal: bool = False,
                   any_all_most: str = 'any',
                   device: str = 'gpu') -> int:
    """Binary search for the maximum number of facts a model architecture can learn.

    Args:
        settings:      Base ModelSettings (n_facts will be overridden per candidate).
        precision:     Stop searching when hi - lo < precision.
        n_epochs:      Max training epochs per run.
        patience:      Early-stopping patience (epochs without accuracy improvement).
        lr:            List of learning rates to try.
        number_of_attempts: Number of independent attempts per candidate n_facts.
                       A single success is sufficient; only all-fail counts as failure.
        log_to_wandb:  Whether to log individual runs to W&B.
        wandb_group:   W&B group name for the search.
        wandb_log_every: How often to log to W&B (in epochs).
        verbose:       Print progress during the search.
        target_accuracy:     Metric to train for ('accuracy' or 'best_guess_accuracy').
        threshold_to_continue: 
                       Minimum accuracy required to continue searching with a lower learning rate.
        loss_type:     Either 'BCE' or 'CE' for the training loss used in each trial.
        use_modal:     If True, run the per-candidate attempts in parallel on Modal
                       (via _try_n_facts_modal.map). If False, run them locally in a loop.
        any_all_most:  Only used if use_modal is True. 
                       If 'any', a single successful attempt counts as success. If 'all',
                       all attempts must succeed. If 'most', a majority of attempts must succeed.
        device:        Only used on the Modal path: 'gpu' runs each training attempt on a
                       GPU container, 'cpu' on a CPU container.
    Returns:
        The maximum n_facts for which the model achieved perfect accuracy
        (within the given precision).
    """
    max_possible = settings.input_vocab_size ** settings.input_len
    lo, hi = 1, max_possible
    best = 0  # highest n_facts confirmed learnable

    kwargs = dict(
        n_epochs=n_epochs, lr=lr, 
        optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm, 
        patience=patience,
        log_to_wandb=log_to_wandb,
        wandb_group=wandb_group,
        wandb_log_every=wandb_log_every,
        name_function=name_function,
        target_accuracy=target_accuracy,
        threshold_to_continue=threshold_to_continue,
        loss_type=loss_type
    )

    if verbose:
        print(f"Searching for max learnable facts in [{lo}, {hi}]")
        print()

    while hi - lo >= precision:
        mid = (lo + hi) // 2

        if verbose:
            print(f"Trying n_facts = {mid} ...")


        learned = _try_n_facts_several_attempts(
            settings,
            mid,
            number_of_attempts,
            verbose,
            use_modal,
            any_all_most,
            device,
            **kwargs
           )

       
        if learned:
            best = mid
            lo = mid + 1
            if verbose:
                print(f"✓  learned all {mid} facts. Now searching: {lo} - {hi}")
        else:
            hi = mid - 1
            if verbose:
                print(f"✗  failed to learn {mid} facts. Now searching: {lo} - {hi}")
            
    if hi == max_possible: #Then we should also check if we can learn max_possible facts
        if verbose:
            print(f"Trying n_facts = {max_possible} ...")
        learned = _try_n_facts_several_attempts(
            settings,
            max_possible,
            number_of_attempts,
            verbose,
            use_modal,
            any_all_most,
            device,
            **kwargs
           )
        
        if learned:
            best = max_possible
        
    if verbose:
        print(f"\nMax learnable facts: {best}")

    return best

@app.function(image=image, timeout=86400, nonpreemptible=True)  # 24h cap; non-preemptible so a
# whole per-variant search never restarts (the inner attempts stay preemptible - cheap to redo)
def find_max_facts_modal(config: dict) -> int:
    """Run one capacity search on Modal.

    `config` is exactly the keyword arguments for find_max_facts (settings,
    precision, lr, wandb_group, name_function, ...). We only add use_modal=True,
    which makes each search nest its attempts onto GPU containers.
    """
    return find_max_facts(use_modal=True, **config)

def _try_n_facts_several_attempts(settings: ModelSettings, 
                                  num_facts: int,
                                  number_of_attempts: int, 
                                  verbose: bool,
                                  use_modal: bool,
                                  any_all_most: str,
                                  device: str,
                                  **kwargs) -> bool:
    if not use_modal:
        learned = False
        for attempt in range(1, number_of_attempts + 1):
            if verbose and number_of_attempts > 1:
                print(f"  Attempt {attempt}/{number_of_attempts}")
            success = _try_n_facts(settings, num_facts,
                                   verbose=verbose, **kwargs)
            if success:
                learned = True
                break

        return learned
    
    else:
        if device == 'gpu':
            fn = _try_n_facts_modal.with_options(gpu=MODAL_GPU)
        elif device == 'cpu':
            fn = _try_n_facts_modal
        else:
            raise ValueError("device must be 'cpu' or 'gpu'")

        # Only the first attempt logs to W&B; the rest are forced off, to avoid N
        # near-identical W&B runs per candidate. log_to_wandb has to be a per-call
        # positional arg here, because .map can vary positional args but not kwargs.
        log_to_wandb = kwargs.pop('log_to_wandb', False)
        log_flags = [log_to_wandb] + [False] * (number_of_attempts - 1)

        successes = list(fn.map(
            [settings] * number_of_attempts,   # -> settings      (positional 1)
            log_flags,                          # -> log_to_wandb  (positional 2)
            kwargs=dict(n_facts=num_facts, verbose=verbose, **kwargs),
        ))

        if any_all_most == 'any':
            learned = any(successes)   # "any attempt wins"
        elif any_all_most == 'all':
            learned = all(successes)   # "all attempts must succeed"
        elif any_all_most == 'most':
            learned = sum(successes) > len(successes) // 2   # "majority wins"

        return learned

def _try_n_facts(base_settings: ModelSettings,
                 n_facts: int,
                 n_epochs: int,
                 lr: list[float],
                 optimizer_type: str,
                 grad_clip_norm: int|None,
                 patience: int,
                 log_to_wandb: bool,
                 wandb_group: str,
                 wandb_log_every: int,
                 verbose: bool,
                 name_function: callable,
                 target_accuracy: str,
                 threshold_to_continue: float,
                 loss_type: str ) -> bool:
    """Train a model with the given n_facts. Returns True if it achieves perfect accuracy."""
    trial_settings = copy.deepcopy(base_settings)
    trial_settings.n_facts = n_facts

    model = MemoryToyModel(trial_settings)

    if log_to_wandb:
        wandb.init(
            project="Memory Toy Models",
            group=wandb_group,
            config=vars(trial_settings),
            reinit=True,
            name=name_function(trial_settings)
        )

    for lr_value in lr:
        success, best_accuracy = train_model(
                                    model, n_epochs=n_epochs, lr=lr_value, 
                                    optimizer_type=optimizer_type, grad_clip_norm=grad_clip_norm,
                                    log_to_wandb=log_to_wandb, wandb_log_every=wandb_log_every, 
                                    wandb_finish=False, wandb_group=wandb_group,
                                    early_stopping=True, patience=patience, verbose=verbose,
                                    target_accuracy=target_accuracy, loss_type=loss_type)
        if success:
            break  # stop if we found a learning rate that works
        if best_accuracy < threshold_to_continue:
            break  # stop if accuracy is very low, unlikely to improve with more training

    if log_to_wandb:
        wandb.log({"capacity_search/n_facts": n_facts,
                    "capacity_search/success": success,
                    "capacity_search/best_accuracy": best_accuracy})
        wandb.finish()

    return success



@app.function(image=image, timeout=60*60*10,  # 10h cap: one full 50k-epoch attempt
              secrets=[modal.Secret.from_name("wandb")])  # injects WANDB_API_KEY for wandb.init
def _try_n_facts_modal(settings, log_to_wandb, **kwargs) -> bool:
    """Variant of _try_n_facts that runs on Modal.

    `settings` and `log_to_wandb` are explicit positional args so that .map can
    vary log_to_wandb per attempt (only one attempt of a candidate logs to W&B);
    everything else is forwarded via kwargs.

    The base function is CPU-only. The caller adds a GPU at runtime via
    .with_options(gpu=MODAL_GPU) when device == 'gpu' (see
    _try_n_facts_several_attempts). A GPU can be *added* this way but not *unset*,
    which is why the base has no gpu=. models.py sets the default device to
    cuda-if-available, so the model uses whichever device the container has.
    """
    return _try_n_facts(settings, log_to_wandb=log_to_wandb, **kwargs)



if __name__ == "__main__":
    # Example: search capacity for the default architecture
    settings = ModelSettings()
    max_facts = find_max_facts(settings, log_to_wandb=False)
    print(f"Result: {max_facts}")
