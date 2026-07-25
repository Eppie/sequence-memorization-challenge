#%%
"""
Hand-coded weights for the sequence-memorisation toy model.

Companion code for the "My attempt" section of the post: it sets the toy
model's weights analytically (no gradient descent) so that a given fact set --
pairs of input tokens mapped to output labels -- is memorised directly. The
post explains the construction and why it works; the comments in this file only
map that construction onto the code.

The model is the one in Figure 4 of the post. It has two weight matrices:

    up_matrix    -- the embedding    : shape (d_ff, 2 * input_vocab)
    down_matrix  -- the unembedding  : shape (n_labels, d_ff)

Naming bridge to the post: d_ff is d_MLP; n_neurons_per_label is S. The neurons
assigned to a label are the rows set to 1 in that label's column of the
connection matrix. The concrete weight values are documented where each matrix
is filled in, in HandCodedModel2.__init__.

File layout
-----------
    1. make_connection_matrix / evaluate_connection_matrix
         assign S neurons to each label   (post: "Assigning neurons to labels")
    2. HandCodedModel2 (+ Settings)
         build up_matrix and down_matrix from a fact set
         (post: "Embedding weights" and "Unembedding weights")
"""

import numpy as np
from itertools import combinations
from typing import Optional
import warnings


# ── 1. Assigning neurons to labels: the connection matrix ─────────────────────
# A connection matrix is a D x T binary matrix; conn[i, j] = 1 means neuron i is
# assigned to label j. (In the connection-matrix code below the columns are
# called "features"; in the model they are the labels.) We want:
#   * exactly S ones per column     -- every label is assigned S neurons,
#   * balanced row sums (~S*T/D)    -- every neuron is assigned to roughly equally many labels,
#   * pairwise column overlap <= 1  -- no two labels share more than one neuron.
# The last property is a combinatorial-design constraint, achievable only when
#   T <= D*(D-1) / (S*(S-1)).
# Beyond that bound some label pairs must share >= 2 neurons, and the
# construction merely minimises the number of such overlaps.


def make_connection_matrix(
    D: int,
    T: int,
    S: int,
    seed: Optional[int] = None,
    n_restarts: int = 5,
    sa_steps: int = 30_000,
) -> np.ndarray:
    """
    Generate a D x T binary connection matrix.

    Args:
        D: number of neurons (rows)
        T: number of features (columns)
        S: neurons per feature -- each column has exactly S ones
        seed: random seed for reproducibility
        n_restarts: independent attempts; returns the best result
        sa_steps: simulated-annealing steps per attempt

    Returns:
        numpy int8 array of shape (D, T)
    """
    if not (0 < S <= D):
        raise ValueError(f"Need 0 < S <= D, got S={S}, D={D}")
    if D <= 0 or T <= 0:
        raise ValueError("D and T must be positive integers")

    if S > 1:
        max_feasible_T = D * (D - 1) // (S * (S - 1))
        if T > max_feasible_T:
            warnings.warn(
                f"T={T} exceeds the theoretical maximum ({max_feasible_T}) for "
                f"pairwise overlap <= 1 with D={D}, S={S}. "
                f"Some feature pairs will inevitably share >= 2 neurons.",
                stacklevel=2,
            )

    rng = np.random.default_rng(seed)
    best_matrix = None
    best_score = float("inf")

    for _ in range(n_restarts):
        m = _greedy_init(D, T, S, rng)
        m = _sa_improve(m, D, T, S, rng, steps=sa_steps)
        m = _balance_fix(m, T)  # greedy balance pass without new violations
        score = _total_score(m)
        if score < best_score:
            best_score = score
            best_matrix = m.copy()

    return best_matrix


def evaluate_connection_matrix(M: np.ndarray) -> dict:
    """
    Compute quality statistics for a connection matrix.

    Returns a dict with column sums (should all equal S), row sum statistics,
    max pairwise overlap between any two feature columns, and overlap violation counts.
    """
    D, T = M.shape
    row_sums = M.sum(axis=1)
    col_sums = M.sum(axis=0)

    pairwise = [int(M[:, j1] @ M[:, j2]) for j1, j2 in combinations(range(T), 2)]

    return {
        "shape": (D, T),
        "col_sum_min": int(col_sums.min()),
        "col_sum_max": int(col_sums.max()),
        "row_sum_min": int(row_sums.min()),
        "row_sum_max": int(row_sums.max()),
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
        "max_pairwise_overlap": max(pairwise) if pairwise else 0,
        "overlap_violation_pairs": sum(1 for o in pairwise if o > 1),
        "overlap_violations_total": sum(o - 1 for o in pairwise if o > 1),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _greedy_init(D: int, T: int, S: int, rng: np.random.Generator) -> np.ndarray:
    """
    Column-by-column greedy construction.

    For each column, picks S rows by minimising:
      - current row sum (favours balance)
      - co-occurrence count with already-selected rows in prior columns (avoids overlap)
    """
    matrix = np.zeros((D, T), dtype=np.int8)
    row_sums = np.zeros(D, dtype=np.int32)
    # pair_count[i, k]: number of existing columns that contain both row i and row k
    pair_count = np.zeros((D, D), dtype=np.int32)

    for col in range(T):
        available = np.ones(D, dtype=bool)
        selected: list[int] = []

        for _ in range(S):
            cands = np.where(available)[0]
            scores = row_sums[cands].astype(np.float64)
            for sel in selected:
                # Heavy penalty for pairs that already co-appear: adding this row
                # would create a column overlap >= 2 with some earlier column.
                scores += pair_count[sel, cands] * 1_000.0
            scores += rng.uniform(0.0, 0.01, len(cands))  # break ties randomly

            pick = int(cands[np.argmin(scores)])
            selected.append(pick)
            available[pick] = False

        for i in selected:
            matrix[i, col] = 1
            row_sums[i] += 1
        for i, k in combinations(selected, 2):
            pair_count[i, k] += 1
            pair_count[k, i] += 1

    return matrix


def _sa_improve(
    matrix: np.ndarray,
    D: int,
    T: int,
    S: int,
    rng: np.random.Generator,
    steps: int,
) -> np.ndarray:
    """
    Simulated-annealing refinement.

    Each step picks a random column and swaps one 1 with one 0 in that column,
    keeping column sums exactly fixed. Accepts moves that improve the objective
    or occasionally accepts worse moves (at high temperature) to escape local optima.

    Objective = overlap_violations * 1000 + variance(row_sums).
    """
    m = matrix.copy().astype(np.int8)
    row_sums = m.sum(axis=1).astype(np.int32)

    # ov[j1, j2] = number of rows shared between columns j1 and j2; diagonal is 0
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)

    T0, Tf = 5_000.0, 0.5
    decay = (Tf / T0) ** (1.0 / max(steps, 1))
    temp = T0

    for _ in range(steps):
        col = int(rng.integers(T))
        ones = np.where(m[:, col] == 1)[0]
        zeros = np.where(m[:, col] == 0)[0]
        if not len(ones) or not len(zeros):
            temp *= decay
            continue

        i_out = int(rng.choice(ones))
        i_in = int(rng.choice(zeros))

        # Swapping i_out -> i_in in column `col` changes ov[col, j] by:
        #   m[i_in, j] - m[i_out, j]   for each j != col
        dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
        dov[col] = 0  # self-overlap stays 0

        new_ov_col = ov[col, :] + dov
        viol_old = int(np.sum(np.maximum(0, ov[col, :] - 1)))
        viol_new = int(np.sum(np.maximum(0, new_ov_col - 1)))
        overlap_delta = viol_new - viol_old

        rs = row_sums.copy()
        rs[i_out] -= 1
        rs[i_in] += 1
        balance_delta = float(np.var(rs)) - float(np.var(row_sums))

        delta = overlap_delta * 1_000.0 + balance_delta

        if delta <= 0.0 or rng.random() < np.exp(-delta / temp):
            m[i_out, col] = 0
            m[i_in, col] = 1
            row_sums = rs
            ov[col, :] = new_ov_col
            ov[:, col] = new_ov_col

        temp *= decay

    return m


def _balance_fix(matrix: np.ndarray, T: int, max_iters: int = 1000) -> np.ndarray:
    """
    Greedy balance pass: repeatedly swap a 1 from a high-degree neuron to a
    low-degree neuron, accepting only moves that do not increase overlap violations.

    Stops when no improving swap exists (converged) or max_iters is reached.
    """
    m = matrix.copy()
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)

    for _ in range(max_iters):
        best_gain = 1  # require gain >= 2 to guarantee variance decreases
        best_swap = None

        for col in range(T):
            ones = np.where(m[:, col] == 1)[0]
            zeros = np.where(m[:, col] == 0)[0]
            if not len(ones) or not len(zeros):
                continue

            # Highest-sum rows that are currently 1, lowest-sum rows that are 0
            top_ones = ones[np.argsort(row_sums[ones])[::-1]][:3]
            bot_zeros = zeros[np.argsort(row_sums[zeros])][:3]

            for i_out in top_ones:
                for i_in in bot_zeros:
                    gain = int(row_sums[i_out]) - int(row_sums[i_in])
                    if gain <= best_gain:
                        continue
                    dov = m[i_in, :].astype(np.int32) - m[i_out, :].astype(np.int32)
                    dov[col] = 0
                    new_ov_col = ov[col, :] + dov
                    viol_delta = (int(np.sum(np.maximum(0, new_ov_col - 1))) -
                                  int(np.sum(np.maximum(0, ov[col, :] - 1))))
                    if viol_delta <= 0:
                        best_gain = gain
                        best_swap = (col, int(i_out), int(i_in), new_ov_col.copy())

        if best_swap is None:
            break

        col, i_out, i_in, new_ov_col = best_swap
        m[i_out, col] = 0
        m[i_in, col] = 1
        row_sums[i_out] -= 1
        row_sums[i_in] += 1
        ov[col, :] = new_ov_col
        ov[:, col] = new_ov_col

    return m


def _total_score(m: np.ndarray) -> float:
    row_sums = m.sum(axis=1).astype(np.int32)
    ov = (m.T.astype(np.int32) @ m.astype(np.int32))
    np.fill_diagonal(ov, 0)
    violations = int(np.sum(np.maximum(0, ov - 1))) // 2
    return violations * 1_000.0 + float(np.var(row_sums))


# ── 2. Building the model from a fact set ─────────────────────────────────────
# Given a fact set and a connection matrix, build up_matrix (the embedding) and
# down_matrix (the unembedding). See the post's "Embedding weights" and
# "Unembedding weights" sections.

import torch
import torch.nn.functional as F
from models import generate_facts


class HandCodedModel2Settings:
    """
    An object with all the settings needed to create the hand coded model
    """

    def __init__(
        self,
        input_vocab_size: int = 32,
        output_vocab_size: int = 16,
        n_facts: int = 16,
        seed: int = 42,
        d_ff: int = 16,
        n_neurons_per_label: int = 3,
        use_top_n_or_top_fraction: str = 'top_fraction',
        top_n: int = 0,
        top_fraction: float = 0.2,
    ):
        self.seq_len = 2                   # each fact has exactly two input tokens (fixed)
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size      # number of labels
        self.n_facts = n_facts                          # number of facts
        self.seed = seed                                # random seed
        self.d_ff = d_ff                                # number of ReLU neurons (d_MLP in the post)
        self.n_neurons_per_label = n_neurons_per_label  # S: number of neurons assigned to each label

        # top_n and top_fraction is two diffren way of expressing the same setting. 
        # Either can be used, but not both. The experiments in the post uses top_fraction.
        self.use_top_n_or_top_fraction = use_top_n_or_top_fraction  # 'top_n' or 'top_fraction'
        self.top_n = top_n                # int -- only used if 'top_n'
        self.top_fraction = top_fraction  # float -- only used if 'top_fraction'


class HandCodedModel2:
    """Analytically constructed toy model for memorising (input, label) facts.

    Forward pass:
        x_enc  = [one_hot(x[:,0]), one_hot(x[:,1])]   # (batch, n_vocab*2)
        hidden = relu(x_enc @ up_matrix.T)            # (batch, hidden_dim)
        logits = hidden @ down_matrix.T               # (batch, n_labels)

    __init__ fills in up_matrix and down_matrix from the fact set and the
    connection matrix; the concrete weight values are explained at each matrix's
    construction there.

    Any n_facts is supported: generate_facts assigns labels as
    arange(n_facts) % output_vocab_size, so facts-per-label differ by at most 1.
    """

    def __init__(self, settings: HandCodedModel2Settings, 
                 precomputed_conn: Optional[np.ndarray] = None):
        self.settings = settings

        # generate_facts supports any n_facts; labels = arange(n_facts) % output_vocab_size
        self.facts = generate_facts(
            n_facts=settings.n_facts,
            input_len=2,
            input_vocab_size=settings.input_vocab_size,
            output_vocab_size=settings.output_vocab_size,
            seed=settings.seed,
        )

        inputs = self.facts['inputs']   # (n_facts, 2) int tensor
        labels = self.facts['targets']  # (n_facts,)   int tensor
        device = labels.device          # stay on whatever device generate_facts used

        S          = settings.n_neurons_per_label
        n_labels   = settings.output_vocab_size
        hidden_dim = settings.d_ff
        n_vocab    = settings.input_vocab_size

        # --- Connection matrix: shape (hidden_dim, n_labels) ---
        # conn[neuron, label] = 1  ↔  neuron is assigned to label.
        # make_connection_matrix guarantees exactly S ones per column (label), balanced
        # row sums, and minimal pairwise overlap between columns.
        # Use a pre-built matrix if supplied (can save a lot of runtime for repeated experiments).
        if precomputed_conn is not None:
            conn_np = precomputed_conn
        else:
            conn_np = make_connection_matrix(D=hidden_dim, T=n_labels, S=S, seed=settings.seed)
        conn = torch.tensor(conn_np, dtype=torch.float32, device=device)  # (hidden_dim, n_labels)

        # --- Up matrix: shape (hidden_dim, n_vocab * 2) ---
        # Weights from the flattened one-hot input [first_token | second_token] to each neuron.
        # Initial value 1: relu(1 + 1) = 2, so every neuron fires on every input by default.
        # We will selectively lower weights so each neuron is silent on the inputs of its assigned labels.
        mlp_up = torch.ones(hidden_dim, n_vocab * 2, device=device)

        for neuron in range(hidden_dim):
            # Identify which facts this neuron must stay silent on.
            # conn[neuron] is a (n_labels,) vector of 0/1; True positions are its assigned labels.
            neuron_guards  = conn[neuron].bool()    # (n_labels,) bool
            guarded_mask   = neuron_guards[labels]  # (n_facts,) bool — True for facts of assigned labels
            guarded_inputs = inputs[guarded_mask]   # (k, 2) — neuron must NOT fire for these

            if guarded_inputs.shape[0] == 0:
                # No current facts fall under this neuron's assigned labels; leave at 1.
                continue

            # Count how often each token appears at each position among these inputs.
            # High-frequency tokens are good candidates for suppression: setting one token to -1
            # silences the neuron on every such input that contains it (relu(-1+1)=0).
            unique_f, counts_f = torch.unique(guarded_inputs[:, 0], return_counts=True)
            unique_s, counts_s = torch.unique(guarded_inputs[:, 1], return_counts=True)

            # Shuffle before ranking so that ties among equal-count tokens are broken randomly.
            perm_f = torch.randperm(unique_f.shape[0], device=device)
            unique_f, counts_f = unique_f[perm_f], counts_f[perm_f]
            perm_s = torch.randperm(unique_s.shape[0], device=device)
            unique_s, counts_s = unique_s[perm_s], counts_s[perm_s]

            # How many top tokens are given weight -1 in each token possition.
            if settings.use_top_n_or_top_fraction == 'top_fraction':
                k_f = max(1, int(len(unique_f) * settings.top_fraction))
                k_s = max(1, int(len(unique_s) * settings.top_fraction))
            else:  # 'top_n'
                k_f = settings.top_n
                k_s = settings.top_n

            # Top tokens: weight → -1.  relu(-1 + 1) = 0 whenever that token appears.
            top_f = unique_f[torch.argsort(counts_f, descending=True)[:k_f]]
            top_s = unique_s[torch.argsort(counts_s, descending=True)[:k_s]]
            if top_f.numel() > 0:
                mlp_up[neuron, top_f] = -1
            if top_s.numel() > 0:
                mlp_up[neuron, top_s + n_vocab] = -1

            # Remaining inputs: not covered by any -1 token in position 0 OR position 1.
            # Without further action these inputs still activate the neuron (relu(1+1)=2).
            # Zero both weights so relu(0+0)=0 — neuron is silenced.
            if top_f.numel() > 0:
                covered_f = (guarded_inputs[:, 0].unsqueeze(1) == top_f.unsqueeze(0)).any(1)
            else:
                covered_f = torch.zeros(guarded_inputs.shape[0], dtype=torch.bool, device=device)

            if top_s.numel() > 0:
                covered_s = (guarded_inputs[:, 1].unsqueeze(1) == top_s.unsqueeze(0)).any(1)
            else:
                covered_s = torch.zeros(guarded_inputs.shape[0], dtype=torch.bool, device=device)

            # An input is "remaining" only if it is uncovered in BOTH positions.
            remaining = guarded_inputs[~covered_f & ~covered_s]
            if remaining.shape[0] > 0:
                mlp_up[neuron, remaining[:, 0]] = 0
                mlp_up[neuron, remaining[:, 1] + n_vocab] = 0

        # --- Down matrix: shape (n_labels, hidden_dim) ---
        # down[l, n] = -2 if neuron n is assigned to label l, else 0.
        #
        # logit[b, l] = sum_n (hidden[b,n] * down[l,n])
        #             = -2 * (number of firing neurons assigned to l)
        #
        # If an assigned neuron fires (wrong input for l): logit[l] ≤ -2  → strictly negative.
        # If NO assigned neuron fires (input IS a fact for l): logit[l] = 0, above every
        # other label's logit, so the argmax picks l.
        self.up_matrix   = mlp_up
        self.down_matrix = -2.0 * conn.T                          # (n_labels, hidden_dim)

    def forward(self, x):
        """Run the model on input pairs x of shape (batch, 2).

        Returns (logits, hidden) — both float tensors on the same device as x.
        """
        first  = F.one_hot(x[:, 0], num_classes=self.settings.input_vocab_size).float()
        second = F.one_hot(x[:, 1], num_classes=self.settings.input_vocab_size).float()
        x_enc  = torch.cat([first, second], dim=-1)               # (batch, n_vocab*2)
        hidden = torch.relu(x_enc @ self.up_matrix.T)             # (batch, hidden_dim)
        logits = hidden @ self.down_matrix.T                      # (batch, n_labels)
        return logits, hidden

    def evaluate(self):
        """Evaluate the model on all stored facts.

        Returns:
            accuracy: fraction of facts where the argmax label matches the target
            logits:   (n_facts, n_labels) float tensor
            hidden:   (n_facts, hidden_dim) float tensor
        """
        logits, hidden = self.forward(self.facts['inputs'])
        accuracy = self.facts['targets'].eq(logits.argmax(dim=1)).float().mean().item()
        return accuracy, logits, hidden






# ── Connection-matrix cache ───────────────────────────────────────────────────
# Building a connection matrix runs simulated annealing, which can be slow for
# large D or S.  The matrix depends only on (D, T, S, seed) — not on n_facts or
# top_n — so we cache it here and reuse it across all evaluations that share the
# same model shape.
_conn_cache: dict = {}


def get_conn_matrix(D: int, T: int, S: int, seed: int) -> np.ndarray:
    """Return a cached connection matrix, building it on first call."""
    key = (D, T, S, seed)
    if key not in _conn_cache:
        # Suppress the "T exceeds theoretical maximum" warning — we know overlap
        # violations may occur and the SA minimises them as best it can.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _conn_cache[key] = make_connection_matrix(D=D, T=T, S=S, seed=seed)
    return _conn_cache[key]


# %%
