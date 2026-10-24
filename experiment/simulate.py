"""Generate RNN rollout data for behavior, policy, and representation analyses."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from environment import KiraEnvironment
from utils import shape_ids_to_input_indices


@dataclass(frozen=True)
class Trial:
    shape_ids: np.ndarray
    cum_loglr_path: np.ndarray
    choice: int | None
    decision_step: int | None
    correct_answer: int | None = None
    policy_path: np.ndarray | None = None
    hidden_path: np.ndarray | None = None


def normalize_policy_path(policy):
    probabilities = np.asarray(policy, dtype=float)
    probabilities = np.clip(probabilities[:, :3], 0.0, np.inf)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def simulate_model(model, *, monkey_id, n_trials, rollout_seed, include_trump=False, max_steps=200):
    
    torch.manual_seed(rollout_seed)
    rng = np.random.default_rng(rollout_seed)
    env = KiraEnvironment(monkey_id, include_trump=include_trump, rng=rng)
    device = next(model.parameters()).device
    trials = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_trials):
            correct_answer = env.sample_target()
            shape_indices = []
            policy_path = []
            hidden_path = []
            evidence_path = []
            evidence = 0.0
            hidden = None
            choice = None

            for _ in range(max_steps):
                shape_index = env.sample_shape(correct_answer) - 1
                shape_indices.append(shape_index)
                x = torch.zeros((1, 8), dtype=torch.float32, device=device)
                x[0, shape_index] = 1.0
                action, policy, _log_prob, _entropy, hidden = model(x, hidden)

                evidence += env.llr[shape_index]
                evidence_path.append(evidence)
                policy_path.append(policy.detach().cpu().numpy()[0])
                hidden_path.append(hidden.detach().cpu().numpy()[0])
                if int(action) != 2:
                    choice = 2 if int(action) == 0 else 1
                    break

            trials.append(
                Trial(
                    shape_ids=np.array([4, 6, 8, 10, 9, 7, 5, 3])[shape_indices],
                    cum_loglr_path=np.asarray(evidence_path),
                    choice=choice,
                    decision_step=len(shape_indices) if choice is not None else None,
                    correct_answer=correct_answer,
                    policy_path=np.asarray(policy_path),
                    hidden_path=np.asarray(hidden_path),
                )
            )
    return trials


def _prefix_permutations(indices, match_timestep, max_permutations, rng):
    prefix = np.asarray(indices[:match_timestep])
    suffix = np.asarray(indices[match_timestep:])
    counts = Counter(prefix)
    possible = math.factorial(len(prefix))
    for count in counts.values():
        possible //= math.factorial(count)

    sequences = []
    seen = set()

    def add(candidate):
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            sequences.append(np.concatenate([candidate, suffix]))

    add(prefix)
    if possible <= max_permutations:
        for candidate in itertools.permutations(prefix):
            add(np.asarray(candidate))
    else:
        while len(sequences) < max_permutations:
            add(rng.permutation(prefix))
    return sequences[:max_permutations], possible


def simulate_permuted_evidence(
    model,
    *,
    monkey_id,
    rollout_seed,
    permutation_seed,
    include_trump=False,
    sequence_length=3,
    match_timestep=None,
    target="random",
    shape_ids=None,
    shape_indices=None,
    max_permutations=128,
):
    env = KiraEnvironment(
        monkey_id,
        include_trump=include_trump,
        rng=np.random.default_rng(rollout_seed),
    )
    shape_values = np.array([4, 6, 8, 10, 9, 7, 5, 3])
    generated_target = None
    if shape_indices is not None:
        base_indices = np.asarray(shape_indices, dtype=int)
    elif shape_ids is not None:
        base_indices = np.asarray(shape_ids_to_input_indices(shape_ids), dtype=int)
    else:
        generated_target = env.sample_target() if target == "random" else int(target)
        base_indices = np.asarray(
            [env.sample_shape(generated_target) - 1 for _ in range(sequence_length)]
        )

    match_timestep = match_timestep or len(base_indices)
    sequences, possible = _prefix_permutations(
        base_indices,
        match_timestep,
        max_permutations,
        np.random.default_rng(permutation_seed),
    )
    device = next(model.parameters()).device
    hidden_rows = []
    hidden_paths = []

    model.eval()
    with torch.no_grad():
        for sequence_id, sequence in enumerate(sequences):
            hidden = None
            path = []
            cumulative_evidence = 0.0
            for time_step, index in enumerate(sequence, start=1):
                x = torch.zeros((1, 8), dtype=torch.float32, device=device)
                x[0, index] = 1.0
                _action, policy, _log_prob, _entropy, hidden = model(x, hidden)
                state = hidden.detach().cpu().numpy()[0]
                probabilities = normalize_policy_path(policy.detach().cpu().numpy())[0]
                cumulative_evidence += env.llr[index]
                row = {
                    "sequence_id": sequence_id,
                    "time_step": time_step,
                    "input_index": index,
                    "shape_id": int(shape_values[index]),
                    "evidence": env.llr[index],
                    "cumulative_logLR": cumulative_evidence,
                    "p_choose_a": probabilities[0],
                    "p_choose_b": probabilities[1],
                    "p_continue": probabilities[2],
                }
                row.update({f"h{i}": value for i, value in enumerate(state)})
                hidden_rows.append(row)
                path.append(state)
            hidden_paths.append(path)

    sequence_rows = []
    for sequence_id, sequence in enumerate(sequences):
        evidence = env.llr[sequence]
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "input_indices": ",".join(map(str, sequence)),
                "shape_ids": ",".join(map(str, shape_values[sequence])),
                "matched_cumulative_logLR": evidence[:match_timestep].sum(),
                "final_cumulative_logLR": evidence.sum(),
            }
        )

    return {
        "hidden_df": pd.DataFrame(hidden_rows),
        "hidden_paths": np.asarray(hidden_paths),
        "sequence_df": pd.DataFrame(sequence_rows),
        "metadata": {
            "rollout_seed": rollout_seed,
            "permutation_seed": permutation_seed,
            "generated_target": generated_target,
            "sequence_length": len(base_indices),
            "match_timestep": match_timestep,
            "n_unique_prefix_permutations_possible": possible,
            "n_sequences_run": len(sequences),
            "base_input_indices": base_indices.tolist(),
            "base_shape_ids": shape_values[base_indices].tolist(),
        },
    }
