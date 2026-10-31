"""Generate RNN rollout data for behavior, policy, and representation analyses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from environment import KiraEnvironment


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
