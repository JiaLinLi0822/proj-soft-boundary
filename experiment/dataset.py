from dataclasses import dataclass

import numpy as np
import torch

from utils import load_mat73, shape_ids_to_input_indices

@dataclass(frozen=True)
class TrialSeq:
    x: np.ndarray
    y: np.ndarray


def collate_trialseq_batch(batch):
    lengths = torch.tensor([len(trial.x) for trial in batch])
    steps = int(lengths.max())
    x = torch.zeros((len(batch), steps, batch[0].x.shape[1]))
    y = torch.full((len(batch), steps), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), steps))

    for i, trial in enumerate(batch):
        length = len(trial.x)
        x[i, :length] = torch.from_numpy(trial.x)
        y[i, :length] = torch.from_numpy(trial.y)
        mask[i, :length] = 1

    return x, y, lengths, mask


class KiraDataset(list):
    def __init__(self, mat_path, monkey_id, *, use_num_accum=True, remove_trump_trials=True, last_accum_override=None):
        if monkey_id not in (1, 2):
            raise ValueError("monkey_id must be 1 or 2")

        raw = load_mat73(mat_path)
        if "info" not in raw or "TM" not in raw["info"]:
            raise KeyError("Expected MATLAB structure with raw['info']['TM']")
        TM = np.asarray(raw["info"]["TM"], dtype=float).T
        shapeMtrx = TM[:, 10:30] + 1

        if remove_trump_trials:
            keep = ~np.sum((shapeMtrx == 1) | (shapeMtrx == 2), axis=1).astype(bool)
            TM = TM[keep, :]
            shapeMtrx = shapeMtrx[keep, :]

        # monkey 1 (Eli): saccade dir col 1; monkey 2 (Joey): chosen color col 9.
        if monkey_id == 1:
            choice = TM[:, 1].astype(int)
        elif monkey_id == 2:
            choice = TM[:, 9].astype(int)
        else:
            raise ValueError("monkey_id must be 1 or 2")

        
        lengths = np.sum(np.isfinite(shapeMtrx), axis=1).astype(int)
        
        if use_num_accum:

            if monkey_id == 1:
                last_accum, stimulus_interval = (270, 250)
            elif monkey_id == 2:
                last_accum, stimulus_interval = (180, 270)
            else:
                raise ValueError("monkey_id must be 1 or 2")

            if last_accum_override is not None:
                last_accum = last_accum_override
            cutoff = last_accum - TM[:, 3]
            lengths[cutoff > 0] -= 1
            while np.any(cutoff > 0):
                cutoff -= stimulus_interval
                lengths[cutoff > 0] -= 1
            lengths = np.maximum(lengths, 0)

        seqs = []
        for shape_row, length, selected_choice in zip(shapeMtrx, lengths, choice):
            length = int(length)
            if length <= 0:
                continue
            shapes = shape_row[:length]
            if not np.all(np.isfinite(shapes)):
                continue

            idx = shape_ids_to_input_indices(shapes.astype(int))
            if np.any(idx < 0):
                continue

            x = np.zeros((length, 8), dtype=np.float32)
            x[np.arange(length), idx] = 1
            y = np.full(length, 2, dtype=np.int64)
            y[-1] = 0 if selected_choice == 2 else 1
            seqs.append(TrialSeq(x, y))

        super().__init__(seqs)
        self.shapeMtrx = shapeMtrx
        self.choice = choice
        self.lengths = lengths


def split_dataset(dataset, *, val_frac, seed):
    val_frac = float(val_frac)
    if not 0 <= val_frac <= 1:
        raise ValueError("val_frac must be between 0 and 1.")

    labels = np.asarray([int(seq.y[-1]) for seq in dataset], dtype=int)
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = round(len(indices) * val_frac)
        if 0 < val_frac < 1 and len(indices) > 1:
            count = min(max(count, 1), len(indices) - 1)
        val_idx.extend(indices[:count])
        train_idx.extend(indices[count:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    meta = {
        "n_total": len(dataset),
        "n_train": len(train_data),
        "n_val": len(val_data),
        "val_frac": val_frac,
        "stratified_by_final_choice": True,
    }
    return train_data, val_data, meta


def stratified_kfold(labels, k, seed):
    k = int(k)
    if k < 2:
        raise ValueError("k must be >= 2")
    labels = np.asarray(labels, dtype=int)
    if len(labels) < k:
        n = len(labels)
        raise ValueError(f"k={k} cannot exceed number of samples n={n}.")

    rng = np.random.default_rng(seed)
    chunks = {}
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        chunks[label] = np.array_split(indices, k)

    all_indices = np.arange(len(labels))
    return [
        (np.setdiff1d(all_indices, val), val)
        for fold in range(k)
        for val in [np.concatenate([parts[fold] for parts in chunks.values()])]
    ]
