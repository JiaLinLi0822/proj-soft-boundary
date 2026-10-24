import numpy as np
import h5py


def load_mat73(file_path):
    """Load MATLAB v7.3 .mat file and return a nested dict of numpy arrays."""

    def load_group(group):
        result = {}
        for key in group.keys():
            if key == "#refs#":
                continue
            obj = group[key]
            if isinstance(obj, h5py.Dataset):
                result[key] = np.array(obj)
            elif isinstance(obj, h5py.Group):
                result[key] = load_group(obj)
        return result

    data = {}
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            if key == "#refs#":
                continue
            if isinstance(f[key], h5py.Group):
                data[key] = load_group(f[key])
            else:
                data[key] = np.array(f[key])
    return data


def set_seed(seed):
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device):
    """Resolve 'auto' to cuda, then mps, else cpu. Otherwise parse the string."""
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def shape_ids_to_input_indices(shape_ids):
    """Map shape ids in {3,4,5,6,7,8,9,10} -> input indices 0..7.

    Shape ids outside the canonical set are mapped to -1.
    Accepts a scalar or array; returns the same shape.
    """
    lookup = np.array([-1, -1, -1, 7, 0, 6, 1, 5, 2, 4, 3])
    shape_ids = np.asarray(shape_ids, dtype=int)
    out = np.full(shape_ids.shape, -1, dtype=int)
    valid = (shape_ids >= 0) & (shape_ids < len(lookup))
    out[valid] = lookup[shape_ids[valid]]
    if out.ndim == 0:
        return int(out.item())
    return out
