from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    """
    A torch Categorical class with action masking.
    """

    def __init__(self, logits, mask):
        self.mask = mask

        if mask is None:
            super(CategoricalMasked, self).__init__(logits = logits)
        else:
            self.mask_value = torch.tensor(
                torch.finfo(logits.dtype).min, dtype = logits.dtype
            )
            logits = torch.where(self.mask, logits, self.mask_value)
            super(CategoricalMasked, self).__init__(logits = logits)


    def entropy(self):
        if self.mask is None:
            return super().entropy()
        
        p_log_p = self.logits * self.probs

        # compute entropy with possible actions only
        p_log_p = torch.where(
            self.mask,
            p_log_p,
            torch.tensor(0, dtype = p_log_p.dtype, device = p_log_p.device),
        )

        return -torch.sum(p_log_p, axis = 1)


class DiagonalLinear(nn.Module):
    """
    Sparse readout: for i < min(in_features, out_features), y_i = w_i * x_i (+ bias);
    remaining outputs are bias-only. Fewer parameters than a full Linear(hidden -> actions).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        n_diag = min(self.in_features, self.out_features)
        self.diag_weight = nn.Parameter(torch.empty(n_diag))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        lim = 1.0 / math.sqrt(max(self.in_features, 1))
        nn.init.uniform_(self.diag_weight, -lim, lim)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -lim, lim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = min(self.in_features, self.out_features)
        out = x.new_zeros(*x.shape[:-1], self.out_features)
        out[..., :n] = self.diag_weight * x[..., :n]
        if self.bias is not None:
            out = out + self.bias
        return out


class TinyGRUPolicy(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_actions: int,
        *,
        diagonal_readout: bool = False,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_actions = int(num_actions)
        self.diagonal_readout = bool(diagonal_readout)

        self.gru = nn.GRUCell(self.input_dim, self.hidden_dim)
        if self.diagonal_readout:
            self.readout = DiagonalLinear(self.hidden_dim, self.num_actions)
        else:
            self.readout = nn.Linear(self.hidden_dim, self.num_actions)

    def forward(self, x, states_gru = None, mask = None):

        if states_gru is None:
            states_gru = torch.zeros(x.size(0), self.hidden_dim, device = x.device)
        else:
            states_gru = states_gru

        hidden = self.gru(x, states_gru)
        logits = self.readout(hidden)

        if mask is None:
            dist = Categorical(logits = logits)
        else:
            dist = CategoricalMasked(logits = logits, mask = mask)

        action = dist.sample()
        policy = dist.probs
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, policy, log_prob, entropy, hidden
        

if __name__ == "__main__":
    torch.manual_seed(0)

    input_dim = 8
    hidden_dim = 2
    num_actions = 3
    model = TinyGRUPolicy(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)

    # Test one full sequence by rolling the GRU hidden state forward in time.
    seq_idx = torch.tensor([1, 2, 4, 5, 5, 3], dtype=torch.long)
    seq_len = int(seq_idx.numel())
    x_seq = F.one_hot(seq_idx, num_classes=input_dim).to(dtype=torch.float32)

    hidden = None
    print("Testing a single sequence through TinyGRUPolicy")
    print(f"sequence input indices: {seq_idx.tolist()}")
    for t in range(seq_len):
        x_t = x_seq[t : t + 1]
        action, policy, log_prob, entropy, hidden = model(x_t, hidden)
        print(
            f"t={t:02d} x={seq_idx[t].item()} "
            f"action={action.item()} "
            f"policy={policy.squeeze(0).tolist()} "
            f"log_prob={log_prob.item():.4f} "
            f"entropy={entropy.item():.4f} "
            f"hidden={hidden.squeeze(0).tolist()}"
        )

