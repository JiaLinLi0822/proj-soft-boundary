from __future__ import annotations
import numpy as np

class KiraEnvironment:

    def __init__(self, monkey_id: int, *, include_trump: bool = False, rng: np.random.Generator | None = None):
        
        self.monkey_id = int(monkey_id)
        self.include_trump = bool(include_trump)
        self.rng = np.random.default_rng() if rng is None else rng

        self.shape_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.probs = self.make_stim_prob(trump=self.include_trump)

        self.probA = self.probs.copy()
        self.probB = self.probs[::-1].copy()

        self.llr = np.array([-0.9, -0.7, -0.5, -0.3, 0.3, 0.5, 0.7, 0.9], dtype=float)

    @staticmethod
    def _ratio_10_llr(llr: np.ndarray) -> np.ndarray:
        """Compute 10**llr / (1 + 10**llr) without overflow."""
        x = np.asarray(llr, dtype=float) * np.log(10.0)
        out = np.empty_like(x, dtype=float)
        pos = x >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
        ex = np.exp(x[~pos])
        out[~pos] = ex / (1.0 + ex)
        return out
    
    def make_stim_prob(self, trump: bool = False) -> np.ndarray:

        if self.monkey_id == 1:
            stim_ratio = np.array([0, 1, 2, 4, 8], dtype=float)
        elif self.monkey_id == 2:
            stim_ratio = np.array([0.1, 1, 2, 4, 8], dtype=float)
        else:
            raise ValueError("monkey_id must be 1 or 2")

        llr = np.array([9999, 9, 7, 5, 3], dtype=float) / 10.0
        sampling_p = stim_ratio / stim_ratio.sum()
        p1 = sampling_p * self._ratio_10_llr(llr)
        p2 = sampling_p - p1
        pp1 = np.concatenate([p1, p2[::-1]])
        pR = pp1[::-1]

        if trump:
            shape_prob = np.r_[pR[1:5], 0.0, 0.0, pR[5:9]]
        else:
            shape_prob = np.r_[pR[1:9]]

        return shape_prob / shape_prob.sum()

    def sample_target(self) -> int:
        return int(self.rng.integers(0, 2))

    def sample_shape(self, target: int) -> int:
        return int(self.rng.choice(self.shape_ids, p=self.probA if target == 0 else self.probB))

    def sample_observation(self, target: int | None = None) -> tuple[int, int]:
        if target is None:
            target = self.sample_target()
        return int(target), self.sample_shape(int(target))


if __name__ == "__main__":

    env = KiraEnvironment(1)
    print(env.shape_ids)
    print(env.probA)
    print(env.probB)
    print(env.llr)

    for i in range(10):
        print(env.sample_shape(0))
