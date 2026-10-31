"""Plot outer-test NLL and save independent-samples Student's t-test results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from plotting_style import _save, set_style


def plot_test_trajectory_nll_by_hidden(summary, out_path):
    set_style()
    
    hidden = np.asarray(sorted(summary["hidden_dim"].unique()))
    groups = [summary[summary["hidden_dim"] == size] for size in hidden]
    values = [group["test_trajectory_nll"].to_numpy() for group in groups]

    fig, ax = plt.subplots(figsize=(3.05, 2.55))
    ax.set_xscale("log", base=2)
    ax.set_box_aspect(1)
    rng = np.random.default_rng(0)
    for size, losses in zip(hidden, values):
        ax.scatter(
            size * 2 ** rng.uniform(-0.035, 0.035, len(losses)),
            losses,
            color="0.5", s=18, alpha=0.5,
        )
    ax.errorbar(
        hidden,
        [
            np.average(losses, weights=group["test_size"])
            for losses, group in zip(values, groups)
        ],
        yerr=[losses.std() for losses in values],
        fmt="o-", color="black", capsize=3,
    )
    ax.set(
        xlabel="Number of hidden units",
        ylabel="Negative log-likelihood",
    )
    ax.set_xticks(hidden, labels=[str(size) for size in hidden])
    ax.minorticks_off()
    fig.tight_layout()
    _save(fig, str(out_path), bbox_inches=None)


def pairwise_ttests(summary, alpha=0.05):
    """Compare hidden sizes using two-sided, equal-variance independent t-tests."""
    if summary.fold_seed.nunique() != 1:
        raise ValueError("Analyze one fold seed at a time; repeated splits are not independent.")
    losses = summary.pivot(
        index="outer_fold", columns="hidden_dim", values="test_trajectory_nll"
    ).sort_index(axis=1)
    expected_folds = summary.outer_splits.unique()
    if (
        len(expected_folds) != 1
        or set(losses.index) != set(range(int(expected_folds[0])))
        or len(losses) < 2
        or not np.isfinite(losses.to_numpy()).all()
    ):
        raise ValueError("Every hidden size must have finite NLL for all matching outer folds.")
    sizes = summary.pivot(index="outer_fold", columns="hidden_dim", values="test_size")
    if not sizes.eq(sizes.iloc[:, 0], axis=0).all().all():
        raise ValueError("Models must use matching test sizes within each outer fold.")

    comparisons = []
    for smaller in losses:
        for larger in losses.columns[losses.columns > smaller]:
            difference = losses[smaller] - losses[larger]
            if (difference == 0).all():
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = ttest_ind(
                    losses[smaller], losses[larger], equal_var=True, alternative="two-sided"
                )
            comparisons.append({
                "smaller_h": smaller,
                "larger_h": larger,
                "n_smaller": len(losses),
                "n_larger": len(losses),
                "df": 2 * len(losses) - 2,
                "mean_nll_reduction": difference.mean(),
                "t_statistic": statistic,
                "p_value": p_value,
                "larger_significantly_better": difference.mean() > 0 and p_value < alpha,
            })
    comparisons = pd.DataFrame(comparisons, columns=[
        "smaller_h", "larger_h", "n_smaller", "n_larger", "df", "mean_nll_reduction",
        "t_statistic", "p_value", "larger_significantly_better",
    ])
    return comparisons


def main():
    root = Path(__file__).resolve().parent / "runs" / "nested_cv"
    paths = sorted(root.glob("slurm_*/h*/outer*/Monkey*/nestedcv_*/summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No nested-CV summaries found in {root}")
    summary = pd.concat(
        [pd.read_csv(path).assign(source_summary=str(path)) for path in paths],
        ignore_index=True,
    )
    # Directory timestamps select the latest rerun of each fold.
    keys = ["monkey_id", "fold_seed", "hidden_dim", "outer_fold"]
    summary["run_timestamp"] = summary.source_summary.map(
        lambda path: Path(path).parent.name
    )
    summary = summary.sort_values("run_timestamp").drop_duplicates(keys, keep="last")
    assert summary.test_trajectory_nll.notna().all(), "Missing outer-test NLL"
    assert (summary.test_size > 0).all(), "Invalid test sizes"
    outdir = root / "plots"
    outdir.mkdir(exist_ok=True)
    for monkey_id, group in summary.groupby("monkey_id"):
        monkey = {1: "MonkeyE", 2: "MonkeyJ"}[monkey_id]
        csv_path = outdir / f"{monkey}_pairwise_ttests.csv"
        pairwise_ttests(group).to_csv(csv_path, index=False)
        output = outdir / f"{monkey}_test_trajectory_nll_vs_hidden"
        plot_test_trajectory_nll_by_hidden(group, output)
        print(f"Saved {output}.png and .svg")
        print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
