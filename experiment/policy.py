"""RNN policy processing and plots."""

from collections import defaultdict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from plotting_style import set_style, _save
from simulate import normalize_policy_path



def collect_psample_observations(trials, max_time_step=None):
    rows = []
    for trial_index, trial in enumerate(trials):
        if trial.policy_path is None:
            continue
        policy = normalize_policy_path(trial.policy_path)
        steps = min(len(policy), len(trial.cum_loglr_path), max_time_step or len(policy))
        for time_step in range(steps):
            row = {
                "trial_index": trial_index,
                "time_step": time_step + 1,
                "p_sample": policy[time_step, 2],
                "cumulative_logLR": trial.cum_loglr_path[time_step],
            }
            if trial.hidden_path is not None:
                row.update(
                    {f"h{i}": value for i, value in enumerate(trial.hidden_path[time_step])}
                )
            rows.append(row)
    return pd.DataFrame(rows)


def select_highlight_trial(observations, trial_index=-1, preferred_length=-1):
    lengths = observations.groupby("trial_index").size()
    if trial_index >= 0:
        selected = trial_index
    elif preferred_length >= 1:
        selected = min(lengths.index, key=lambda i: abs(lengths[i] - preferred_length))
    else:
        selected = lengths.idxmax()
    return observations[observations["trial_index"] == selected].sort_values("time_step")


def trials_to_policy_points(trials, timesteps, action=2):
    points = []
    for trial_index, trial in enumerate(trials):
        if trial.policy_path is None:
            continue
        for timestep in timesteps:
            if timestep <= len(trial.cum_loglr_path):
                points.append(
                    {
                        "trial_index": trial_index,
                        "timestep": timestep,
                        "cum_loglr": trial.cum_loglr_path[timestep - 1],
                        "p_sample": trial.policy_path[timestep - 1, action],
                    }
                )
    return points


def bin_by_timestep(points, n_bins, min_bin_count, bin_start=None, bin_end=None):
    evidence = np.asarray([point["cum_loglr"] for point in points])
    if bin_start is None:
        limit = max(np.percentile(np.abs(evidence), 99), 1e-3)
        bin_start, bin_end = -limit, limit
    edges = np.linspace(bin_start, bin_end, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    buckets = defaultdict(lambda: defaultdict(list))
    for point in points:
        index = np.digitize(point["cum_loglr"], edges) - 1
        if 0 <= index < n_bins:
            buckets[point["timestep"]][index].append(point["p_sample"])

    output = {}
    for timestep, bins in buckets.items():
        mean = np.full(n_bins, np.nan)
        sem = np.full(n_bins, np.nan)
        count = np.zeros(n_bins, dtype=int)
        for index, values in bins.items():
            count[index] = len(values)
            if len(values) >= min_bin_count:
                mean[index] = np.mean(values)
                sem[index] = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        output[timestep] = {"x": centers, "mean": mean, "se": sem, "count": count}
    return output


def plot_psample(trials, out_path, timesteps=range(1, 11), n_bins=30, min_bin_count=5, bin_start=None, bin_end=None, figsize=(3.6, 2.8)):
    
    binned = bin_by_timestep(trials_to_policy_points(trials, timesteps), n_bins, min_bin_count, bin_start, bin_end)
    set_style()
    fig, ax = plt.subplots(figsize=figsize)
    timesteps = sorted(binned)
    norm = Normalize(min(timesteps), max(timesteps))
    for timestep in timesteps:
        values = binned[timestep]
        valid = np.isfinite(values["mean"])
        color = plt.cm.plasma(norm(timestep))
        ax.plot(values["x"][valid], values["mean"][valid], color=color, lw=2)
        ax.fill_between(
            values["x"][valid],
            np.clip(values["mean"][valid] - values["se"][valid], 0, 1),
            np.clip(values["mean"][valid] + values["se"][valid], 0, 1),
            color=color,
            alpha=0.16,
        )
    ax.axvline(0, color="0.4", ls="--")
    ax.set(xlabel="Cumulative evidence (log odds)", xlim=(-4, 4), xticks=[-4, -2, 0, 2, 4], ylim=(-0.03, 1.03))
    ax.set_ylabel(r"$p(\mathrm{sample})$", math_fontfamily="stix", fontsize=14)
    colorbar = ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    fig.colorbar(colorbar, ax=ax, pad=0.02).set_label("Timestep")
    fig.tight_layout()
    _save(fig, out_path)


def plot_psample_scatter(trials, out_path, max_time_step=10, point_size=5, point_alpha=1, highlight_trial=-1, highlight_length=-1):
    observations = collect_psample_observations(trials, max_time_step)
    highlight = select_highlight_trial(observations, highlight_trial, highlight_length)
    set_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.scatter(
        observations["cumulative_logLR"],
        observations["p_sample"],
        color="0.8",
        s=point_size,
        alpha=point_alpha,
    )
    norm = Normalize(1, max_time_step)
    ax.plot(highlight["cumulative_logLR"], highlight["p_sample"], color="0.35")
    trajectory = highlight[["cumulative_logLR", "p_sample"]].to_numpy()
    for start, end in zip(trajectory[:-1], trajectory[1:]):
        ax.annotate(
            "", xy=start + 0.7 * (end - start),
            xytext=start + 0.35 * (end - start),
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.0,
                            mutation_scale=8, shrinkA=0, shrinkB=0),
            zorder=2,
        )
    points = ax.scatter(
        highlight["cumulative_logLR"],
        highlight["p_sample"],
        c=highlight["time_step"],
        cmap="plasma",
        norm=norm,
        s=42,
        edgecolors="none",
        zorder=3,
    )
    ax.set(xlabel="Cumulative evidence (log odds)", xlim=(-4, 4), xticks=[-4, -2, 0, 2, 4], ylim=(-0.03, 1.03))
    ax.set_ylabel(r"$p(\mathrm{sample})$", math_fontfamily="stix", fontsize=14)
    fig.colorbar(points, ax=ax, pad=0.02).set_label("Timestep")
    fig.tight_layout()
    _save(fig, out_path)


def plot_sampling_variability(trials, out_path, timesteps=range(1, 11), evidence_range=4, evidence_bin_width=0.1, min_bin_count=10, title=None):
    timesteps = list(timesteps)
    time_ticks = np.arange(0, timesteps[-1] + 1, 2)
    centers = np.arange(-evidence_range, evidence_range + evidence_bin_width / 2, evidence_bin_width)
    edges = np.r_[centers - evidence_bin_width / 2, centers[-1] + evidence_bin_width / 2]
    values = np.full((len(timesteps), len(centers)), np.nan)
    buckets = defaultdict(list)
    for point in trials_to_policy_points(trials, timesteps):
        index = np.digitize(point["cum_loglr"], edges) - 1
        if 0 <= index < len(centers):
            buckets[point["timestep"], index].append(point["p_sample"])
    for (timestep, index), samples in buckets.items():
        if len(samples) >= min_bin_count:
            values[timesteps.index(timestep), index] = np.std(samples, ddof=1)

    mean = np.nanmean(values, axis=1)
    count = np.sum(np.isfinite(values), axis=1)
    sem = np.divide(
        np.nanstd(values, axis=1, ddof=1),
        np.sqrt(count),
        out=np.zeros_like(mean),
        where=count > 1,
    )

    set_style()
    fig = plt.figure(figsize=(3.55, 5.0), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.055], hspace=0.20)
    axes = [fig.add_subplot(grid[0])]
    axes.append(fig.add_subplot(grid[1], sharex=axes[0]))
    colorbar_ax = fig.add_subplot(grid[2])
    axes[0].errorbar(
        timesteps, mean, yerr=sem,
        fmt="o", color="black", lw=2, capsize=3,
    )
    valid = np.isfinite(mean)
    if valid.sum() >= 2:
        fit_x = np.asarray(timesteps)[valid]
        slope, intercept = np.polyfit(fit_x, mean[valid], 1)
        ax_x = np.array([fit_x.min(), fit_x.max()])
        axes[0].plot(ax_x, intercept + slope * ax_x, color="black", lw=2, ls="--")
    axes[0].set(
        xlabel="Time step", ylabel=r"Variability of $p(\mathrm{sample})$",
        xlim=(0, timesteps[-1] + 0.5), xticks=time_ticks,
    )
    axes[0].yaxis.label.set_math_fontfamily("custom")
    heatmap = axes[1].pcolormesh(
        np.r_[np.asarray(timesteps) - 0.5, timesteps[-1] + 0.5],
        edges,
        values.T,
        cmap="viridis",
    )
    axes[1].set(
        xlabel="Time step", ylabel="Cumulative evidence (log odds)",
        ylim=(-3.2, 3.2),
    )
    fig.colorbar(
        heatmap, cax=colorbar_ax, orientation="horizontal",
        label=r"Variability of $p(\mathrm{sample})$",
    )
    colorbar_ax.xaxis.label.set_math_fontfamily("custom")
    colorbar_ax.xaxis.set_ticks_position("top")
    colorbar_ax.xaxis.set_label_position("top")
    if title:
        fig.suptitle(title)
    with plt.rc_context({
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
    }):
        _save(fig, out_path)
