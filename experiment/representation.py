"""RNN hidden-state and representation-noise plots."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from plotting_style import set_style, _save

def trials_to_hidden_df(trials, cum_col="cumulative_logLR"):
    return pd.DataFrame(
        [
            {
                "episode": episode,
                "time_step": timestep + 1,
                "shape_id": trial.shape_ids[timestep],
                cum_col: trial.cum_loglr_path[timestep],
                **{f"h{i}": value for i, value in enumerate(state)},
            }
            for episode, trial in enumerate(trials)
            if trial.hidden_path is not None
            for timestep, state in enumerate(trial.hidden_path)
        ]
    )


def compute_representation_noise(
    trials, max_time_step=10, evidence_bin_width=0.2, min_count=10,
    cum_col="cumulative_logLR",
):
    if not np.isfinite(evidence_bin_width) or evidence_bin_width <= 0:
        raise ValueError("evidence_bin_width must be positive and finite")
    frame = trials_to_hidden_df(trials, cum_col)
    hidden_cols = [column for column in frame if column.startswith("h")]
    frame = frame[frame["time_step"] <= max_time_step].copy()
    # Half-open bins centered on multiples of the bin width; stabilize boundary rounding.
    bin_index = np.floor(np.round(frame[cum_col] / evidence_bin_width + 0.5, 12))
    frame["evidence_level"] = (bin_index * evidence_bin_width).round(12)

    rows = []
    for (timestep, evidence), group in frame.groupby(
        ["time_step", "evidence_level"]
    ):
        hidden = group[hidden_cols].to_numpy()
        if len(hidden) < min_count:
            continue
        variance = hidden.var(axis=0, ddof=1 if len(hidden) > 1 else 0)
        rows.append({
            "time_step": timestep,
            "evidence_level": evidence,
            "count": len(hidden),
            "variability": np.sqrt(variance.sum()),
        })
    noise = pd.DataFrame(rows)

    summary = []
    for timestep, group in noise.groupby("time_step"):
        values = group["variability"].to_numpy()
        summary.append({
            "time_step": timestep,
            "variability": np.average(values, weights=group["count"]),
            "sem": values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0,
        })
    return noise, pd.DataFrame(summary)


def plot_representation_noise(noise, summary, out_path, evidence_bin_width=0.2):
    set_style()
    timesteps = np.arange(1, 11)
    evidence = np.round(np.arange(-3.2, 3.2 + evidence_bin_width / 2, evidence_bin_width), 12)
    pivot = noise.pivot(
        index="evidence_level", columns="time_step", values="variability"
    ).reindex(index=evidence, columns=timesteps)

    fig = plt.figure(figsize=(3.55, 5.0), layout="constrained")
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.055], hspace=0.20)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    colorbar_ax = fig.add_subplot(grid[2])
    top.errorbar(
        summary["time_step"], summary["variability"], yerr=summary["sem"],
        fmt="o", color="black", lw=2, capsize=3,
    )
    valid = np.isfinite(summary["variability"])
    if valid.sum() >= 2:
        fit_x = summary.loc[valid, "time_step"].to_numpy()
        slope, intercept = np.polyfit(fit_x, summary.loc[valid, "variability"], 1)
        endpoints = np.array([fit_x.min(), fit_x.max()])
        top.plot(endpoints, intercept + slope * endpoints, "--", color="black", lw=2)
    top.set(
        xlabel="Time step", ylabel="Hidden-state variability",
        xlim=(0, 10.5), xticks=np.arange(0, 11, 2),
    )
    image = bottom.pcolormesh(
        np.r_[timesteps - 0.5, timesteps[-1] + 0.5],
        np.r_[evidence - evidence_bin_width / 2, evidence[-1] + evidence_bin_width / 2],
        pivot.to_numpy(), cmap="viridis",
    )
    bottom.set(
        xlabel="Time step", ylabel="Cumulative evidence (log odds)",
        ylim=(-3.2, 3.2),
    )
    fig.colorbar(
        image, cax=colorbar_ax, orientation="horizontal",
        label="Hidden-state variability",
    )
    colorbar_ax.xaxis.set_ticks_position("top")
    colorbar_ax.xaxis.set_label_position("top")
    _save(fig, out_path)


def _plot_hidden(trials, out_path, color_column, cmap, colorbar_label, max_time_step, cum_col, xlim, ylim):
    
    frame = trials_to_hidden_df(trials, cum_col)
    frame = frame[frame["time_step"] <= max_time_step]
    colors = frame[color_column].to_numpy()
    if color_column == cum_col:
        limit = max(abs(colors.min()), abs(colors.max())) or 1
        norm = Normalize(-limit, limit)
    else:
        norm = Normalize(colors.min(), colors.max())

    set_style()
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.set_box_aspect(1)
    
    segments = []
    segment_colors = []
    
    for _, episode in frame.groupby("episode"):
        episode = episode.sort_values("time_step")
        points = episode[["h0", "h1"]].to_numpy()
        values = episode[color_column].to_numpy()
        segments.extend(np.stack([points[:-1], points[1:]], axis=1))
        segment_colors.extend(plt.get_cmap(cmap)(norm(values[1:])))
    ax.add_collection(
        LineCollection(
            segments, colors=segment_colors, linewidths=0.6, alpha=0.15,
            rasterized=True,
        )
    )
    scatter = ax.scatter(
        frame["h0"], frame["h1"], c=colors, cmap=cmap, norm=norm,
        s=12, alpha=0.8, edgecolors="none", rasterized=True,
    )

    if xlim is None:
        padding = 0.03 * max(frame["h0"].max() - frame["h0"].min(), 1e-6)
        xlim = frame["h0"].min() - padding, frame["h0"].max() + padding
    
    if ylim is None:
        padding = 0.03 * max(frame["h1"].max() - frame["h1"].min(), 1e-6)
        ylim = frame["h1"].min() - padding, frame["h1"].max() + padding
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xlabel("Hidden unit 1 activity", fontsize=14)
    ax.set_ylabel("Hidden unit 2 activity", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    # Lines + scatter become one embedded bitmap in SVG (axes/colorbar stay vector).
    ax.set_rasterization_zorder(2)
    
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    _save(fig, out_path)


def plot_hidden_by_timestep(trials, out_path, cum_col="cumulative_logLR", max_time_step=15, xlim=None, ylim=None):
    _plot_hidden(trials, out_path, "time_step", "plasma", "Time step", max_time_step, cum_col, xlim, ylim)


def plot_hidden_by_evidence(trials, out_path, cum_col="cumulative_logLR", max_time_step=15, xlim=None, ylim=None):
    _plot_hidden(trials, out_path, cum_col, "RdBu_r", "Cumulative evidence (log odds)", max_time_step, cum_col, xlim, ylim)


def plot_hidden_by_policy(
    trials,
    out_path,
    max_time_step=15,
    n_bins=60,
    min_bin_count=1,
    xlim=None,
    ylim=None,
    cell_alpha=1,
    show_trajectory=True,
    trajectory_trial_index=None,
    figsize=(7.2, 3.7),
):
    rows = []
    for trial in trials:
        steps = min(len(trial.hidden_path), len(trial.policy_path), max_time_step)
        rows.extend(
            np.c_[trial.hidden_path[:steps, :2], trial.policy_path[:steps, :3]]
        )
    values = np.asarray(rows)
    x, y = values[:, 0], values[:, 1]
    probabilities = values[:, 2:5]
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if xlim is None:
        padding = 0.03 * max(x.max() - x.min(), 1e-6)
        xlim = x.min() - padding, x.max() + padding
    if ylim is None:
        padding = 0.03 * max(y.max() - y.min(), 1e-6)
        ylim = y.min() - padding, y.max() + padding

    x_edges = np.linspace(*xlim, n_bins + 1)
    y_edges = np.linspace(*ylim, n_bins + 1)
    counts = np.histogram2d(x, y, bins=[x_edges, y_edges])[0]
    mean_policy = np.zeros((n_bins, n_bins, 3))
    for action in range(3):
        weighted = np.histogram2d(
            x, y, bins=[x_edges, y_edges], weights=probabilities[:, action]
        )[0]
        np.divide(weighted, counts, out=mean_policy[:, :, action], where=counts > 0)
    rgba = np.zeros((n_bins, n_bins, 4))
    rgba[:, :, 0] = mean_policy[:, :, 0]
    rgba[:, :, 1] = mean_policy[:, :, 2]
    rgba[:, :, 2] = mean_policy[:, :, 1]
    rgba[:, :, 3] = (counts >= min_bin_count) * cell_alpha

    set_style()
    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 0.78], wspace=0.18)
    ax = fig.add_subplot(grid[0])
    side = fig.add_subplot(grid[1])
    ax.set_box_aspect(1)
    side.axis("off")
    ax.imshow(
        rgba.transpose(1, 0, 2), extent=(*xlim, *ylim), origin="lower", aspect="auto",
        rasterized=True,
    )
    ax.set(xlim=xlim, ylim=ylim)
    ax.set_xlabel("Hidden unit 1 activity", fontsize=14)
    ax.set_ylabel("Hidden unit 2 activity", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    if show_trajectory:
        trial = trials[trajectory_trial_index] if trajectory_trial_index is not None else max(
            trials, key=lambda item: len(item.hidden_path)
        )
        points = trial.hidden_path[:max_time_step, :2]
        shapes = np.asarray(trial.shape_ids[:len(points)])
        shape_order = [4, 6, 8, 10, 9, 7, 5, 3]
        markers = dict(zip(shape_order, ["o", "s", "^", "v", "D", "P", "X", "*"]))
        labels = dict(zip(
            shape_order, ["-0.9", "-0.7", "-0.5", "-0.3", "+0.3", "+0.5", "+0.7", "+0.9"]
        ))
        ax.plot(points[:, 0], points[:, 1], color="0.4", lw=1.5, zorder=2, rasterized=True)
        for start, end in zip(points[:-1], points[1:]):
            ax.annotate(
                "", xy=start + 0.7 * (end - start),
                xytext=start + 0.35 * (end - start),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1,
                                mutation_scale=8, shrinkA=0, shrinkB=0),
                zorder=2,
            )
        for shape in shape_order:
            selected = shapes == shape
            if selected.any():
                ax.scatter(
                    points[selected, 0], points[selected, 1],
                    marker=markers[shape], facecolor="white", edgecolor="black",
                    linewidth=0.8, s=48, zorder=3, rasterized=True,
                )
        # Bundle heatmap + trajectory into one SVG raster layer.
        ax.set_rasterization_zorder(4)
        side.legend(
            handles=[
                Line2D(
                    [0], [0], marker=markers[shape], linestyle="none",
                    markerfacecolor="white", markeredgecolor="black",
                    markeredgewidth=0.8, markersize=7, label=labels[shape],
                )
                for shape in shape_order
            ],
            title="Stimulus (LLR)", ncol=2, frameon=True,
            loc="lower center", bbox_to_anchor=(0.5, 0.02),
            columnspacing=0.6, handlelength=1, handletextpad=0.3, borderpad=0.4,
            fontsize=9, title_fontsize=10,
        )

    # side.text(0.5, 0.98, "Action mixture", ha="center", va="top", fontsize=11)
    triangle_ax = side.inset_axes([0.05, 0.52, 0.9, 0.38])
    height = np.sqrt(3) / 2
    triangle_x, triangle_y = np.meshgrid(
        np.linspace(0, 1, 240), np.linspace(0, height, 210)
    )
    p_sample = triangle_y / height
    p_choose_b = triangle_x - p_sample / 2
    p_choose_a = 1 - p_sample - p_choose_b
    inside = (p_choose_a >= 0) & (p_choose_b >= 0) & (p_sample >= 0)
    triangle = np.dstack([
        np.clip(p_choose_a, 0, 1),
        np.clip(p_sample, 0, 1),
        np.clip(p_choose_b, 0, 1),
        inside,
    ])
    triangle_ax.imshow(
        triangle, extent=(0, 1, 0, height), origin="lower", aspect="equal"
    )
    triangle_ax.plot([0, 1, 0.5, 0], [0, 0, height, 0], color="0.25", lw=0.8)
    triangle_ax.text(0, -0.07, r"choose $H_1$", color="red", ha="center", va="top", fontsize=9)
    triangle_ax.text(1, -0.07, r"choose $H_0$", color="blue", ha="center", va="top", fontsize=9)
    triangle_ax.text(0.5, height + 0.06, "Sample", color="green", ha="center", va="bottom", fontsize=9)
    triangle_ax.set(xlim=(-0.18, 1.18), ylim=(-0.15, height + 0.15))
    triangle_ax.axis("off")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.97)
    _save(fig, out_path)
