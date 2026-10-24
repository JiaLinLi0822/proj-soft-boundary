"""RNN hidden-state and representation-noise plots."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
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
    trials, max_time_step=10, evidence_decimals=1, min_count=5,
    cum_col="cumulative_logLR",
):
    frame = trials_to_hidden_df(trials, cum_col)
    hidden_cols = [column for column in frame if column.startswith("h")]
    frame = frame[frame["time_step"] <= max_time_step].copy()
    frame["evidence_level"] = frame[cum_col].round(evidence_decimals)

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


def plot_representation_noise_over_time(summary, out_path):
    set_style()
    fig, ax = plt.subplots(figsize=(3.25, 2.65))
    x = summary["time_step"].to_numpy()
    y = summary["variability"].to_numpy()
    sem = summary["sem"].to_numpy()
    ax.plot(x, y, "-o", color="black", lw=1.6, ms=4)
    ax.fill_between(x, y - sem, y + sem, color="black", alpha=0.16)
    ax.set(
        xlabel="Timestep",
        ylabel="Hidden-state variability",
        xlim=(0, x.max()),
        xticks=np.arange(0, int(x.max()) + 1, 2),
    )
    _save(fig, out_path)


def plot_representation_noise_heatmap(noise, out_path):
    set_style()
    pivot = noise.pivot(
        index="evidence_level", columns="time_step", values="variability"
    ).sort_index()
    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    image = ax.imshow(
        pivot.to_numpy(), aspect="auto", origin="lower",
        interpolation="nearest", cmap="viridis",
    )
    ax.set_xticks(np.arange(len(pivot.columns)), pivot.columns.astype(int))
    ticks = np.unique(np.linspace(0, len(pivot.index) - 1, 9).round().astype(int))
    ax.set_yticks(ticks, [f"{pivot.index[i]:.1f}" for i in ticks])
    ax.set(xlabel="Timestep", ylabel="Cumulative evidence level")
    fig.colorbar(image, ax=ax, pad=0.02).set_label("Hidden-state variability")
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
    ax.set_xlabel("Hidden unit 1 activity", fontsize=16)
    ax.set_ylabel("Hidden unit 2 activity", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    # Lines + scatter become one embedded bitmap in SVG (axes/colorbar stay vector).
    ax.set_rasterization_zorder(2)
    
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(colorbar_label, fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    _save(fig, out_path)


def plot_hidden_by_timestep(trials, out_path, cum_col="cumulative_logLR", max_time_step=15, xlim=None, ylim=None):
    _plot_hidden(trials, out_path, "time_step", "plasma", "Time step", max_time_step, cum_col, xlim, ylim)


def plot_hidden_by_evidence(trials, out_path, cum_col="cumulative_logLR", max_time_step=15, xlim=None, ylim=None):
    _plot_hidden(trials, out_path, cum_col, "RdBu_r", "Cumulative evidence (LLR)", max_time_step, cum_col, xlim, ylim)


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
    ax.set_xlabel("Hidden unit 1", fontsize=16)
    ax.set_ylabel("Hidden unit 2", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

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
            title="Stimulus (LLR)", ncol=4, frameon=True,
            loc="lower center", bbox_to_anchor=(0.5, 0.02),
            columnspacing=1.1, handletextpad=0.45, borderpad=0.6,
        )

    side.text(0.5, 0.98, "Action mixture", ha="center", va="top", fontsize=11)
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


def _project_hidden(hidden_paths, projection):
    sequences, timesteps, dimensions = hidden_paths.shape
    if dimensions == 1:
        return np.stack(
            [
                np.broadcast_to(np.arange(1, timesteps + 1), (sequences, timesteps)),
                hidden_paths[:, :, 0],
            ],
            axis=-1,
        ), "Timestep", "Hidden unit 1"
    if projection in ("auto", "first2"):
        return hidden_paths[:, :, :2], "Hidden unit 1", "Hidden unit 2"
    flat = hidden_paths.reshape(-1, dimensions)
    centered = flat - flat.mean(axis=0)
    components = np.linalg.svd(centered, full_matrices=False)[2][:2]
    return (centered @ components.T).reshape(sequences, timesteps, 2), "PC1", "PC2"


def plot_permuted_evidence_trajectories(
    hidden_paths,
    hidden_df,
    out_path,
    match_timestep,
    projection="auto",
    line_alpha=0.42,
):
    paths, x_label, y_label = _project_hidden(np.asarray(hidden_paths), projection)
    n_sequences, n_time, _ = paths.shape
    norm = Normalize(1, n_time)
    colors = plt.cm.viridis(norm(np.arange(1, n_time + 1)))
    set_style()
    fig, (hidden_ax, evidence_ax) = plt.subplots(1, 2, figsize=(7, 3.15))

    for sequence_id, path in enumerate(paths):
        hidden_ax.add_collection(
            LineCollection(
                np.stack([path[:-1], path[1:]], axis=1),
                colors=colors[1:],
                alpha=line_alpha,
            )
        )
        hidden_ax.scatter(path[:, 0], path[:, 1], color=colors, s=16)
        sequence = hidden_df[hidden_df["sequence_id"] == sequence_id]
        evidence_ax.plot(
            sequence["time_step"], sequence["cumulative_logLR"], alpha=line_alpha
        )
    matched = hidden_df[hidden_df["time_step"] == match_timestep]
    evidence_ax.scatter(
        matched["time_step"], matched["cumulative_logLR"], marker="s", color="black"
    )
    hidden_ax.set(xlabel=x_label, ylabel=y_label, title="Hidden trajectories")
    hidden_ax.set_box_aspect(1)
    hidden_ax.autoscale()
    evidence_ax.axvline(match_timestep, color="black", alpha=0.35)
    evidence_ax.set(xlabel="Time steps", ylabel="Cumulative evidence", title="Matched evidence")
    colorbar = ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    fig.colorbar(colorbar, ax=[hidden_ax, evidence_ax], pad=0.02).set_label("Time steps")
    fig.tight_layout()
    _save(fig, out_path)
