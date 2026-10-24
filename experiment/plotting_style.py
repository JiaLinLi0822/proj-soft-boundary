import matplotlib.pyplot as plt
import os

def set_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

def _save(fig, path, dpi=500):
    stem, _ = os.path.splitext(path)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    fig.savefig(stem + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(stem + ".svg", dpi=dpi, bbox_inches="tight")
    plt.close(fig)