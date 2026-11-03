"""Append-only training logs (CSV) with truncate-on-resume, and a mean-alpha plot helper."""
from __future__ import annotations

from pathlib import Path

from fedrl.utils.logging_utils import CsvLogger

TERMS = ("r_global", "r_pers", "r_fair", "comm_norm", "comp_norm", "final_bonus")
PPO_KEYS = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var")
VAL_FIELDS = ["episode", "val_return", "is_best"]


def train_fields(agent_names: list[str]) -> list[str]:
    base = ["episode", "scenario_a", "drift", "scenario_seed", "return", *TERMS, "mean_alpha",
            "policy_entropy", "sel_entropy", "env_steps", "wall_time"]
    return base + [f"{a}_{k}" for a in agent_names for k in PPO_KEYS]


class TrainLog:
    """train_log.csv, val_log.csv and selection_counts.csv inside a run folder."""

    def __init__(self, run_dir: Path | str, agent_names: list[str], n_clients: int):
        rd = Path(run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        self.fields = train_fields(agent_names)
        self.agent_names = list(agent_names)
        self.train = CsvLogger(rd / "train_log.csv", self.fields)
        self.val = CsvLogger(rd / "val_log.csv", VAL_FIELDS)
        self.sel_fields = ["episode"] + [f"c{i}" for i in range(n_clients)]
        self.sel = CsvLogger(rd / "selection_counts.csv", self.sel_fields)

    def truncate_after(self, episode: int) -> None:
        """Drop rows of episodes > episode from all logs (called on resume)."""
        for lg in (self.train, self.val, self.sel):
            lg.truncate_after(episode)

    def log_episode(self, row: dict, ppo_stats: dict | None = None) -> None:
        full = {k: "" for k in self.fields}
        full.update(row)
        if ppo_stats:
            for a, st in ppo_stats.items():
                for k in PPO_KEYS:
                    full[f"{a}_{k}"] = st.get(k, "")
        self.train.append(full)

    def log_validation(self, episode: int, val_return: float, is_best: bool) -> None:
        self.val.append({"episode": episode, "val_return": val_return, "is_best": int(is_best)})

    def log_selection(self, episode: int, counts) -> None:
        row = {"episode": episode}
        row.update({f"c{i}": int(c) for i, c in enumerate(counts)})
        self.sel.append(row)


def plot_mean_alpha(run_dir: Path | str, out_path: Path | str | None = None, window: int = 10) -> Path:
    """Plot mean alpha of chosen clients over training episodes (with a moving average)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    rd = Path(run_dir)
    df = pd.read_csv(rd / "train_log.csv")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df["episode"], df["mean_alpha"], alpha=0.3, label="per episode")
    ax.plot(df["episode"], df["mean_alpha"].rolling(window, min_periods=1).mean(), label=f"MA({window})")
    ax.set_xlabel("episode")
    ax.set_ylabel("mean alpha (chosen clients)")
    ax.legend()
    fig.tight_layout()
    out = Path(out_path) if out_path else rd / "mean_alpha.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out