"""Build results tables, plots and statements from result JSONs (MASTER 2.13, 2.14)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fedrl.eval.metrics import to_jsonable
from fedrl.eval.pilot import moving_average
from fedrl.eval.stats import (DEFAULT_METRICS, compare_methods, non_rl_methods, select_best_baseline,
                              summarize)
from fedrl.methods import is_rl
from fedrl.utils.logging_utils import get_logger
from fedrl.eval.logs import read_alpha_curve, read_val_curve
from fedrl.runs.run_manager import resolve_runs_dir  # Part 6 precedence (re-exported for scripts/analyze.py)

logger = get_logger(__name__)

LOCAL_ONLY_SENTENCE = ("L_i is the global model fine-tuned by the client for ONE round (it is overwritten at every "
                       "participation), not a persistent personal model.")
RL_MAIN = ["joint", "stacked_shared", "stacked_specialized"]




def load_result_records(root: str | Path) -> list[dict]:
    """All result JSONs under root (any depth) as flat records."""
    recs = []
    for p in sorted(Path(root).glob("**/results/*.json")):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            logger.warning("skipping corrupt result file %s", p)
            continue
        m, sc = r.get("meta", {}), r.get("meta", {}).get("scenario", {})
        d = r.get("diagnostics", {})
        rec = {"dataset": m.get("dataset"), "method": m.get("method"), "variant": m.get("variant", "default"),
               "train_seed": m.get("train_seed", 0), "a": sc.get("a"), "drift": sc.get("drift"),
               "scenario_seed": sc.get("seed"), "eval_seed": m.get("eval_seed"), "tag": m.get("tag", ""),
               "N": m.get("N"), "k": m.get("k"), "experiment": m.get("experiment"), "path": str(p)}
        rec.update(r.get("metrics", {}))
        for key, col in (("pers_gain_chosen_per_round", "pers_gain_chosen_mean"),
                         ("pers_gain_all_per_round", "pers_gain_all_mean"),
                         ("mean_alpha_per_round", "mean_alpha_mean")):
            v = d.get(key)
            rec[col] = float(np.nanmean([np.nan if x is None else x for x in v])) if v else float("nan")
        recs.append(rec)
    return recs


def results_dataframe(root: str | Path) -> pd.DataFrame:
    """DataFrame with one row per result JSON."""
    return pd.DataFrame(load_result_records(root))


def baseline_validation_scores(root: str | Path, dataset: str) -> dict[str, float]:
    """Mean validation return per non-RL method from validation.json files (variant 'default')."""
    out: dict[str, float] = {}
    for p in Path(root).glob("**/validation.json"):
        try:
            v = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        parts = p.parts
        method = v.get("method", parts[-5] if len(parts) >= 5 else None)
        ds = v.get("dataset", parts[-6] if len(parts) >= 6 else None)
        variant = v.get("variant", parts[-4] if len(parts) >= 4 else "default")
        if method is None or ds != dataset or variant != "default" or is_rl(method):
            continue
        out[method] = float(v["mean_return"])
    return out


def _load_curve(path: str, key: str = "global_test_acc") -> list[float] | None:
    r = json.loads(Path(path).read_text())
    return r.get("curves", {}).get(key)


# ----------------------------------------------------------------------------- tables
def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (float, np.floating)):
        return f"{v:.4f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table writer."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def with_pooled(df: pd.DataFrame) -> pd.DataFrame:
    """Append a copy of df with a='ALL', drift='ALL' (pooled over scenarios)."""
    pooled = df.copy()
    pooled["a"], pooled["drift"] = "ALL", "ALL"
    out = df.copy()
    out["a"] = out["a"].astype(object)
    return pd.concat([out, pooled], ignore_index=True)


def format_summary(summ: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """'mean ± std' strings across training seeds."""
    rows = []
    for _, r in summ.iterrows():
        row = {k: r[k] for k in ("dataset", "method", "variant", "a", "drift", "n_seeds") if k in r.index}
        for m in metrics:
            if f"{m}_mean" in r:
                row[m] = "–" if not np.isfinite(r[f"{m}_mean"]) else f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def per_seed_long(summ: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Long table of per-training-seed values (each averaged over evaluation seeds)."""
    rows = []
    for _, r in summ.iterrows():
        for m in metrics:
            for seed, val in r.get(f"{m}_per_seed", {}).items():
                rows.append({"dataset": r["dataset"], "method": r["method"], "variant": r["variant"],
                             "a": r["a"], "drift": r["drift"], "metric": m, "train_seed": seed, "value": val})
    return pd.DataFrame(rows)


def e5_table(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """E5 (free): accuracy per communication / computation unit and drift recovery, from logs only."""
    d = df[(df["dataset"] == dataset) & (df["variant"] == "default") & (df["tag"] == "")].copy()
    if d.empty:
        return pd.DataFrame()
    for acc in ("global_test_acc", "pers_test_acc"):
        d[f"{acc}_per_comm"] = d[acc] / d["comm"].replace(0, np.nan)
        d[f"{acc}_per_comp"] = d[acc] / d["comp"].replace(0, np.nan)
    mets = ["global_test_acc_per_comm", "global_test_acc_per_comp", "pers_test_acc_per_comm",
            "pers_test_acc_per_comp", "drift_recovery"]
    return format_summary(summarize(d, mets, by=("dataset", "method", "variant", "drift")), mets)


def scenario_statements(cmp_df: pd.DataFrame, ref: str = "joint") -> list[str]:
    """Plain per-scenario statement of where ref wins, ties or loses (ties/losses are never hidden)."""
    lines = []
    for (a, dr), g in cmp_df.groupby(["a", "drift"], sort=True):
        def names(v):
            return ", ".join(sorted(g[g["verdict"] == v]["other"].unique())) or "none"
        lines.append(f"[{cmp_df['metric'].iloc[0]}] a={a}, drift={dr}: {ref} is BETTER than: {names('better')}; "
                     f"TIES with: {names('tie')}; WORSE than: {names('worse')}.")
    return lines


def methods_paragraph(n_comparisons: int, df: pd.DataFrame, n_resamples: int, ci: float) -> str:
    """Methods text with the local-only definition and the number of comparisons."""
    rl = df[df["method"].map(is_rl)] if len(df) else df
    n_train = int(rl.groupby("method")["train_seed"].nunique().max()) if len(rl) else 0
    n_eval = int(df["eval_seed"].nunique()) if len(df) else 0
    return (
        f"All methods were evaluated on held-out test scenarios with {n_eval} evaluation seeds; RL methods were "
        f"trained with up to {n_train} independent training seeds. Differences between methods are paired "
        f"bootstrap estimates ({n_resamples} resamples, {int(ci * 100)}% confidence interval) over scenarios "
        f"matched on scenario seeds; training seeds are averaged first and rounds within a run are never treated "
        f"as independent samples. A method is called better only if the interval excludes zero; otherwise the "
        f"result is reported as a tie. In total {n_comparisons} comparisons were made. The best non-RL baseline "
        f"was selected once on validation scenarios, never on test results. Local-only caveat: {LOCAL_ONLY_SENTENCE} "
        f"The personalized model of a client is B_i = alpha_i * G + (1 - alpha_i) * L_i, and clients that never "
        f"trained use the global model G.\n")


def param_count_table(cfg: Any, dim: int | None = None) -> pd.DataFrame:
    """Parameter counts: joint policy vs the three stacked agents (per module and total)."""
    import torch.nn as nn
    from fedrl.env.state import state_dim
    from fedrl.policy.joint_policy import JointPolicy
    from fedrl.policy.stacked import StackedPolicy

    d = dim if dim is not None else state_dim(cfg)

    def count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    def rows_for(name: str, pol: nn.Module) -> list[dict]:
        rows = [{"policy": name, "module": "TOTAL", "params": count(pol)}]
        for cname, child in pol.named_children():
            if isinstance(child, (nn.ModuleDict, nn.ModuleList)):
                items = child.items() if isinstance(child, nn.ModuleDict) else enumerate(child)
                rows += [{"policy": name, "module": f"{cname}.{k}", "params": count(v)} for k, v in items]
            else:
                rows.append({"policy": name, "module": cname, "params": count(child)})
        return rows

    return pd.DataFrame(rows_for("joint", JointPolicy(cfg, d)) + rows_for("stacked", StackedPolicy(cfg, d)))


# ----------------------------------------------------------------------------- plots
def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _log_key(p: Path) -> tuple[str, str, str, str]:
    parts = p.parts
    return parts[-6], parts[-5], parts[-4], parts[-3]  # dataset, method, variant, seedS

def plot_learning_curves(root: str | Path, out_dir: Path) -> list[Path]:
    """Validation-return learning curves (val_log.csv), one line per training seed per (dataset, method, variant)."""
    groups: dict[tuple, list] = {}
    for p in Path(root).glob("**/val_log.csv"):
        if len(p.parts) < 6:
            continue
        v = read_val_curve(p.parent)
        if v is None:
            continue
        ds, method, variant, seed = _log_key(p)
        groups.setdefault((ds, method, variant), []).append((seed, v))
    paths = []
    for (ds, method, variant), items in groups.items():
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for seed, (eps, vals) in sorted(items, key=lambda t: t[0]):
            ax.plot(eps, vals, marker="o", ms=3, label=seed)
        ax.set(xlabel="episode", ylabel="validation return", title=f"{ds}/{method}/{variant}")
        ax.legend(fontsize=7)
        paths.append(_save(fig, out_dir / f"learning_curves_{ds}_{method}_{variant}.png"))
    return paths


def plot_mean_alpha(root: str | Path, out_dir: Path) -> list[Path]:
    """Mean alpha of chosen clients over training (5-point moving average, one line per run)."""
    by_ds: dict[str, list] = {}
    for p in Path(root).glob("**/train_log.csv"):
        if len(p.parts) < 6:
            continue
        a = read_alpha_curve(p.parent)
        if a is None or a.size == 0:
            continue
        ds, method, variant, seed = _log_key(p)
        by_ds.setdefault(ds, []).append((f"{method}/{variant}/{seed}", a))
    paths = []
    for ds, items in by_ds.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, a in items:
            ax.plot(moving_average(a, 5), lw=1, label=label)
        ax.axhspan(0.2, 0.8, color="grey", alpha=0.1)
        ax.set(xlabel="logged training row", ylabel="mean alpha (chosen clients)", title=f"Mean alpha over training ({ds})")
        ax.legend(fontsize=5, ncol=2)
        paths.append(_save(fig, out_dir / f"mean_alpha_{ds}.png"))
    return paths


def plot_acc_vs_comm(df: pd.DataFrame, out_dir: Path, dataset: str, focus_a: float = 0.2,
                     metric: str = "pers_test_acc") -> Path | None:
    """E3: accuracy vs communication cost; joint reward variants as circles, baselines marked with crosses."""
    d = df[(df["dataset"] == dataset) & (df["tag"] == "") & np.isclose(pd.to_numeric(df["a"], errors="coerce"), focus_a)]
    if d.empty:
        return None
    seed_level = d.groupby(["method", "variant", "train_seed"])[[metric, "comm"]].mean().reset_index()
    agg = seed_level.groupby(["method", "variant"])[[metric, "comm"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in agg.iterrows():
        base = not is_rl(r["method"])
        if base:
            ax.scatter(r["comm"], r[metric], marker="x", c="k")
            ax.annotate(r["method"], (r["comm"], r[metric]), fontsize=6)
        elif r["method"] == "joint":
            ax.scatter(r["comm"], r[metric], marker="o", c="tab:blue")
            ax.annotate(r["variant"], (r["comm"], r[metric]), fontsize=6)
        else:
            ax.scatter(r["comm"], r[metric], marker="^", c="tab:orange")
            ax.annotate(r["method"], (r["comm"], r[metric]), fontsize=6)
    ax.set(xlabel="communication cost (sum over rounds)", ylabel=metric,
           title=f"Accuracy vs communication, a={focus_a}, {dataset} (x = baselines)")
    return _save(fig, out_dir / f"acc_vs_comm_{dataset}.png")


def plot_drift_recovery(df: pd.DataFrame, out_dir: Path, dataset: str, focus_a: float = 0.2,
                        methods: Sequence[str] = tuple(RL_MAIN) + ("fedavg", "fixed_alpha_0.5", "loss_heuristic")) -> list[Path]:
    """Mean global-test-accuracy curve per method for each drifting scenario, drift start marked."""
    paths = []
    for drift in ("abrupt", "gradual"):
        d = df[(df["dataset"] == dataset) & (df["drift"] == drift) & (df["tag"] == "") & (df["variant"] == "default")
               & np.isclose(pd.to_numeric(df["a"], errors="coerce"), focus_a)]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        start = None
        for m in methods:
            cs = [c for c in (_load_curve(p) for p in d[d["method"] == m]["path"]) if c]
            if not cs:
                continue
            n = min(len(c) for c in cs)
            ax.plot(range(1, n + 1), np.mean([c[:n] for c in cs], axis=0), label=m)
        s = d["drift_start"].dropna()
        if len(s):
            start = int(s.iloc[0])
            ax.axvline(start, color="red", ls=":", label="drift start")
        ax.set(xlabel="round", ylabel="global test accuracy", title=f"Drift recovery ({drift}, a={focus_a}, {dataset})")
        ax.legend(fontsize=6)
        paths.append(_save(fig, out_dir / f"drift_recovery_{dataset}_{drift}.png"))
    return paths


def plot_selection_heatmaps(df: pd.DataFrame, out_dir: Path, dataset: str,
                            methods: Sequence[str] = tuple(RL_MAIN) + ("fedavg", "loss_heuristic")) -> list[Path]:
    """Selection counts: rows = (a, drift, eval seed) runs of the first training seed, columns = client id."""
    paths = []
    for m in methods:
        d = df[(df["dataset"] == dataset) & (df["method"] == m) & (df["variant"] == "default") & (df["tag"] == "")]
        if d.empty:
            continue
        d = d[d["train_seed"] == d["train_seed"].min()].sort_values(["a", "drift", "eval_seed"])
        mats, labels = [], []
        for _, r in d.iterrows():
            c = json.loads(Path(r["path"]).read_text()).get("diagnostics", {}).get("selection_counts")
            if c:
                mats.append(c)
                labels.append(f"a={r['a']:g},{r['drift']},{r['eval_seed']}")
        if not mats:
            continue
        fig, ax = plt.subplots(figsize=(7, max(3, 0.18 * len(mats))))
        im = ax.imshow(np.array(mats), aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=5)
        ax.set(xlabel="client", title=f"Selection counts: {m} ({dataset})")
        fig.colorbar(im, ax=ax)
        paths.append(_save(fig, out_dir / f"selection_heatmap_{dataset}_{m}.png"))
    return paths


def plot_pers_gain(df: pd.DataFrame, out_dir: Path, dataset: str) -> Path | None:
    """Chosen-client vs all-client personalization gain (mean over rounds) per method."""
    d = df[(df["dataset"] == dataset) & (df["variant"] == "default") & (df["tag"] == "")]
    if d.empty:
        return None
    g = d.groupby("method")[["pers_gain_chosen_mean", "pers_gain_all_mean"]].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(g))
    ax.bar(x - 0.2, g["pers_gain_chosen_mean"], 0.4, label="chosen clients")
    ax.bar(x + 0.2, g["pers_gain_all_mean"], 0.4, label="all clients")
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=7)
    ax.set(ylabel="personalization gain over local-only (mean over rounds)", title=f"Personalization gain ({dataset})")
    ax.legend()
    return _save(fig, out_dir / f"pers_gain_{dataset}.png")


def plot_e5(e5: pd.DataFrame, out_dir: Path, dataset: str) -> Path | None:
    """E5 bars: global accuracy per communication unit and per computation unit (scenario-pooled by method)."""
    if e5.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ("global_test_acc_per_comm", "global_test_acc_per_comp")):
        vals = e5.assign(v=pd.to_numeric(e5[col].str.split(" ").str[0], errors="coerce")).copy()
        g = vals.groupby("method")["v"].mean()
        ax.bar(range(len(g)), g.values)
        ax.set_xticks(range(len(g)))
        ax.set_xticklabels(g.index, rotation=60, ha="right", fontsize=7)
        ax.set_title(col)
    return _save(fig, out_dir / f"e5_efficiency_{dataset}.png")


# ----------------------------------------------------------------------------- driver
def run_analysis(root: str | Path, out_dir: str | Path, cfg: Any = None, ref: str = "joint",
                 primary_metric: str = "pers_test_acc",
                 compare_metric_list: Sequence[str] = ("pers_test_acc", "global_test_acc", "worst10_acc"),
                 n_resamples: int = 10000, ci: float = 0.95, include_param_table: bool = True,
                 focus_a: float = 0.2, seed: int = 0) -> dict:
    """Write all tables, plots and text of the analysis; returns paths and the number of comparisons."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = results_dataframe(root)
    if df.empty:
        raise ValueError(f"no result JSONs found under {root}")
    main_df = df[df["tag"] == ""]
    written: list[str] = []
    total_cmp, statements = 0, []
    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    for ds in sorted(main_df["dataset"].dropna().unique()):
        sub = main_df[main_df["dataset"] == ds]
        summ = summarize(with_pooled(sub), metrics)
        fmt = format_summary(summ, metrics)
        fmt.to_csv(out / f"summary_{ds}.csv", index=False)
        (out / f"summary_{ds}.md").write_text(f"## Summary {ds} (mean ± std across training seeds)\n\n" + df_to_markdown(fmt))
        per_seed_long(summ, metrics).to_csv(out / f"per_seed_{ds}.csv", index=False)
        written += [f"summary_{ds}.md", f"summary_{ds}.csv", f"per_seed_{ds}.csv"]
        scores = baseline_validation_scores(root, ds)
        best = None
        if scores:
            best = select_best_baseline(scores)
            (out / f"best_baseline_{ds}.json").write_text(json.dumps({"best": best, "validation_scores": scores}, indent=2))
        else:
            logger.warning("no validation.json for baselines of %s: best-baseline comparison skipped", ds)
        for m in compare_metric_list:
            if m not in sub.columns:
                continue
            try:
                cmp = compare_methods(sub, metric=m, ref=ref, dataset=ds, best_baseline=best,
                                      n_resamples=n_resamples, ci=ci, seed=seed)
            except ValueError as exc:
                logger.warning("comparison skipped (%s, %s): %s", ds, m, exc)
                continue
            total_cmp += len(cmp)
            cmp.to_csv(out / f"compare_{ds}_{m}.csv", index=False)
            (out / f"compare_{ds}_{m}.md").write_text(df_to_markdown(cmp.drop(columns=["metric"])))
            written += [f"compare_{ds}_{m}.csv", f"compare_{ds}_{m}.md"]
            if m == primary_metric or m == "global_test_acc":
                statements += [f"{ds}: {s}" for s in scenario_statements(cmp, ref)]
        e5 = e5_table(df, ds)
        if not e5.empty:
            e5.to_csv(out / f"e5_{ds}.csv", index=False)
            (out / f"e5_{ds}.md").write_text(df_to_markdown(e5))
        plots = [lambda: plot_acc_vs_comm(sub, out, ds, focus_a), lambda: plot_drift_recovery(sub, out, ds, focus_a),
                 lambda: plot_selection_heatmaps(sub, out, ds), lambda: plot_pers_gain(sub, out, ds),
                 lambda: plot_e5(e5, out, ds)]
        for fn in plots:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - a failed plot must not stop the analysis
                logger.warning("plot failed: %r", exc)
    for fn in (plot_learning_curves, plot_mean_alpha):
        try:
            fn(root, out)
        except Exception as exc:  # noqa: BLE001
            logger.warning("plot failed: %r", exc)
    statements.append(f"Total comparisons made: {total_cmp}")
    (out / "statements.txt").write_text("\n".join(statements) + "\n")
    (out / "methods.md").write_text(methods_paragraph(total_cmp, main_df, n_resamples, ci))
    if include_param_table and cfg is not None:
        pc = param_count_table(cfg)
        pc.to_csv(out / "param_counts.csv", index=False)
        (out / "param_counts.md").write_text(df_to_markdown(pc))
    logger.info("\n".join(statements))
    logger.info("total comparisons across all metrics/datasets: %d", total_cmp)
    return {"n_comparisons": total_cmp, "files": written, "out_dir": str(out)}