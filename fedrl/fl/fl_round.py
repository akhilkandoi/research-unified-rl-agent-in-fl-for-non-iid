"""FLWorld: one federated run (MASTER 2.5), measurements per MASTER 2.9."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from fedrl.fl.client import local_train
from fedrl.fl.costs import cost_reference, round_costs
from fedrl.fl.evaluate import evaluate_vec
from fedrl.fl.resources import Battery, eligible_mask, epochs_mask
from fedrl.fl.server import aggregate
from fedrl.models.cnn import build_model
from fedrl.utils.flat import blend, get_flat, num_params, rel_div
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_torch_gen

log = get_logger(__name__)


class IneligibleClientError(ValueError):
    """A chosen client is ineligible or a masked epochs option was chosen."""


def init_model_vec(dataset: str, init_seed: int) -> torch.Tensor:
    """Flat initial weights for `dataset`, deterministic in `init_seed` (CPU tensor)."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(init_seed))
        model = build_model(dataset)
        return get_flat(model).detach().clone().cpu()


_MODELS: Dict[Tuple[str, str], torch.nn.Module] = {}


def get_shared_model(dataset: str, device: torch.device) -> torch.nn.Module:
    """One model object per (dataset, device) per process, reused by every FLWorld."""
    key = (dataset, str(device))
    if key not in _MODELS:
        _MODELS[key] = build_model(dataset).to(device)
    return _MODELS[key]


@dataclass
class RoundResult:
    selected: List[int]
    epochs_vals: List[int]
    global_val_acc_before: float
    global_val_acc_after: float
    fair_p10_before: float
    fair_p10_after: float
    pers_gain_chosen: float
    pers_gain_all: float
    comm: float
    comp: float
    latency: float
    divergence: torch.Tensor
    probe_loss: torch.Tensor          # probe of G_new (next round's probe)
    probe_acc: torch.Tensor
    battery: torch.Tensor             # after drain and recharge
    rounds_since_local: torch.Tensor
    times_picked: torch.Tensor
    idle_rounds: torch.Tensor
    alpha_vals: torch.Tensor          # alpha used for ALL clients
    # ---- additive extras ----
    lr_vals: List[float] = field(default_factory=list)
    batch_vals: List[int] = field(default_factory=list)
    update_cos: Optional[torch.Tensor] = None
    prev_probe_loss: Optional[torch.Tensor] = None
    prev_probe_acc: Optional[torch.Tensor] = None
    n_probes: int = 0
    timings: Dict[str, float] = field(default_factory=dict)


@dataclass
class _Action:
    selected: List[int]
    lr_vals: List[float]
    epochs_vals: List[int]
    batch_vals: List[int]
    alpha_all: List[float]
    prox_mu: float


class _Sections:
    """Optional wall-clock profiling of round sections (syncs CUDA when enabled)."""

    def __init__(self, device: torch.device, enabled: bool) -> None:
        self.device, self.enabled = device, enabled
        self.acc: Dict[str, float] = {}
        self.t0 = 0.0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        if self.enabled:
            self._sync()
            self.t0 = time.perf_counter()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.acc[name] = self.acc.get(name, 0.0) + time.perf_counter() - t

    def finish(self) -> Dict[str, float]:
        if not self.enabled:
            return {}
        self._sync()
        total = time.perf_counter() - self.t0
        out = {k: self.acc.get(k, 0.0) for k in ("probe", "train", "aggregate", "pers_eval", "global_eval")}
        out["other"] = max(total - sum(out.values()), 0.0)
        out["total"] = total
        return out


def _per_client(value, N: int, name: str) -> list:
    t = torch.as_tensor(value)
    if t.ndim == 0:
        return [t.item()] * N
    t = t.reshape(-1)
    if t.numel() != N:
        raise ValueError(f"action['{name}'] has {t.numel()} entries, expected {N}")
    return t.tolist()


def make_action(cfg, N: int, selected: Sequence[int], lr_idx=None, epochs_idx=None,
                batch_idx=None, alpha_idx=None, **extra) -> dict:
    """Build a full action dict. Each *_idx may be an int or a length-N sequence (default: menu defaults)."""
    def full(v, default):
        t = torch.as_tensor(default if v is None else v, dtype=torch.long)
        return t.expand(N).clone() if t.ndim == 0 else t.clone()

    m = cfg.menus
    act = {
        "selected": torch.as_tensor(list(selected), dtype=torch.long),
        "lr_idx": full(lr_idx, m.default_lr_idx),
        "epochs_idx": full(epochs_idx, m.default_epochs_idx),
        "batch_idx": full(batch_idx, m.default_batch_idx),
        "alpha_idx": full(alpha_idx, m.default_alpha_idx),
    }
    act.update(extra)
    return act


class FLWorld:
    """Federated world for one scenario: global model G, local models L_i, batteries, probes."""

    def __init__(self, scenario, bundle, pools, cfg, device) -> None:
        self.scenario, self.bundle, self.pools, self.cfg = scenario, bundle, pools, cfg
        self.device = torch.device(device)
        self.N = int(scenario.N)
        self.k = int(cfg.fl.k)
        self.model = get_shared_model(bundle.name, self.device)
        self.P = num_params(self.model)
        self.model_bytes = float(self.P * 4)
        self.X, self.y = bundle.train_x, bundle.train_y
        self.global_val_idx = pools.global_val_idx.to(self.device)
        self.profile = False
        self._G: torch.Tensor = torch.zeros(self.P, device=self.device)
        self._gval: Optional[float] = None
        self._refresh_client_tensors()
        self.reset()

    # ------------------------------------------------------------------ state
    @property
    def G(self) -> torch.Tensor:
        return self._G

    @G.setter
    def G(self, vec: torch.Tensor) -> None:
        self._G = vec
        self._gval = None

    @property
    def battery(self) -> torch.Tensor:
        return self.bat.level

    def _refresh_client_tensors(self) -> None:
        cl = self.scenario.clients
        f32 = dict(dtype=torch.float32, device=self.device)
        self.n = torch.tensor([float(c.train_idx.numel()) for c in cl], **f32)
        self.speed = torch.tensor([float(c.compute_speed) for c in cl], **f32)
        self.bw = torch.tensor([float(c.bandwidth) for c in cl], **f32)
        if hasattr(self, "bat"):
            self.bat.n = self.n

    def reset(self) -> None:
        """Fresh G (from scenario.init_seed), L_i=None, batteries from scenario, zero counters, initial probe."""
        N, dev = self.N, self.device
        self.G = init_model_vec(self.bundle.name, self.scenario.init_seed).to(dev)
        self.L: List[Optional[torch.Tensor]] = [None] * N
        b0 = torch.tensor([float(c.battery0) for c in self.scenario.clients], dtype=torch.float32, device=dev)
        self.bat = Battery(b0, self.n, self.cfg)
        self.mom: Optional[torch.Tensor] = None
        z = lambda: torch.zeros(N, dtype=torch.float32, device=dev)
        self.divergence, self.update_cos = z(), z()
        self.times_picked, self.idle_rounds, self.rounds_since_local = z(), z(), z()
        self.refs = cost_reference(self.n, self.speed, self.bw, self.model_bytes, self.cfg)
        self.n_probes = 0
        self.global_val_acc()  # prime the cache
        loss, acc = self.probe_all()
        self.probe_loss, self.probe_acc = loss, acc
        self.prev_probe_loss, self.prev_probe_acc = loss.clone(), acc.clone()

    # ---------------------------------------------------------------- measure
    def probe_all(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate G on every client's client-validation set (no grad, batched)."""
        losses, accs = [], []
        for c in self.scenario.clients:
            l, a = evaluate_vec(self.model, self.G, self.X, self.y, c.val_idx.to(self.device))
            losses.append(l)
            accs.append(a)
        self.n_probes += 1
        f32 = dict(dtype=torch.float32, device=self.device)
        return torch.tensor(losses, **f32), torch.tensor(accs, **f32)

    def _store_probe(self, loss: torch.Tensor, acc: torch.Tensor) -> None:
        self.prev_probe_loss, self.prev_probe_acc = self.probe_loss, self.probe_acc
        self.probe_loss, self.probe_acc = loss, acc

    def global_val_acc(self) -> float:
        """Accuracy of G on the balanced global-validation set (cached until G changes)."""
        if self._gval is None:
            self._gval = float(evaluate_vec(self.model, self.G, self.X, self.y, self.global_val_idx)[1])
        return self._gval

    def _p10(self, acc: torch.Tensor) -> float:
        return float(np.percentile(acc.detach().cpu().numpy(), self.cfg.reward.fair_percentile))

    def _pers_acc(self, i: int, vec: torch.Tensor) -> float:
        idx = self.scenario.clients[i].personal_val_idx.to(self.device)
        return float(evaluate_vec(self.model, vec, self.X, self.y, idx)[1])

    def personalized_vec(self, i: int, alpha: float) -> torch.Tensor:
        """alpha*G + (1-alpha)*L_i; G itself for never-trained clients (D5)."""
        return self.G.clone() if self.L[i] is None else blend(alpha, self.G, self.L[i])

    # ----------------------------------------------------------------- drift
    def apply_drift_if_due(self, round_t: int) -> bool:
        """Apply scenario drift; if data changed, refresh sizes and do the free re-probe."""
        changed = bool(self.scenario.apply_drift(round_t))
        if changed:
            self._refresh_client_tensors()
            self._store_probe(*self.probe_all())
        return changed

    # ---------------------------------------------------------------- action
    def _parse_action(self, action: dict, eligible: torch.Tensor, emask: torch.Tensor) -> _Action:
        cfg, N = self.cfg, self.N
        if "selected" not in action:
            raise KeyError("action is missing 'selected'")
        sel = [int(v) for v in torch.as_tensor(action["selected"]).reshape(-1).tolist()]
        if len(set(sel)) != len(sel):
            raise ValueError(f"Duplicate clients in selection {sel}")
        if len(sel) > self.k:
            raise ValueError(f"Selected {len(sel)} clients but k={self.k}")
        bad = [i for i in sel if not 0 <= i < N]
        if bad:
            raise ValueError(f"Client ids out of range [0,{N}): {bad}")
        elig = eligible.cpu().tolist()
        for i in sel:
            if not elig[i]:
                raise IneligibleClientError(
                    f"Client {i} is not eligible (battery={float(self.bat.level[i]):.4f}, "
                    f"min_battery={cfg.clients.min_battery})")

        def resolve(idx_key, val_key, menu, clients):
            if action.get(val_key) is not None:
                vals = _per_client(action[val_key], N, val_key)
                return [float(vals[i]) for i in clients]
            if idx_key not in action:
                raise KeyError(f"action needs '{idx_key}' (or '{val_key}')")
            idxs = _per_client(action[idx_key], N, idx_key)
            out = []
            for i in clients:
                j = int(idxs[i])
                if not 0 <= j < len(menu):
                    raise ValueError(f"{idx_key}[{i}]={j} outside menu of size {len(menu)}")
                out.append(menu[j])
            return out

        lr_vals = resolve("lr_idx", "lr_value", cfg.menus.lr, sel)
        batch_vals = self._menu_only(action, "batch_idx", cfg.menus.batch, sel)
        epochs_vals = []
        e_idx = _per_client(action["epochs_idx"], N, "epochs_idx") if "epochs_idx" in action else None
        if e_idx is None and sel:
            raise KeyError("action is missing 'epochs_idx'")
        em = emask.cpu().tolist()
        for i in sel:
            j = int(e_idx[i])
            if not 0 <= j < len(cfg.menus.epochs):
                raise ValueError(f"epochs_idx[{i}]={j} outside menu")
            if not em[i][j]:
                raise IneligibleClientError(
                    f"Epochs option {cfg.menus.epochs[j]} is masked for client {i} "
                    f"(battery={float(self.bat.level[i]):.4f})")
            epochs_vals.append(int(cfg.menus.epochs[j]))
        alpha_all = [float(a) for a in resolve("alpha_idx", "alpha_value", cfg.menus.alpha, range(N))]
        if any(not 0.0 <= a <= 1.0 for a in alpha_all):
            raise ValueError("alpha values must lie in [0, 1]")
        mu = float(action.get("prox_mu", cfg.fl.prox_mu) or 0.0)
        return _Action(sel, [float(v) for v in lr_vals], epochs_vals, batch_vals, alpha_all, mu)

    def _menu_only(self, action: dict, key: str, menu: list, clients: Sequence[int]) -> List[int]:
        if key not in action:
            raise KeyError(f"action is missing '{key}'")
        idxs = _per_client(action[key], self.N, key)
        out = []
        for i in clients:
            j = int(idxs[i])
            if not 0 <= j < len(menu):
                raise ValueError(f"{key}[{i}]={j} outside menu of size {len(menu)}")
            out.append(int(menu[j]))
        return out

    # ------------------------------------------------------------------ round
    def run_round(self, round_t: int, action: dict) -> RoundResult:
        """One FL round in the exact order of MASTER 2.5 (drift is handled before, see apply_drift_if_due)."""
        cfg, N, dev = self.cfg, self.N, self.device
        tm = _Sections(dev, self.profile)
        tm.start()
        eligible = eligible_mask(self.bat.level, self.n, cfg)
        emask = epochs_mask(self.bat.level, self.n, cfg)
        act = self._parse_action(action, eligible, emask)
        sel, m = act.selected, len(act.selected)
        before_training = cfg.fl.blend_mode == "before_training"
        G = self.G
        gval_before = self.global_val_acc()
        p10_before = self._p10(self.probe_acc)
        seed = self.scenario.spec.seed

        # c. local training from G (own lr / epochs / batch)
        trained: List[torch.Tensor] = []
        shadows: Dict[int, torch.Tensor] = {}
        with tm.section("train"):
            for j, i in enumerate(sel):
                idx = self.scenario.clients[i].train_idx.to(dev)
                start = None
                if before_training and self.L[i] is not None:
                    start = blend(act.alpha_all[i], G, self.L[i])
                args = (self.model, G, self.X, self.y, idx, act.lr_vals[j], act.epochs_vals[j],
                        act.batch_vals[j], act.prox_mu)
                w = local_train(*args, make_torch_gen(seed, f"local_{round_t}_{i}"), init_vec=start)
                if start is not None:  # local-only reference for the personalization gain (not charged)
                    shadows[i] = local_train(*args, make_torch_gen(seed, f"local_{round_t}_{i}"))
                trained.append(w)
                self.divergence[i] = rel_div(w, G)
        for i, w in zip(sel, trained):
            self.L[i] = w

        # d. aggregate (sample-count weights, optional server momentum)
        with tm.section("aggregate"):
            if m > 0:
                sel_t = torch.as_tensor(sel, dtype=torch.long, device=dev)
                G_new, self.mom = aggregate(torch.stack(trained), self.n[sel_t], G, self.mom,
                                            float(cfg.fl.server_momentum))
                agg = (G_new - G).unsqueeze(0)
                for i, w in zip(sel, trained):
                    self.update_cos[i] = F.cosine_similarity((w - G).unsqueeze(0), agg).item()
                self.G = G_new

        # e/f. measure: global accuracy, personalization gains
        with tm.section("global_eval"):
            gval_after = self.global_val_acc()
        with tm.section("pers_eval"):
            gain_chosen, gain_all = self._pers_gains(sel, act.alpha_all, shadows, before_training)

        # costs, battery, counters
        comm, comp, latency = round_costs(sel, act.epochs_vals, self.n, self.speed, self.bw,
                                          self.model_bytes, cfg)
        self.bat.apply_drain(sel, act.epochs_vals)
        self.bat.recharge()
        picked = torch.zeros(N, dtype=torch.bool, device=dev)
        if m:
            picked[torch.as_tensor(sel, dtype=torch.long, device=dev)] = True
        self.times_picked += picked.float()
        self.idle_rounds = torch.where(picked, torch.zeros_like(self.idle_rounds), self.idle_rounds + 1)
        self.rounds_since_local = torch.where(picked, torch.zeros_like(self.rounds_since_local),
                                              self.rounds_since_local + 1)

        # probe G_new on all clients: next round's probe (free)
        with tm.section("probe"):
            if m > 0:
                self._store_probe(*self.probe_all())
            else:
                self._store_probe(self.probe_loss, self.probe_acc)
        p10_after = self._p10(self.probe_acc)

        return RoundResult(
            selected=list(sel), epochs_vals=list(act.epochs_vals),
            global_val_acc_before=gval_before, global_val_acc_after=gval_after,
            fair_p10_before=p10_before, fair_p10_after=p10_after,
            pers_gain_chosen=gain_chosen, pers_gain_all=gain_all,
            comm=comm, comp=comp, latency=latency,
            divergence=self.divergence.clone(), probe_loss=self.probe_loss.clone(),
            probe_acc=self.probe_acc.clone(), battery=self.bat.level.clone(),
            rounds_since_local=self.rounds_since_local.clone(), times_picked=self.times_picked.clone(),
            idle_rounds=self.idle_rounds.clone(),
            alpha_vals=torch.tensor(act.alpha_all, dtype=torch.float32, device=dev),
            lr_vals=list(act.lr_vals), batch_vals=list(act.batch_vals),
            update_cos=self.update_cos.clone(),
            prev_probe_loss=self.prev_probe_loss.clone(), prev_probe_acc=self.prev_probe_acc.clone(),
            n_probes=self.n_probes, timings=tm.finish())

    def _pers_gains(self, chosen: List[int], alpha_all: List[float], shadows: Dict[int, torch.Tensor],
                    before_training: bool) -> Tuple[float, float]:
        """Mean gain (personal-val accuracy of B_i minus local-only L_i) over chosen and over all clients."""
        log_all = bool(self.cfg.fl.log_all_client_gain)
        chosen_set = set(chosen)
        gains: Dict[int, float] = {}
        for i in (range(self.N) if log_all else chosen):
            li = self.L[i]
            if li is None or alpha_all[i] == 0.0:       # never trained (D5) / alpha=0: gain exactly 0
                gains[i] = 0.0
            elif before_training and i in chosen_set:
                gains[i] = (self._pers_acc(i, li) - self._pers_acc(i, shadows[i])) if i in shadows else 0.0
            else:
                b = blend(alpha_all[i], self.G, li)
                gains[i] = self._pers_acc(i, b) - self._pers_acc(i, li)
        g_ch = float(np.mean([gains[i] for i in chosen])) if chosen else 0.0
        g_all = float(np.mean([gains[i] for i in range(self.N)])) if log_all else float("nan")
        return g_ch, g_all