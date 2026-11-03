"""Scenario: N clients with data, proportions and resources, built deterministically from spec.seed.

Imports ``get_official_test`` ONLY to build evaluation-only personal test sets (allowed exception).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from fedrl.data import drift as drift_mod
from fedrl.data.datasets import DatasetBundle, get_official_test
from fedrl.data.partition import draw_client_data, draw_indices, sample_client_sizes, sample_proportions
from fedrl.data.pools import Pools
from fedrl.data.scenario_sets import ScenarioSpec
from fedrl.utils.logging_utils import get_logger
from fedrl.utils.seeding import make_rng

log = get_logger(__name__)


@dataclass
class ClientData:
    """Everything one client owns. Index tensors point into shared on-device data (no copies)."""

    n: int
    proportions: np.ndarray
    train_idx: torch.Tensor
    val_idx: torch.Tensor
    personal_val_idx: torch.Tensor
    compute_speed: float
    bandwidth: float
    battery0: float


def sample_resources(rng: np.random.Generator, n_clients: int, cfg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compute_speed, bandwidth ~ lognormal(median 1, sigma); battery0 ~ U(low, high)."""
    c = cfg.clients
    sig = float(c.resource_sigma)
    speed = rng.lognormal(0.0, sig, size=int(n_clients))
    bw = rng.lognormal(0.0, sig, size=int(n_clients))
    b0 = rng.uniform(float(c.battery_init_low), float(c.battery_init_high), size=int(n_clients))
    return speed, bw, b0


class Scenario:
    """A fully specified FL scenario. Attributes: N, spec, clients, init_seed."""

    def __init__(self, spec: ScenarioSpec, bundle: DatasetBundle, pools: Pools, cfg) -> None:
        drift_mod.check_drift_kind(spec.drift)
        self.spec, self.bundle, self.pools, self.cfg = spec, bundle, pools, cfg
        self.N = int(cfg.fl.N)
        self.num_classes = int(bundle.num_classes)
        seed = int(spec.seed)
        self.sizes = sample_client_sizes(make_rng(seed, "sizes"), self.N, cfg)
        self.base_proportions = sample_proportions(make_rng(seed, "proportions"), spec.a, self.num_classes, self.N)
        speed, bw, b0 = sample_resources(make_rng(seed, "resources"), self.N, cfg)
        self.init_seed = int(make_rng(seed, "init").integers(0, 2 ** 31 - 1))
        self.data_version = 0
        self._applied: set[int] = set()
        self._target: np.ndarray | None = None
        self._test_by_class: list[np.ndarray] | None = None
        self._ptest_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        self.clients: list[ClientData] = []
        for i in range(self.N):
            d = draw_client_data(seed, i, int(self.sizes[i]), self.base_proportions[i], pools, cfg, "init")
            self.clients.append(self._make_client(d, self.base_proportions[i].copy(),
                                                  float(speed[i]), float(bw[i]), float(b0[i])))

    # ------------------------------------------------------------------ helpers
    def _t(self, arr: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.long, device=self.bundle.device)

    def _make_client(self, d, props, speed, bw, b0) -> ClientData:
        return ClientData(n=len(d.train_idx), proportions=props, train_idx=self._t(d.train_idx),
                          val_idx=self._t(d.val_idx), personal_val_idx=self._t(d.personal_val_idx),
                          compute_speed=speed, bandwidth=bw, battery0=b0)

    @property
    def target_proportions(self) -> np.ndarray:
        """Post-drift proportions p_new [N, C] (deterministic from the seed; unused when drift is none)."""
        if self._target is None:
            self._target = drift_mod.target_proportions(int(self.spec.seed), self.spec.a, self.N, self.num_classes)
        return self._target

    def proportions_matrix(self) -> np.ndarray:
        """Current class proportions of all clients, [N, C]."""
        return np.stack([c.proportions for c in self.clients])

    # ------------------------------------------------------------------ drift
    def apply_drift(self, round_t: int) -> bool:
        """Apply drift due at the START of round ``round_t`` (1-based). Returns True if any client's
        data changed. Calling it again for an already-applied round returns False."""
        round_t = int(round_t)
        if round_t < 1:
            raise ValueError(f"round_t is 1-based, got {round_t}")
        if round_t in self._applied or not drift_mod.drift_due(self.spec.drift, round_t, self.cfg):
            return False
        p_new = self.target_proportions if self.spec.drift != "none" else self.base_proportions
        upd = drift_mod.apply_drift(self.spec.drift, round_t, int(self.spec.seed), self.base_proportions,
                                    p_new, self.sizes, self.pools, self.cfg)
        for i, d in enumerate(upd.clients):
            old = self.clients[i]
            self.clients[i] = self._make_client(d, upd.proportions[i].copy(),
                                                old.compute_speed, old.bandwidth, old.battery0)
        self._applied.add(round_t)
        self.data_version += 1
        self._ptest_cache.clear()
        log.info("Drift (%s) applied at round %d", self.spec.drift, round_t)
        return True

    # ------------------------------------------------------------------ evaluation only
    def _test_pools(self) -> list[np.ndarray]:
        if self._test_by_class is None:
            labels = get_official_test(self.bundle)[1].detach().cpu().numpy()
            self._test_by_class = [np.flatnonzero(labels == c).astype(np.int64) for c in range(self.num_classes)]
        return self._test_by_class

    def personal_test_idx(self, i: int, eval_round: int | None = None) -> torch.Tensor:
        """EVALUATION ONLY. ``personal_test_size`` official-test indices following client i's CURRENT
        proportions. The RNG stream is keyed by (scenario seed, client id, evaluation round), so every
        method sees identical sets (D14). ``eval_round`` defaults to ``cfg.fl.T``."""
        r = int(self.cfg.fl.T) if eval_round is None else int(eval_round)
        key = (int(i), r, self.data_version)
        if key not in self._ptest_cache:
            rng = make_rng(int(self.spec.seed), f"personal_test_{int(i)}_{r}")
            idx = draw_indices(rng, int(self.cfg.data.personal_test_size), self.clients[i].proportions,
                               self._test_pools(), "personal-test")
            self._ptest_cache[key] = self._t(idx)
        return self._ptest_cache[key]