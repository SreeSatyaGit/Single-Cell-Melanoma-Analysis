from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from ode_model import (
    READOUT_TO_SPECIES,
    SPECIES_ORDER,
    build_correction_fn,
    default_initial_state,
    make_constant_schedule,
    rhs,
)


@dataclass
class PredictionConfig:
    time_min: float
    time_max: float
    n_timepoints: int
    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 10.0


def load_param_set(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def simulate(
    params: Dict[str, float],
    schedule: Dict[str, float],
    config: PredictionConfig,
    correction_weights: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    t_eval = np.linspace(config.time_min, config.time_max, config.n_timepoints)
    schedule_obj = make_constant_schedule(
        vem=schedule.get("vem", 0.0),
        tram=schedule.get("tram", 0.0),
        pi3k=schedule.get("pi3k", 0.0),
        panras=schedule.get("panras", 0.0),
    )
    correction_fn = build_correction_fn(correction_weights, correction_weights is not None)
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params, schedule_obj, correction_fn),
        (t_eval.min(), t_eval.max()),
        default_initial_state(),
        t_eval=t_eval,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.t, sol.y.T


def summarize_ensemble(
    trajectories: List[np.ndarray],
    percentiles: Tuple[float, float] = (10, 90),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(trajectories, axis=0)
    median = np.median(stack, axis=0)
    lower = np.percentile(stack, percentiles[0], axis=0)
    upper = np.percentile(stack, percentiles[1], axis=0)
    return median, lower, upper


def build_output_rows(
    times: np.ndarray,
    species: np.ndarray,
    readouts: Iterable[str],
    label: str,
    percentiles: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> List[Dict[str, object]]:
    species_index = {name: idx for idx, name in enumerate(SPECIES_ORDER)}
    rows: List[Dict[str, object]] = []
    for readout in readouts:
        species_name = READOUT_TO_SPECIES.get(readout, readout)
        if species_name not in species_index:
            continue
        vals = species[:, species_index[species_name]]
        for idx, t in enumerate(times):
            row = {
                "condition": label,
                "time_min": float(t),
                "readout": readout,
                "value": float(vals[idx]),
            }
            if percentiles is not None:
                lower, upper = percentiles
                row["p10"] = float(lower[idx, species_index[species_name]])
                row["p90"] = float(upper[idx, species_index[species_name]])
            rows.append(row)
    return rows
