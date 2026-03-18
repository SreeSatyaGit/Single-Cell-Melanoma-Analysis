from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ode_model import (
    READOUT_TO_SPECIES,
    SPECIES_ORDER,
    build_correction_fn,
    default_initial_state,
    default_parameters,
    init_correction_weights,
    make_constant_schedule,
    rhs,
)
from wb_data import WBCondition


@dataclass
class FitConfig:
    fit_param_names: List[str]
    fixed_params: Dict[str, float]
    use_correction_term: bool = False
    correction_width: int = 8
    l2_correction: float = 1e-3
    l2_params: float = 1e-4
    t_span_min: float = 0.0
    t_span_max: float = 2880.0
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 10.0
    method: str = "BDF"


def _pack_params(params: Dict[str, float], fit_param_names: List[str]) -> np.ndarray:
    return np.array([np.log(params[name]) for name in fit_param_names], dtype=float)


def _unpack_params(
    theta: np.ndarray, base_params: Dict[str, float], fit_param_names: List[str]
) -> Dict[str, float]:
    params = dict(base_params)
    for idx, name in enumerate(fit_param_names):
        params[name] = float(np.exp(theta[idx]))
    return params


def simulate_condition(
    condition: WBCondition,
    params: Dict[str, float],
    y0: np.ndarray,
    correction_weights: Optional[Dict[str, np.ndarray]],
    config: FitConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    schedule = make_constant_schedule(
        vem=condition.drug_vem_uM,
        tram=condition.drug_tram_nM,
        pi3k=condition.drug_pi3k_uM,
        panras=condition.drug_panras_uM,
    )
    correction_fn = build_correction_fn(correction_weights, config.use_correction_term)
    t_eval = condition.time_min
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params, schedule, correction_fn),
        (t_eval.min(), t_eval.max()),
        y0,
        t_eval=t_eval,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed for condition {condition.condition}: {sol.message}")
    return sol.t, sol.y.T


def build_residuals(
    theta: np.ndarray,
    conditions: Sequence[WBCondition],
    base_params: Dict[str, float],
    y0: np.ndarray,
    config: FitConfig,
    correction_weights: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    params = _unpack_params(theta, base_params, config.fit_param_names)
    residuals: List[float] = []
    for condition in conditions:
        t_eval, y_pred = simulate_condition(condition, params, y0, correction_weights, config)
        species_index = {name: idx for idx, name in enumerate(SPECIES_ORDER)}
        for readout, values in condition.readouts.items():
            species_name = READOUT_TO_SPECIES.get(readout, readout)
            if species_name not in species_index:
                continue
            preds = y_pred[:, species_index[species_name]]
            sem = condition.sems.get(readout)
            if sem is None:
                sem = np.ones_like(values)
            residuals.extend(((preds - values) / sem).tolist())
    if config.l2_params > 0.0:
        residuals.extend(np.sqrt(config.l2_params) * theta.tolist())
    if config.use_correction_term and correction_weights is not None:
        penalty = config.l2_correction * np.sum(
            np.concatenate(
                [
                    correction_weights["w1"].ravel(),
                    correction_weights["b1"].ravel(),
                    correction_weights["w2"].ravel(),
                    correction_weights["b2"].ravel(),
                ]
            )
            ** 2
        )
        residuals.append(np.sqrt(penalty))
    return np.array(residuals, dtype=float)


def fit_model(
    conditions: Sequence[WBCondition],
    config: FitConfig,
    seed: int = 0,
    initial_params: Optional[Dict[str, float]] = None,
    y0: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], Optional[Dict[str, np.ndarray]], float]:
    rng = np.random.default_rng(seed)
    params = default_parameters()
    params.update(config.fixed_params)
    if initial_params:
        params.update(initial_params)

    y0 = y0 if y0 is not None else default_initial_state()
    theta0 = _pack_params(params, config.fit_param_names)
    theta0 = theta0 + rng.normal(scale=0.1, size=theta0.shape)

    correction_weights = None
    if config.use_correction_term:
        correction_weights = init_correction_weights(rng, width=config.correction_width)

    result = least_squares(
        build_residuals,
        theta0,
        loss="soft_l1",
        args=(conditions, params, y0, config, correction_weights),
        max_nfev=200,
    )
    best_params = _unpack_params(result.x, params, config.fit_param_names)
    return best_params, correction_weights, float(result.cost)


def fit_ensemble(
    conditions: Sequence[WBCondition],
    config: FitConfig,
    n_models: int = 10,
    seed: int = 0,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for idx in range(n_models):
        fit_params, correction_weights, loss = fit_model(
            conditions=conditions,
            config=config,
            seed=seed + idx,
        )
        results.append(
            {
                "params": fit_params,
                "correction_weights": correction_weights,
                "loss": loss,
                "seed": seed + idx,
            }
        )
    return results


def save_params(path: str, params: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)


def save_ensemble(directory: str, ensemble: List[Dict[str, object]]) -> None:
    for idx, entry in enumerate(ensemble):
        save_params(f"{directory}/fit_{idx:02d}.json", entry)
