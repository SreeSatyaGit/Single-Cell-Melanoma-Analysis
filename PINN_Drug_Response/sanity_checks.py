from __future__ import annotations

from typing import Dict

import numpy as np

from ode_model import SPECIES_ORDER, default_initial_state, default_parameters, make_constant_schedule, rhs
from prediction import PredictionConfig
from scipy.integrate import solve_ivp


def _simulate_endpoint(params: Dict[str, float], schedule: Dict[str, float]) -> np.ndarray:
    config = PredictionConfig(time_min=0.0, time_max=120.0, n_timepoints=25)
    t_eval = np.linspace(config.time_min, config.time_max, config.n_timepoints)
    schedule_obj = make_constant_schedule(
        vem=schedule.get("vem", 0.0),
        tram=schedule.get("tram", 0.0),
        pi3k=schedule.get("pi3k", 0.0),
        panras=schedule.get("panras", 0.0),
    )
    sol = solve_ivp(
        lambda t, y: rhs(t, y, params, schedule_obj, None),
        (t_eval.min(), t_eval.max()),
        default_initial_state(),
        t_eval=t_eval,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        max_step=config.max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Solver failed during sanity checks: {sol.message}")
    return sol.y[:, -1]


def run_sanity_checks(params: Dict[str, float]) -> Dict[str, bool]:
    species_index = {name: idx for idx, name in enumerate(SPECIES_ORDER)}
    checks = {}

    vem_low = _simulate_endpoint(params, {"vem": 0.1})
    vem_high = _simulate_endpoint(params, {"vem": 1.0})
    checks["vem_reduces_braf_activity"] = vem_high[species_index["pMEK"]] <= vem_low[
        species_index["pMEK"]
    ]

    tram_low = _simulate_endpoint(params, {"tram": 0.1})
    tram_high = _simulate_endpoint(params, {"tram": 1.0})
    checks["tram_reduces_pERK"] = tram_high[species_index["pERK"]] <= tram_low[
        species_index["pERK"]
    ]

    pi3k_low = _simulate_endpoint(params, {"pi3k": 0.1})
    pi3k_high = _simulate_endpoint(params, {"pi3k": 1.0})
    checks["pi3k_reduces_pAKT"] = pi3k_high[species_index["pAKT"]] <= pi3k_low[
        species_index["pAKT"]
    ]

    panras_low = _simulate_endpoint(params, {"panras": 0.1})
    panras_high = _simulate_endpoint(params, {"panras": 1.0})
    checks["panras_reduces_pERK"] = panras_high[species_index["pERK"]] <= panras_low[
        species_index["pERK"]
    ]
    checks["panras_reduces_pAKT"] = panras_high[species_index["pAKT"]] <= panras_low[
        species_index["pAKT"]
    ]

    return checks


if __name__ == "__main__":
    params = default_parameters()
    results = run_sanity_checks(params)
    for key, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{key}: {status}")
