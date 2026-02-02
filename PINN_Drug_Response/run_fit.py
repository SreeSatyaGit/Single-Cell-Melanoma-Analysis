from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from fitting import FitConfig, fit_ensemble, save_ensemble, save_params
from ode_model import default_parameters
from sanity_checks import run_sanity_checks
from wb_data import build_conditions, load_western_blot_csv, preprocess_wb_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit MAPK/PI3K ODE model to western blot data.")
    parser.add_argument("--data", required=True, help="Path to western blot CSV file.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output", default="fit_outputs", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_cfg = config.get("data", {})
    fit_cfg = config.get("fit", {})
    seed = int(config.get("seed", 0))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_western_blot_csv(args.data)
    df = preprocess_wb_dataframe(
        df,
        normalize_to_t0=bool(data_cfg.get("normalize_to_t0", True)),
        log_transform=bool(data_cfg.get("log_transform", False)),
    )
    conditions = build_conditions(df)
    train_conditions = [
        c
        for c in conditions
        if (c.drug_vem_uM > 0 and c.drug_tram_nM == 0)
        or (c.drug_vem_uM == 0 and c.drug_tram_nM > 0)
        or (c.drug_vem_uM > 0 and c.drug_tram_nM > 0)
    ]

    fixed_params = default_parameters()
    fixed_params.update(config.get("fixed_params", {}))
    fit_param_names = fit_cfg.get(
        "fit_param_names",
        [
            "k_craf_act",
            "k_mek_act",
            "k_erk_act",
            "k_akt_act",
            "IC50_vem",
            "n_vem",
            "alpha_vem",
            "IC50_tram",
            "n_tram",
        ],
    )

    fit_config = FitConfig(
        fit_param_names=fit_param_names,
        fixed_params=fixed_params,
        use_correction_term=bool(fit_cfg.get("use_correction_term", False)),
        correction_width=int(fit_cfg.get("correction_width", 8)),
        l2_correction=float(fit_cfg.get("l2_correction", 1e-3)),
        l2_params=float(fit_cfg.get("l2_params", 1e-4)),
        method=str(fit_cfg.get("solver", "BDF")),
    )

    ensemble = fit_ensemble(
        conditions=train_conditions,
        config=fit_config,
        n_models=int(fit_cfg.get("n_models", 5)),
        seed=seed,
    )
    save_ensemble(str(output_dir), ensemble)

    best = min(ensemble, key=lambda x: x["loss"])
    save_params(str(output_dir / "best_fit.json"), best)

    checks = run_sanity_checks(best["params"])
    with open(output_dir / "sanity_checks.json", "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2)

    print(f"Saved {len(ensemble)} fits to {output_dir}")


if __name__ == "__main__":
    main()
