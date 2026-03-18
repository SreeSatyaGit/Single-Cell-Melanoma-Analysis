from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from prediction import PredictionConfig, build_output_rows, load_param_set, simulate, summarize_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict new drug combinations.")
    parser.add_argument("--params", required=True, help="Path to parameter JSON or directory.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--mode", required=True, choices=["vem_pi3k", "vem_panras"])
    parser.add_argument("--output", default="predictions", help="Output directory.")
    return parser.parse_args()


def _load_param_sets(path: Path) -> list:
    if path.is_dir():
        param_files = sorted(path.glob("fit_*.json"))
        return [load_param_set(str(p)) for p in param_files]
    return [load_param_set(str(path))]


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    pred_cfg = config.get("prediction", {})
    schedule_cfg = pred_cfg.get("schedules", {})
    readouts = pred_cfg.get("readouts", ["pERK", "pMEK", "pAKT", "pS6"])

    pred_config = PredictionConfig(
        time_min=float(pred_cfg.get("time_min", 0.0)),
        time_max=float(pred_cfg.get("time_max", 180.0)),
        n_timepoints=int(pred_cfg.get("n_timepoints", 120)),
        method=str(pred_cfg.get("solver", "BDF")),
    )

    param_sets = _load_param_sets(Path(args.params))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "vem_pi3k":
        vem_doses = schedule_cfg.get("vem_doses", [1.0])
        pi3k_doses = schedule_cfg.get("pi3k_doses", [0.1, 0.3, 1.0])
        combos = [
            {"vem": vem, "pi3k": pi3k, "tram": 0.0, "panras": 0.0}
            for vem in vem_doses
            for pi3k in pi3k_doses
        ]
    else:
        vem_doses = schedule_cfg.get("vem_doses", [1.0])
        panras_doses = schedule_cfg.get("panras_doses", [0.1, 0.3, 1.0])
        combos = [
            {"vem": vem, "panras": panras, "tram": 0.0, "pi3k": 0.0}
            for vem in vem_doses
            for panras in panras_doses
        ]

    rows = []
    for combo in combos:
        label = f"{args.mode}_vem{combo.get('vem', 0)}"
        label += f"_pi3k{combo.get('pi3k', 0)}" if "pi3k" in combo else ""
        label += f"_panras{combo.get('panras', 0)}" if "panras" in combo else ""

        trajectories = []
        times = None
        for entry in param_sets:
            params = entry["params"]
            correction = entry.get("correction_weights")
            t, y = simulate(params, combo, pred_config, correction)
            trajectories.append(y)
            times = t
        if len(trajectories) == 1:
            rows.extend(build_output_rows(times, trajectories[0], readouts, label))
        else:
            median, lower, upper = summarize_ensemble(trajectories)
            rows.extend(build_output_rows(times, median, readouts, label, (lower, upper)))

    df = pd.DataFrame(rows)
    output_path = output_dir / f"predictions_{args.mode}.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
