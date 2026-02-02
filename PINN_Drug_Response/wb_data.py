from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


READOUTS = {"pERK", "pMEK", "pAKT", "pS6", "pS6K", "totalERK", "totalAKT"}


@dataclass
class WBCondition:
    condition: str
    drug_vem_uM: float
    drug_tram_nM: float
    drug_pi3k_uM: float
    drug_panras_uM: float
    time_min: np.ndarray
    readouts: Dict[str, np.ndarray]
    sems: Dict[str, Optional[np.ndarray]]


def load_western_blot_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "condition",
        "drug_vem_uM",
        "drug_tram_nM",
        "drug_pi3k_uM",
        "drug_panras_uM",
        "time_min",
        "readout",
        "value",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def preprocess_wb_dataframe(
    df: pd.DataFrame,
    normalize_to_t0: bool = True,
    log_transform: bool = False,
) -> pd.DataFrame:
    data = df.copy()
    data["readout"] = data["readout"].astype(str)
    data = data[data["readout"].isin(READOUTS)]

    if normalize_to_t0:
        grouped = data.groupby(["condition", "readout"], as_index=False)
        t0 = grouped.apply(lambda x: x.loc[x["time_min"].idxmin(), "value"]).rename(
            columns={None: "t0"}
        )
        data = data.merge(t0, on=["condition", "readout"], how="left")
        data["value"] = data["value"] / data["t0"].replace(0, np.nan)
        data = data.drop(columns=["t0"])

    if log_transform:
        data["value"] = np.log1p(data["value"])

    return data


def build_conditions(df: pd.DataFrame) -> List[WBCondition]:
    conditions: List[WBCondition] = []
    for cond_name, cond_df in df.groupby("condition"):
        summary = cond_df.iloc[0]
        readouts: Dict[str, np.ndarray] = {}
        sems: Dict[str, Optional[np.ndarray]] = {}
        for readout, ro_df in cond_df.groupby("readout"):
            ro_sorted = ro_df.sort_values("time_min")
            readouts[readout] = ro_sorted["value"].to_numpy(dtype=float)
            if "value_sem" in ro_df.columns:
                sems[readout] = ro_sorted["value_sem"].to_numpy(dtype=float)
            else:
                sems[readout] = None
        time_min = (
            cond_df.sort_values("time_min")["time_min"].drop_duplicates().to_numpy(dtype=float)
        )
        conditions.append(
            WBCondition(
                condition=cond_name,
                drug_vem_uM=float(summary["drug_vem_uM"]),
                drug_tram_nM=float(summary["drug_tram_nM"]),
                drug_pi3k_uM=float(summary["drug_pi3k_uM"]),
                drug_panras_uM=float(summary["drug_panras_uM"]),
                time_min=time_min,
                readouts=readouts,
                sems=sems,
            )
        )
    return conditions


def filter_conditions(
    conditions: Iterable[WBCondition],
    include_conditions: Optional[List[str]] = None,
) -> List[WBCondition]:
    if include_conditions is None:
        return list(conditions)
    return [c for c in conditions if c.condition in include_conditions]
