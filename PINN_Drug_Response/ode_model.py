from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np


SPECIES_ORDER = [
    "pEGFR",
    "HER2",
    "HER3",
    "pDGFR",
    "pCRAF",
    "pMEK",
    "pERK",
    "DUSP6",
    "pAKT",
    "pS6K",
    "p4EBP1",
]

READOUT_TO_SPECIES = {
    "pERK": "pERK",
    "pMEK": "pMEK",
    "pAKT": "pAKT",
    "pS6": "pS6K",
    "pS6K": "pS6K",
    "pS6k": "pS6K",
    "totalERK": "pERK",
    "totalAKT": "pAKT",
}


def hill_inhibition(conc: float, ic50: float, hill_n: float) -> float:
    return (conc**hill_n) / (ic50**hill_n + conc**hill_n + 1e-12)


@dataclass
class DrugSchedule:
    vem: Callable[[float], float]
    tram: Callable[[float], float]
    pi3k: Callable[[float], float]
    panras: Callable[[float], float]

    def evaluate(self, t: float) -> Tuple[float, float, float, float]:
        return (self.vem(t), self.tram(t), self.pi3k(t), self.panras(t))


def make_constant_schedule(
    vem: float = 0.0,
    tram: float = 0.0,
    pi3k: float = 0.0,
    panras: float = 0.0,
) -> DrugSchedule:
    return DrugSchedule(
        vem=lambda _t: vem,
        tram=lambda _t: tram,
        pi3k=lambda _t: pi3k,
        panras=lambda _t: panras,
    )


def default_parameters() -> Dict[str, float]:
    return {
        "Km": 0.5,
        "hill_coeff": 2.0,
        "IC50_vem": 0.8,
        "n_vem": 2.0,
        "alpha_vem": 0.6,
        "IC50_tram": 0.4,
        "n_tram": 2.0,
        "IC50_pi3k": 0.5,
        "n_pi3k": 2.0,
        "IC50_panras": 0.5,
        "n_panras": 2.0,
        "k_dusp_synth": 0.8,
        "k_dusp_deg": 0.5,
        "Km_dusp": 0.4,
        "n_dusp": 2.5,
        "k_dusp_cat": 0.6,
        "k_erk_sos": 0.4,
        "k_mek_inhib": 0.2,
        "k_s6k_irs": 0.7,
        "Km_s6k": 0.5,
        "k_s6k_mtor": 0.3,
        "k_4ebp1_comp": 0.25,
        "k_akt_rtk": 0.15,
        "k_akt_raf": 0.5,
        "k_erk_pi3k": 0.45,
        "k_raf_pi3k": 0.2,
        "k_erk_rtk": 0.1,
        "Km_erk_rtk": 0.5,
        "k_egfr_phos": 0.5,
        "k_egfr_dephos": 0.2,
        "k_her_phos": 0.4,
        "k_her_dephos": 0.15,
        "k_pdgfr_phos": 0.3,
        "k_pdgfr_dephos": 0.2,
        "w_her3": 1.5,
        "k_craf_act": 1.2,
        "k_craf_deg": 0.35,
        "k_mek_act": 1.0,
        "k_mek_deg": 0.4,
        "k_erk_act": 1.2,
        "k_erk_deg": 0.45,
        "k_akt_act": 1.0,
        "k_akt_deg": 0.4,
        "k_s6k_act": 0.9,
        "k_s6k_deg": 0.5,
        "k_4ebp1_act": 0.85,
        "k_4ebp1_deg": 0.45,
    }


def default_initial_state() -> np.ndarray:
    return np.array(
        [
            0.25,  # pEGFR
            0.25,  # HER2
            0.25,  # HER3
            0.25,  # pDGFR
            0.25,  # pCRAF
            0.25,  # pMEK
            0.25,  # pERK
            0.25,  # DUSP6
            0.25,  # pAKT
            0.25,  # pS6K
            0.25,  # p4EBP1
        ],
        dtype=float,
    )


def compute_drug_effects(
    t: float, params: Dict[str, float], schedule: DrugSchedule
) -> Dict[str, float]:
    vem, tram, pi3k, panras = schedule.evaluate(t)
    I_vem = hill_inhibition(vem, params["IC50_vem"], params["n_vem"])
    I_tram = hill_inhibition(tram, params["IC50_tram"], params["n_tram"])
    I_pi3k = hill_inhibition(pi3k, params["IC50_pi3k"], params["n_pi3k"])
    I_panras = hill_inhibition(panras, params["IC50_panras"], params["n_panras"])
    return {
        "vem": vem,
        "tram": tram,
        "pi3k": pi3k,
        "panras": panras,
        "I_vem": I_vem,
        "I_tram": I_tram,
        "I_pi3k": I_pi3k,
        "I_panras": I_panras,
    }


def rhs(
    t: float,
    y: np.ndarray,
    params: Dict[str, float],
    schedule: DrugSchedule,
    correction_fn: Optional[Callable[[np.ndarray], float]] = None,
) -> np.ndarray:
    y = np.clip(y, 0.0, None)
    (
        pEGFR,
        HER2,
        HER3,
        pDGFR,
        pCRAF,
        pMEK,
        pERK,
        DUSP6,
        pAKT,
        pS6K,
        p4EBP1,
    ) = y

    effects = compute_drug_effects(t, params, schedule)
    Km = params["Km"]

    Tram_effect = effects["I_tram"]
    PI3Ki_effect = effects["I_pi3k"]
    Vem_bound_braf = effects["I_vem"]
    PanRAS_effect = effects["I_panras"]

    k_dusp_synth = params["k_dusp_synth"]
    k_dusp_deg = params["k_dusp_deg"]
    Km_dusp = params["Km_dusp"]
    n_dusp = params["n_dusp"]
    k_dusp_cat = params["k_dusp_cat"]
    k_erk_sos = params["k_erk_sos"]
    k_mek_inhib = params["k_mek_inhib"]

    DUSP6_synthesis = k_dusp_synth * (pERK**n_dusp) / (Km_dusp**n_dusp + pERK**n_dusp + 1e-12)
    DUSP6_inhibition = (k_dusp_cat * DUSP6) / (Km + DUSP6 + 1e-12)

    ERK_to_SOS_inhibition = (k_erk_sos * pERK) / (Km + pERK + 1e-12)
    MEK_substrate_inhibition = k_mek_inhib * pMEK / (Km + pMEK + 1e-12)

    k_s6k_irs = params["k_s6k_irs"]
    Km_s6k = params["Km_s6k"]
    k_s6k_mtor = params["k_s6k_mtor"]
    k_4ebp1_comp = params["k_4ebp1_comp"]
    k_akt_rtk = params["k_akt_rtk"]
    k_akt_raf = params["k_akt_raf"]
    k_erk_pi3k = params["k_erk_pi3k"]
    k_raf_pi3k = params["k_raf_pi3k"]
    k_erk_rtk = params["k_erk_rtk"]

    S6K_to_IRS1_inhibition = (k_s6k_irs * pS6K) / (Km_s6k + pS6K + 1e-12)
    S6K_to_mTOR_feedback = (k_s6k_mtor * pS6K) / (Km + pS6K + 1e-12)
    mTOR_4EBP1_competition = k_4ebp1_comp * p4EBP1 / (Km + p4EBP1 + 1e-12)
    mTOR_total_feedback = S6K_to_IRS1_inhibition + S6K_to_mTOR_feedback
    AKT_to_RTK_feedback = k_akt_rtk * pAKT / (Km + pAKT + 1e-12)
    AKT_to_RAF_inhibition = (k_akt_raf * pAKT) / (Km + pAKT + 1e-12)
    ERK_to_PI3K_inhibition = (k_erk_pi3k * pERK) / (Km + pERK + 1e-12)

    ERK_feedback = (k_erk_rtk * pERK) / (params["Km_erk_rtk"] + pERK + 1e-12)

    k_egfr_phos = params["k_egfr_phos"]
    k_egfr_dephos = params["k_egfr_dephos"]
    k_her_phos = params["k_her_phos"]
    k_her_dephos = params["k_her_dephos"]
    k_pdgfr_phos = params["k_pdgfr_phos"]
    k_pdgfr_dephos = params["k_pdgfr_dephos"]

    res_pEGFR = k_egfr_phos * (1.0 - pEGFR) - (k_egfr_dephos + ERK_feedback) * pEGFR
    res_HER2 = k_her_phos * (1.0 - HER2) - (k_her_dephos + ERK_feedback) * HER2
    res_HER3 = k_her_phos * (1.0 - HER3) - (k_her_dephos + ERK_feedback) * HER3
    res_pDGFR = k_pdgfr_phos * (1.0 - pDGFR) - (k_pdgfr_dephos + ERK_feedback) * pDGFR

    w_her3 = params["w_her3"]
    RTK_base = pEGFR + HER2 + w_her3 * HER3 + pDGFR

    RAS_GTP = (
        RTK_base
        * (1.0 - ERK_to_SOS_inhibition)
        * (1.0 - AKT_to_RTK_feedback)
        * (1.0 - PanRAS_effect)
    )
    if correction_fn is not None:
        RAS_GTP = RAS_GTP * (1.0 + correction_fn(np.array([pERK, pAKT])))

    PI3K_input = RTK_base * (1.0 - ERK_to_PI3K_inhibition) * (1.0 - mTOR_total_feedback)
    PI3K_total_input = PI3K_input + (k_raf_pi3k * pCRAF) / (Km + pCRAF + 1e-12)

    k_craf_act = params["k_craf_act"]
    k_craf_deg = params["k_craf_deg"]

    alpha_vem = params["alpha_vem"]

    res_pCRAF = (
        k_craf_act * RAS_GTP
        + alpha_vem * Vem_bound_braf
        - k_craf_deg * pCRAF
        - AKT_to_RAF_inhibition * pCRAF
    )

    k_mek_act = params["k_mek_act"]
    k_mek_deg = params["k_mek_deg"]

    res_pMEK = (
        k_mek_act
        * pCRAF
        * (1.0 - Vem_bound_braf)
        * (1.0 - Tram_effect)
        * (1.0 - AKT_to_RAF_inhibition)
        - k_mek_deg * pMEK
        - MEK_substrate_inhibition * pMEK
    )

    k_erk_act = params["k_erk_act"]
    k_erk_deg = params["k_erk_deg"]

    res_pERK = k_erk_act * pMEK * (1.0 - DUSP6_inhibition) - k_erk_deg * pERK
    res_pDUSP6 = DUSP6_synthesis - k_dusp_deg * DUSP6

    k_akt_act = params["k_akt_act"]
    k_akt_deg = params["k_akt_deg"]

    res_pAKT = (
        k_akt_act * PI3K_total_input * (1.0 - PI3Ki_effect)
        - k_akt_deg * pAKT
        - mTOR_total_feedback * pAKT
    )

    k_s6k_act = params["k_s6k_act"]
    k_s6k_deg = params["k_s6k_deg"]

    mTOR_activity = pAKT * (1.0 - mTOR_4EBP1_competition)
    res_pS6K = k_s6k_act * mTOR_activity - k_s6k_deg * pS6K

    k_4ebp1_act = params["k_4ebp1_act"]
    k_4ebp1_deg = params["k_4ebp1_deg"]
    res_p4EBP1 = k_4ebp1_act * pAKT - k_4ebp1_deg * p4EBP1

    return np.array(
        [
            res_pEGFR,
            res_HER2,
            res_HER3,
            res_pDGFR,
            res_pCRAF,
            res_pMEK,
            res_pERK,
            res_pDUSP6,
            res_pAKT,
            res_pS6K,
            res_p4EBP1,
        ],
        dtype=float,
    )


def build_correction_fn(
    weights: Optional[Dict[str, np.ndarray]], use_correction: bool
) -> Optional[Callable[[np.ndarray], float]]:
    if not use_correction:
        return None
    if weights is None:
        return None

    w1 = weights["w1"]
    b1 = weights["b1"]
    w2 = weights["w2"]
    b2 = weights["b2"]

    def correction(x: np.ndarray) -> float:
        hidden = np.tanh(w1 @ x + b1)
        return float(w2 @ hidden + b2)

    return correction


def init_correction_weights(rng: np.random.Generator, width: int = 8) -> Dict[str, np.ndarray]:
    w1 = rng.normal(scale=0.05, size=(width, 2))
    b1 = rng.normal(scale=0.01, size=(width,))
    w2 = rng.normal(scale=0.05, size=(1, width))
    b2 = rng.normal(scale=0.01, size=(1,))
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
