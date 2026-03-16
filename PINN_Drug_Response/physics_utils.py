import torch
import torch.nn as nn
def compute_physics_loss(model, t_physics, drugs, k_params, scalers):
    """
    Computes the Physics-Informed loss based on a system of 10 ODEs for
    MAPK and PI3K signaling in A375 (BRAF V600E) melanoma cells.
    Uses Michaelis-Menten saturation K_sat/(K_sat + X) instead of logistic (1 - X)
    to correctly handle raw western blot intensities (A.U.) that can exceed 1.0.
    Equations modelled:
    1-4. Receptor Tyr Kinases: d(pEGFR)/dt, d(HER2)/dt, d(HER3)/dt, d(IGF1R)/dt
    5-8. MAPK Signaling: d(pCRAF)/dt, d(pMEK)/dt, d(pERK)/dt, d(DUSP6)/dt
    9. PI3K/AKT Signaling: d(pAKT)/dt
    10. Protein Synthesis Level: d(p4EBP1)/dt
    Feedback Loops:
    - ERK -> RTK inhibition (negative)
    - ERK -> SOS inhibition (negative, via RAS_GTP intermediate)
    - AKT -> RTK inhibition (negative)
    - ERK -> PI3K inhibition (negative)
    - RAF -> PI3K activation (positive)
    - ERK -> DUSP6 induction (positive, transcriptional)
    - DUSP6 -> ERK dephosphorylation (negative)
    - AKT -> RAF inhibition (negative crosstalk)
    """
    t_physics.requires_grad_(True)
    y_pred_norm = model(t_physics, drugs)
    dy_dt_norm = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=t_physics,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True
    )[0]
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    dy_dt = (dy_dt_norm * scalers['y_std']) / scalers['t_range']
    pEGFR  = y[:, 0]
    HER2   = y[:, 1]
    HER3   = y[:, 2]
    IGF1R  = y[:, 3]
    pCRAF  = y[:, 4]
    pMEK   = y[:, 5]
    pERK   = y[:, 6]
    DUSP6  = y[:, 7]
    pAKT   = y[:, 8]
    p4EBP1 = y[:, 9]
    Vem    = drugs[:, 0]
    Tram   = drugs[:, 1]
    PI3Ki  = drugs[:, 2]
    RasInh = drugs[:, 3]
    n = torch.clamp(k_params.get('hill_coeff', 2.0), 1.0, 4.0)
    eps = 1e-7
    Vem_inhibition = ((Vem+eps)**n) / (torch.abs(k_params.get('IC50_vem', 0.8))**n + (Vem+eps)**n + 1e-8)
    Tram_effect    = ((Tram+eps)**n) / (torch.abs(k_params.get('IC50_tram', 0.3))**n + (Tram+eps)**n + 1e-8)
    PI3Ki_effect   = ((PI3Ki+eps)**n) / (torch.abs(k_params.get('IC50_pi3k', 0.5))**n + (PI3Ki+eps)**n + 1e-8)
    Ras_effect     = ((RasInh+eps)**n) / (torch.abs(k_params.get('IC50_ras', 0.5))**n + (RasInh+eps)**n + 1e-8)
    K_sat_egfr  = torch.abs(k_params.get('K_sat_egfr', 1.0))
    K_sat_her2  = torch.abs(k_params.get('K_sat_her2', 2.0))
    K_sat_her3  = torch.abs(k_params.get('K_sat_her3', 2.0))
    K_sat_igfr  = torch.abs(k_params.get('K_sat_igfr', 1.5))
    K_sat_craf  = torch.abs(k_params.get('K_sat_craf', 3.0))
    K_sat_mek   = torch.abs(k_params.get('K_sat_mek', 2.5))
    K_sat_erk   = torch.abs(k_params.get('K_sat_erk', 3.5))
    K_sat_akt   = torch.abs(k_params.get('K_sat_akt', 1.0))
    K_sat_4ebp1 = torch.abs(k_params.get('K_sat_4ebp1', 1.0))
    k_paradox = torch.abs(k_params.get('k_paradox', 0.25))
    Vem_paradox = k_paradox * Vem * K_sat_craf / (K_sat_craf + pCRAF + 1e-8)

    k_erk_rtk = torch.abs(k_params.get('k_erk_rtk', 0.1))
    Km_rtk = torch.abs(k_params.get('Km_rtk', 0.5))
    ERK_feedback = k_erk_rtk * pERK / (Km_rtk + pERK + 1e-8)
    k_up = torch.abs(k_params.get('k_up', 0.3))
    drug_relief = k_up * (Vem_inhibition + Tram_effect + PI3Ki_effect)
    k_erk_sos = torch.abs(k_params.get('k_erk_sos', 0.4))
    Km_sos = torch.abs(k_params.get('Km_sos', 0.5))
    k_akt_rtk = torch.abs(k_params.get('k_akt_rtk', 0.15))
    Km_artk = torch.abs(k_params.get('Km_artk', 0.5))
    ERK_to_SOS = k_erk_sos * pERK / (Km_sos + pERK + 1e-8)
    AKT_to_RTK = k_akt_rtk * pAKT / (Km_artk + pAKT + 1e-8)
    RTK_total  = pEGFR + HER2 + 1.5 * HER3 + IGF1R
    RAS_GTP    = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_effect)
    k_raf_pi3k = torch.abs(k_params.get('k_raf_pi3k', 0.2))
    Km_raf_pi3k = torch.abs(k_params.get('Km_raf_pi3k', 0.5))
    k_erk_pi3k = torch.abs(k_params.get('k_erk_pi3k', 0.45))
    Km_erk_pi3k = torch.abs(k_params.get('Km_erk_pi3k', 0.5))
    RAF_to_PI3K = k_raf_pi3k * pCRAF / (Km_raf_pi3k + pCRAF + 1e-8)
    ERK_to_PI3K = k_erk_pi3k * pERK / (Km_erk_pi3k + pERK + 1e-8)
    PI3K_input  = RTK_total * (1.0 - ERK_to_PI3K) + RAF_to_PI3K
    k_akt_raf = torch.abs(k_params.get('k_akt_raf', 0.5))
    Km_akt_raf = torch.abs(k_params.get('Km_akt_raf', 0.5))
    AKT_RAF_inhib  = k_akt_raf * pAKT / (Km_akt_raf + pAKT + 1e-8)
    k_egfr = torch.abs(k_params.get('k_egfr', 0.5))
    k_egfr_deg = torch.abs(k_params.get('k_egfr_deg', 0.2))
    res_pEGFR = dy_dt[:, 0] - (k_egfr * (1.0 + drug_relief) * K_sat_egfr / (K_sat_egfr + pEGFR + 1e-8) - (k_egfr_deg + ERK_feedback) * pEGFR)
    k_her2 = torch.abs(k_params.get('k_her2', 0.4))
    k_her2_deg = torch.abs(k_params.get('k_her2_deg', 0.15))
    res_HER2  = dy_dt[:, 1] - (k_her2 * (1.0 + drug_relief) * K_sat_her2 / (K_sat_her2 + HER2 + 1e-8) - (k_her2_deg + ERK_feedback) * HER2)
    k_her3 = torch.abs(k_params.get('k_her3', 0.4))
    k_her3_deg = torch.abs(k_params.get('k_her3_deg', 0.15))
    res_HER3  = dy_dt[:, 2] - (k_her3 * (1.0 + 2.0 * drug_relief) * K_sat_her3 / (K_sat_her3 + HER3 + 1e-8) - (k_her3_deg + ERK_feedback) * HER3)
    k_igf = torch.abs(k_params.get('k_igf', 0.3))
    k_igf_deg = torch.abs(k_params.get('k_igf_deg', 0.2))
    res_IGF1R = dy_dt[:, 3] - (k_igf * (1.0 + drug_relief) * K_sat_igfr / (K_sat_igfr + IGF1R + 1e-8) - (k_igf_deg + ERK_feedback) * IGF1R)
    k_craf = torch.abs(k_params.get('k_craf', 1.2))
    k_craf_deg = torch.abs(k_params.get('k_craf_deg', 0.35))
    res_pCRAF = dy_dt[:, 4] - (k_craf * RAS_GTP * (1.0 - Vem_inhibition) * K_sat_craf / (K_sat_craf + pCRAF + 1e-8) + Vem_paradox - (k_craf_deg + AKT_RAF_inhib) * pCRAF)
    k_mek = torch.abs(k_params.get('k_mek', 1.0))
    k_mek_deg = torch.abs(k_params.get('k_mek_deg', 0.4))
    res_pMEK  = dy_dt[:, 5] - (k_mek * pCRAF * (1.0 - Tram_effect) * K_sat_mek / (K_sat_mek + pMEK + 1e-8) - k_mek_deg * pMEK)
    k_dusp_cat = torch.abs(k_params.get('k_dusp_cat', 0.6))
    Km_dusp = torch.abs(k_params.get('Km_dusp', 0.4))
    DUSP6_activity = k_dusp_cat * DUSP6 / (Km_dusp + DUSP6 + 1e-8)
    k_erk = torch.abs(k_params.get('k_erk', 1.2))
    k_erk_deg = torch.abs(k_params.get('k_erk_deg', 0.45))
    res_pERK  = dy_dt[:, 6] - (k_erk * pMEK * K_sat_erk / (K_sat_erk + pERK + 1e-8) - (k_erk_deg + DUSP6_activity) * pERK)
    n_dusp = torch.clamp(k_params.get('n_dusp', 2.0), 1.5, 3.5)
    k_dusp_synth = torch.abs(k_params.get('k_dusp_synth', 0.8))
    Km_dusp_s = torch.abs(k_params.get('Km_dusp_s', 0.4))
    DUSP6_induction = k_dusp_synth * ((pERK+eps)**n_dusp) / (Km_dusp_s**n_dusp + (pERK+eps)**n_dusp + 1e-8)
    k_dusp_deg = torch.abs(k_params.get('k_dusp_deg', 0.5))
    res_DUSP6 = dy_dt[:, 7] - (DUSP6_induction - k_dusp_deg * DUSP6)
    k_akt = torch.abs(k_params.get('k_akt', 1.0))
    k_akt_deg = torch.abs(k_params.get('k_akt_deg', 0.4))
    res_pAKT  = dy_dt[:, 8] - (k_akt * PI3K_input * (1.0 - PI3Ki_effect) * K_sat_akt / (K_sat_akt + pAKT + 1e-8) - k_akt_deg * pAKT)
    k_4ebp1 = torch.abs(k_params.get('k_4ebp1', 0.85))
    k_4ebp1_deg = torch.abs(k_params.get('k_4ebp1_deg', 0.45))
    res_p4EBP1 = dy_dt[:, 9] - (k_4ebp1 * pAKT * K_sat_4ebp1 / (K_sat_4ebp1 + p4EBP1 + 1e-8) - k_4ebp1_deg * p4EBP1)
    physics_loss = (
        torch.abs(k_params.get('w_egfr', 1.0)) * torch.mean(res_pEGFR**2) +
        torch.abs(k_params.get('w_her2', 1.0)) * torch.mean(res_HER2**2) +
        torch.abs(k_params.get('w_her3', 1.2)) * torch.mean(res_HER3**2) +
        torch.abs(k_params.get('w_igf1r', 1.0)) * torch.mean(res_IGF1R**2) +
        torch.abs(k_params.get('w_craf', 1.2)) * torch.mean(res_pCRAF**2) +
        torch.abs(k_params.get('w_mek', 1.8)) * torch.mean(res_pMEK**2) +
        torch.abs(k_params.get('w_erk', 2.5)) * torch.mean(res_pERK**2) +
        torch.abs(k_params.get('w_dusp6', 1.5)) * torch.mean(res_DUSP6**2) +
        torch.abs(k_params.get('w_akt', 2.5)) * torch.mean(res_pAKT**2) +
        torch.abs(k_params.get('w_4ebp1', 1.3)) * torch.mean(res_p4EBP1**2)
    )
    return physics_loss
def compute_conservation_loss(y_pred_norm, scalers):
    """
    Computes conservation and biological constraint losses for pathway consistency.
    """
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    pCRAF  = y[:, 4]
    pMEK   = y[:, 5]
    pERK   = y[:, 6]
    DUSP6  = y[:, 7]
    pAKT   = y[:, 8]
    p4EBP1 = y[:, 9]
    neg_penalty = torch.mean(torch.relu(-y))
    mapk_order = torch.mean(torch.relu(pMEK - 2.5 * pCRAF)**2) +                 torch.mean(torch.relu(pERK - 2.5 * pMEK)**2)
    dusp6_corr = torch.mean((DUSP6 - 0.8 * pERK)**2)
    conservation_loss = neg_penalty +                        0.02 * mapk_order +                        0.015 * dusp6_corr
    return conservation_loss
def compute_steady_state_loss(model, t_physics, drugs, scalers):
    """
    Computes the steady-state loss: penalizes dX/dt ≠ 0 at collocation
    points where ALL drugs = 0.
    Biological rationale:
        In BRAF V600E melanoma without drug treatment, the constitutive
        MAPK/PI3K pathway is at steady state. All species should have
        dX/dt ≈ 0 because there is no perturbation driving the system
        away from equilibrium.
    This loss is ONLY applied to no-drug collocation points (drugs = [0,0,0,0]).
    Args:
        model: The PINN model.
        t_physics: Collocation time points (normalized), shape (N, 1).
        drugs: Drug concentrations at collocation points, shape (N, 4).
        scalers: Dict with 'y_std', 'y_mean', 't_range' for un-normalization.
    Returns:
        Scalar loss tensor (mean squared temporal derivative at no-drug points).
    """
    drug_sum = drugs.sum(dim=1)
    no_drug_mask = (drug_sum < 1e-6)
    n_no_drug = no_drug_mask.sum().item()
    if n_no_drug == 0:
        return torch.tensor(0.0, device=t_physics.device, requires_grad=True)
    t_nd = t_physics[no_drug_mask].clone().detach().requires_grad_(True)
    drugs_nd = drugs[no_drug_mask]
    y_pred_norm = model(t_nd, drugs_nd)
    dy_dt_norm = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=t_nd,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True
    )[0]
    dy_dt = (dy_dt_norm * scalers['y_std']) / scalers['t_range']
    y_scale = scalers['y_std'] + 1e-8
    dy_dt_scaled = dy_dt / y_scale.unsqueeze(0)
    steady_state_loss = torch.mean(dy_dt_scaled ** 2)
    return steady_state_loss
