import torch
import torch.nn as nn

def compute_physics_loss(model, t_physics, drugs, k_params, scalers):
    """
    Enhanced PINN with comprehensive negative feedback loops in MAPK and PI3K pathways.
    
    Key feedback mechanisms:
    ========================
    MAPK Pathway Negative Feedbacks:
    1. ERK → DUSP6 → ERK (phosphatase-mediated negative feedback)
    2. ERK → SOS → RAS/RAF (inhibition of guanine exchange factor)
    3. ERK → RSK → SOS (dual negative feedback)
    4. pMEK → MEK (substrate inhibition)
    
    PI3K Pathway Negative Feedbacks:
    1. S6K → IRS1 → PI3K (classic mTORC1 feedback)
    2. S6K → mTOR → PI3K (direct feedback)
    3. AKT → PDK1 (feedback regulation)
    4. 4EBP1 → mTOR (substrate competition)
    
    Crosstalk (Bidirectional):
    1. ERK ⊣ IRS1/PI3K (negative regulation)
    2. AKT ⊣ RAF (negative regulation)
    3. RAF → PI3K (compensatory activation under MAPK inhibition)
    4. mTOR ⊣ RTK signaling (long-term feedback)
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
    
    # Extract all species
    pEGFR = y[:, 0]
    HER2 = y[:, 1]
    HER3 = y[:, 2]
    IGF1R = y[:, 3]
    pCRAF = y[:, 4]
    pMEK = y[:, 5]
    pERK = y[:, 6]
    DUSP6 = y[:, 7]
    pAKT = y[:, 8]
    pS6K = y[:, 9]
    p4EBP1 = y[:, 10]
    
    # Drug Concentrations
    Vem = drugs[:, 0]
    Tram = drugs[:, 1]
    PI3Ki = drugs[:, 2]
    RasInh = drugs[:, 3]
    
    # Kinetic parameters with defaults
    k = k_params
    Km = k.get('Km', 0.5)
    IC50 = k.get('IC50', 0.5)
    n = k.get('hill_coeff', 2.0)
    
    # Drug Effects

    
    # Tramametinib inhibits MEK
    Tram_effect = (Tram**n) / (IC50**n + Tram**n + 1e-8)
    
    # PI3K inhibitor
    PI3Ki_effect = (PI3Ki**n) / (IC50**n + PI3Ki**n + 1e-8)
    
    # RAS Inhbiitor
    Ras_effect = (RasInh**n) / (IC50**n + RasInh**n + 1e-8)
    
    # Vemurafenib (BRAF inhibitor) - includes paradoxical activation
    IC50_vem = k.get('IC50_vem', 0.8)
    Vem_inhibition = (Vem**n) / (IC50_vem**n + Vem**n + 1e-8)
    
    k_paradox = k.get('k_vem_paradox', 0.25)
    vem_opt = k.get('vem_optimal', 0.3)
    Vem_activation = k_paradox * Vem * torch.exp(-((Vem - vem_opt)**2) / (2 * 0.15**2))
    
    # ==================================================================
    # MAPK PATHWAY - NEGATIVE FEEDBACK LOOPS
    # ==================================================================
    
    k_dusp_synth = k.get('k_dusp_synth', 0.8)
    k_dusp_deg = k.get('k_dusp_deg', 0.5)
    Km_dusp = k.get('Km_dusp', 0.4)
    n_dusp = k.get('n_dusp', 2.5)  
    DUSP6_synthesis = k_dusp_synth * (pERK**n_dusp) / (Km_dusp**n_dusp + pERK**n_dusp + 1e-8)
    
    k_dusp_cat = k.get('k_dusp_cat', 0.6)
    DUSP6_inhibition = (k_dusp_cat * DUSP6) / (Km + DUSP6 + 1e-8)
    
    k_erk_sos = k.get('k_erk_sos', 0.4)
    ERK_to_SOS_inhibition = (k_erk_sos * pERK) / (Km + pERK + 1e-8)
    
    k_mek_inhib = k.get('k_mek_inhib', 0.2)
    MEK_substrate_inhibition = k_mek_inhib * pMEK / (Km + pMEK + 1e-8)
    
    # PI3K Pathway Feedbacks

    
    k_s6k_irs = k.get('k_s6k_irs', 0.7)
    Km_s6k = k.get('Km_s6k', 0.5)
    S6K_to_IRS1_inhibition = (k_s6k_irs * pS6K) / (Km_s6k + pS6K + 1e-8)
    
    k_s6k_mtor = k.get('k_s6k_mtor', 0.3)
    S6K_to_mTOR_feedback = (k_s6k_mtor * pS6K) / (Km + pS6K + 1e-8)
    
    k_4ebp1_comp = k.get('k_4ebp1_comp', 0.25)
    mTOR_4EBP1_competition = k_4ebp1_comp * p4EBP1 / (Km + p4EBP1 + 1e-8)
    
    mTOR_total_feedback = S6K_to_IRS1_inhibition + S6K_to_mTOR_feedback
    
    k_akt_rtk = k.get('k_akt_rtk', 0.15)
    AKT_to_RTK_feedback = k_akt_rtk * pAKT / (Km + pAKT + 1e-8)
    
    # Bidirectional Crosstalk

    
    k_akt_raf = k.get('k_akt_raf', 0.5)
    AKT_to_RAF_inhibition = (k_akt_raf * pAKT) / (Km + pAKT + 1e-8)
    
    k_erk_pi3k = k.get('k_erk_pi3k', 0.45)
    ERK_to_PI3K_inhibition = (k_erk_pi3k * pERK) / (Km + pERK + 1e-8)
    
    k_raf_pi3k = k.get('k_raf_pi3k', 0.2)
    RAF_to_PI3K_activation = (k_raf_pi3k * pCRAF) / (Km + pCRAF + 1e-8)
    
    k_akt_mek = k.get('k_akt_mek', 0.18)
    AKT_to_MEK_promotion = (k_akt_mek * pAKT) / (k.get('Km_akt_mek', 1.2) + pAKT + 1e-8)
    
    # RTK signaling and crosstalk inputs
    w_her3 = k.get('w_her3', 1.5)
    RTK_base = pEGFR + HER2 + w_her3 * HER3 + IGF1R
    
    RAS_GTP = RTK_base * (1.0 - ERK_to_SOS_inhibition) * (1.0 - AKT_to_RTK_feedback) * (1.0 - Ras_effect)
    w_ras_pi3k = k.get('w_ras_pi3k', 0.5)
    PI3K_input = (RTK_base + w_ras_pi3k * RAS_GTP) * (1.0 - ERK_to_PI3K_inhibition) * (1.0 - mTOR_total_feedback)
    PI3K_total_input = PI3K_input + RAF_to_PI3K_activation
    
    # --- MAPK Pathway ODEs ---
    k_craf_act, k_craf_deg = k.get('k_craf_act', 1.2), k.get('k_craf_deg', 0.35)
    res_pCRAF = dy_dt[:, 4] - ((k_craf_act * RAS_GTP * (1.0 - Vem_inhibition) + Vem_activation) * (1.0 - pCRAF) - (k_craf_deg + AKT_to_RAF_inhibition) * pCRAF)
    
    k_mek_act, k_mek_deg = k.get('k_mek_act', 1.0), k.get('k_mek_deg', 0.4)
    res_pMEK = dy_dt[:, 5] - ((k_mek_act * pCRAF * (1.0 - Tram_effect) * (1.0 - AKT_to_RAF_inhibition) + AKT_to_MEK_promotion) * (1.0 - pMEK) - (k_mek_deg + MEK_substrate_inhibition) * pMEK)
    
    k_erk_act, k_erk_deg = k.get('k_erk_act', 1.2), k.get('k_erk_deg', 0.45)
    res_pERK = dy_dt[:, 6] - (k_erk_act * pMEK * (1.0 - pERK) - (k_erk_deg + DUSP6_inhibition) * pERK)
    
    res_pDUSP6 = dy_dt[:, 7] - (DUSP6_synthesis - k_dusp_deg * DUSP6)
    
    # --- PI3K Pathway ODEs ---
    k_akt_act, k_akt_deg = k.get('k_akt_act', 1.0), k.get('k_akt_deg', 0.4)
    res_pAKT = dy_dt[:, 8] - (k_akt_act * PI3K_total_input * (1.0 - PI3Ki_effect) * (1.0 - pAKT) - (k_akt_deg + mTOR_total_feedback) * pAKT)
    
    k_s6k_act, k_s6k_deg = k.get('k_s6k_act', 0.9), k.get('k_s6k_deg', 0.5)
    res_pS6K = dy_dt[:, 9] - (k_s6k_act * (pAKT * (1.0 - mTOR_4EBP1_competition)) * (1.0 - pS6K) - k_s6k_deg * pS6K)
    
    k_4ebp1_act, k_4ebp1_deg = k.get('k_4ebp1_act', 0.85), k.get('k_4ebp1_deg', 0.45)
    res_p4EBP1 = dy_dt[:, 10] - (k_4ebp1_act * pAKT * (1.0 - p4EBP1) - k_4ebp1_deg * p4EBP1)

    # Receptor Dynamics (calculated here to avoid repetition)
    k_erk_rtk = k.get('k_erk_rtk', 0.1)
    ERK_feedback = (k_erk_rtk * pERK) / (k.get('Km_erk_rtk', 0.5) + pERK + 1e-8)
    
    k_egfr_phos, k_egfr_dephos = k.get('k_egfr_phos', 0.5), k.get('k_egfr_dephos', 0.2)
    res_pEGFR = dy_dt[:, 0] - (k_egfr_phos * (1.0 - pEGFR) - (k_egfr_dephos + ERK_feedback) * pEGFR)
    
    k_her_phos, k_her_dephos = k.get('k_her_phos', 0.4), k.get('k_her_dephos', 0.15)
    res_HER2 = dy_dt[:, 1] - (k_her_phos * (1.0 - HER2) - (k_her_dephos + ERK_feedback) * HER2)
    res_HER3 = dy_dt[:, 2] - (k_her_phos * (1.0 - HER3) - (k_her_dephos + ERK_feedback) * HER3)
    
    k_igf_phos, k_igf_dephos = k.get('k_igf_phos', 0.3), k.get('k_igf_dephos', 0.2)
    res_IGF1R = dy_dt[:, 3] - (k_igf_phos * (1.0 - IGF1R) - (k_igf_dephos + ERK_feedback) * IGF1R)
    
    # Weighted Physics Loss
    weights = {
        'pEGFR': k.get('w_egfr', 1.0), 'HER2': k.get('w_her2', 1.0), 'HER3': k.get('w_her3_w', 1.0),
        'IGF1R': k.get('w_igf1r', 1.0), 'pCRAF': k.get('w_craf', 1.2), 'pMEK': k.get('w_mek', 1.8),
        'pERK': k.get('w_erk', 2.5), 'DUSP6': k.get('w_dusp6', 1.5), 'pAKT': k.get('w_akt', 2.5),
        'pS6K': k.get('w_s6k', 1.5), 'p4EBP1': k.get('w_4ebp1', 1.3),
    }
    
    physics_loss = (
        weights['pEGFR'] * torch.mean(res_pEGFR**2) +
        weights['HER2'] * torch.mean(res_HER2**2) +
        weights['HER3'] * torch.mean(res_HER3**2) +
        weights['IGF1R'] * torch.mean(res_IGF1R**2) +
        weights['pCRAF'] * torch.mean(res_pCRAF**2) +
        weights['pMEK'] * torch.mean(res_pMEK**2) +
        weights['pERK'] * torch.mean(res_pERK**2) +
        weights['DUSP6'] * torch.mean(res_pDUSP6**2) +
        weights['pAKT'] * torch.mean(res_pAKT**2) +
        weights['pS6K'] * torch.mean(res_pS6K**2) +
        weights['p4EBP1'] * torch.mean(res_p4EBP1**2)
    )
    
    return physics_loss

def compute_conservation_loss(y_pred_norm, scalers):
    """Biological constraints for pathway consistency."""
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    neg_penalty = torch.mean(torch.relu(-y))
    
    pCRAF, pMEK, pERK = y[:, 4], y[:, 5], y[:, 6]
    DUSP6, pAKT, pS6K, p4EBP1 = y[:, 7], y[:, 8], y[:, 9], y[:, 10]
    
    mapk_flow = torch.mean(torch.relu(pMEK - 2.5 * pCRAF)**2) + torch.mean(torch.relu(pERK - 2.5 * pMEK)**2)
    dusp6_corr = torch.mean((DUSP6 - 0.8 * pERK)**2)
    pi3k_flow = torch.mean(torch.relu(pS6K - 2.0 * pAKT)**2) + torch.mean(torch.relu(p4EBP1 - 2.0 * pAKT)**2)
    mtor_balance = torch.mean((pS6K - p4EBP1)**2)
    erk_dusp_fdbk = torch.mean(torch.relu(pERK - DUSP6 - 0.5)**2)
    
    return neg_penalty + 0.02 * mapk_flow + 0.015 * dusp6_corr + 0.02 * pi3k_flow + 0.01 * mtor_balance + 0.015 * erk_dusp_fdbk

