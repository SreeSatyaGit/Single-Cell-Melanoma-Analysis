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
    
    # Kinetic parameters with defaults
    k = k_params
    Km = k.get('Km', 0.5)  # Michaelis-Menten constant
    IC50 = k.get('IC50', 0.5)  # Half-maximal inhibitory concentration
    n = k.get('hill_coeff', 2.0)  # Hill coefficient for cooperativity
    
    # ==================================================================
    # DRUG EFFECTS - Hill equation with dose-response
    # ==================================================================
    
    # Tramametinib inhibits MEK (MEK inhibitor)
    Tram_effect = (Tram**n) / (IC50**n + Tram**n + 1e-8)
    
    # PI3K inhibitor
    PI3Ki_effect = (PI3Ki**n) / (IC50**n + PI3Ki**n + 1e-8)
    
    # Vemurafenib (BRAF inhibitor) - includes paradoxical activation
    # At low doses: can activate RAF (paradoxical activation)
    # At high doses: strong inhibition
    IC50_vem = k.get('IC50_vem', 0.8)
    Vem_inhibition = (Vem**n) / (IC50_vem**n + Vem**n + 1e-8)
    
    # Paradoxical activation: bell-shaped curve peaking at intermediate doses
    k_paradox = k.get('k_vem_paradox', 0.25)
    vem_opt = k.get('vem_optimal', 0.3)  # Optimal dose for paradox
    Vem_activation = k_paradox * Vem * torch.exp(-((Vem - vem_opt)**2) / (2 * 0.15**2))
    
    # ==================================================================
    # MAPK PATHWAY - NEGATIVE FEEDBACK LOOPS
    # ==================================================================
    
    # Feedback 1: ERK induces DUSP6 (phosphatase) which dephosphorylates ERK
    # This creates a delayed negative feedback loop
    k_dusp_synth = k.get('k_dusp_synth', 0.8)
    k_dusp_deg = k.get('k_dusp_deg', 0.5)
    Km_dusp = k.get('Km_dusp', 0.4)
    
    # DUSP6 synthesis: cooperative induction by ERK (Hill coefficient > 1)
    n_dusp = k.get('n_dusp', 2.5)  # Cooperativity for DUSP6 induction
    DUSP6_synthesis = k_dusp_synth * (pERK**n_dusp) / (Km_dusp**n_dusp + pERK**n_dusp + 1e-8)
    
    # DUSP6 phosphatase activity on ERK
    k_dusp_cat = k.get('k_dusp_cat', 0.6)
    DUSP6_inhibition = (k_dusp_cat * DUSP6) / (Km + DUSP6 + 1e-8)
    
    # Feedback 2: ERK inhibits SOS (RAS activator) - reduces upstream signaling
    # This feedback reduces RAF activation
    k_erk_sos = k.get('k_erk_sos', 0.4)
    ERK_to_SOS_inhibition = (k_erk_sos * pERK) / (Km + pERK + 1e-8)
    
    # Feedback 3: MEK substrate inhibition (product inhibition)
    k_mek_inhib = k.get('k_mek_inhib', 0.2)
    MEK_substrate_inhibition = k_mek_inhib * pMEK / (Km + pMEK + 1e-8)
    
    # ==================================================================
    # PI3K PATHWAY - NEGATIVE FEEDBACK LOOPS
    # ==================================================================
    
    # Feedback 1: S6K inhibits IRS1 - classic mTORC1 negative feedback
    # When S6K is high, it phosphorylates IRS1, reducing PI3K activation
    k_s6k_irs = k.get('k_s6k_irs', 0.7)
    Km_s6k = k.get('Km_s6k', 0.5)
    
    # S6K-mediated IRS1 inhibition (Michaelis-Menten saturation)
    S6K_to_IRS1_inhibition = (k_s6k_irs * pS6K) / (Km_s6k + pS6K + 1e-8)
    
    # Feedback 2: S6K feedback to mTOR itself
    k_s6k_mtor = k.get('k_s6k_mtor', 0.3)
    S6K_to_mTOR_feedback = (k_s6k_mtor * pS6K) / (Km + pS6K + 1e-8)
    
    # Feedback 3: 4EBP1 competes for mTOR activity
    k_4ebp1_comp = k.get('k_4ebp1_comp', 0.25)
    mTOR_4EBP1_competition = k_4ebp1_comp * p4EBP1 / (Km + p4EBP1 + 1e-8)
    
    # Combined mTOR feedback
    mTOR_total_feedback = S6K_to_IRS1_inhibition + S6K_to_mTOR_feedback
    
    # Feedback 4: AKT negative feedback (can inhibit upstream RTKs via mTOR)
    k_akt_rtk = k.get('k_akt_rtk', 0.15)
    AKT_to_RTK_feedback = k_akt_rtk * pAKT / (Km + pAKT + 1e-8)
    
    # ==================================================================
    # BIDIRECTIONAL CROSSTALK
    # ==================================================================
    
    # Crosstalk 1: AKT inhibits RAF (well-established negative crosstalk)
    # AKT phosphorylates RAF on inhibitory sites
    k_akt_raf = k.get('k_akt_raf', 0.5)
    AKT_to_RAF_inhibition = (k_akt_raf * pAKT) / (Km + pAKT + 1e-8)
    
    # Crosstalk 2: ERK inhibits PI3K/IRS1 (negative crosstalk)
    # ERK can phosphorylate IRS1 and reduce PI3K activation
    k_erk_pi3k = k.get('k_erk_pi3k', 0.45)
    ERK_to_PI3K_inhibition = (k_erk_pi3k * pERK) / (Km + pERK + 1e-8)
    
    # Crosstalk 3: RAF→PI3K compensatory activation
    # When MAPK is inhibited, RAF can activate PI3K pathway
    k_raf_pi3k = k.get('k_raf_pi3k', 0.2)
    RAF_to_PI3K_activation = (k_raf_pi3k * pCRAF) / (Km + pCRAF + 1e-8)
    
    # Crosstalk 4: AKT can relieve MEK inhibition (context-dependent)
    # Under certain conditions, AKT promotes MAPK signaling
    k_akt_mek = k.get('k_akt_mek', 0.18)
    AKT_to_MEK_promotion = (k_akt_mek * pAKT) / (k.get('Km_akt_mek', 1.2) + pAKT + 1e-8)
    
    # ==================================================================
    # RECEPTOR DYNAMICS (RTK Phosphorylation)
    # ==================================================================
    
    # Negative feedback from ERK to Receptors (from MATLAB ERK_EGFR_effect etc.)
    k_erk_rtk = k.get('k_erk_rtk', 0.1)
    ERK_feedback = (k_erk_rtk * pERK) / (k.get('Km_erk_rtk', 0.5) + pERK + 1e-8)
    
    # d(pEGFR)/dt
    # Activation depends on time (stimulus) and baseline EGFR
    # Inhibition from ERK feedback and dephosphorylation
    k_egfr_phos = k.get('k_egfr_phos', 0.5)
    k_egfr_dephos = k.get('k_egfr_dephos', 0.2)
    res_pEGFR = dy_dt[:, 0] - (
        k_egfr_phos * (1.0 - pEGFR) - (k_egfr_dephos + ERK_feedback) * pEGFR
    )
    
    # d(HER2)/dt and d(HER3)/dt
    k_her_phos = k.get('k_her_phos', 0.4)
    k_her_dephos = k.get('k_her_dephos', 0.15)
    res_HER2 = dy_dt[:, 1] - (k_her_phos * (1.0 - HER2) - (k_her_dephos + ERK_feedback) * HER2)
    res_HER3 = dy_dt[:, 2] - (k_her_phos * (1.0 - HER3) - (k_her_dephos + ERK_feedback) * HER3)
    
    # d(IGF1R)/dt
    k_igf_phos = k.get('k_igf_phos', 0.3)
    k_igf_dephos = k.get('k_igf_dephos', 0.2)
    res_IGF1R = dy_dt[:, 3] - (k_igf_phos * (1.0 - IGF1R) - (k_igf_dephos + ERK_feedback) * IGF1R)
    
    # ==================================================================
    # RTK SIGNALING WITH MULTIPLE FEEDBACKS
    # ==================================================================
    
    # Base RTK signal (weighted by receptor activity)
    w_her3 = k.get('w_her3', 1.5)  # HER3 is particularly important for PI3K
    RTK_base = pEGFR + HER2 + w_her3 * HER3 + IGF1R
    
    # SOS/RAS suppression by ERK (feeds MAPK arm)
    RAS_GTP = RTK_base * (1.0 - ERK_to_SOS_inhibition) * (1.0 - AKT_to_RTK_feedback)
    
    # RTK signal to PI3K pathway (includes ERK and mTOR feedbacks)
    PI3K_input = RTK_base * (1.0 - ERK_to_PI3K_inhibition) * (1.0 - mTOR_total_feedback)
    
    # Add compensatory RAF→PI3K activation
    PI3K_total_input = PI3K_input + RAF_to_PI3K_activation
    
    # ==================================================================
    # ORDINARY DIFFERENTIAL EQUATIONS (ODEs)
    # ==================================================================
    
    # ---------------------- MAPK Pathway ODEs ----------------------
    
    # d(pCRAF)/dt
    # Activation: RTK signaling, paradoxical Vem activation
    # Inhibition: Vemurafenib, AKT feedback, degradation
    k_craf_act = k.get('k_craf_act', 1.2)
    k_craf_deg = k.get('k_craf_deg', 0.35)
    
    res_pCRAF = dy_dt[:, 4] - (
        k_craf_act * RAS_GTP * (1.0 - Vem_inhibition)
        + Vem_activation  # Paradoxical activation
        - k_craf_deg * pCRAF
        - AKT_to_RAF_inhibition * pCRAF  # AKT negative crosstalk
    )
    
    # d(pMEK)/dt
    # Activation: pCRAF, AKT promotion (context-dependent)
    # Inhibition: Tramametinib, degradation, substrate inhibition
    k_mek_act = k.get('k_mek_act', 1.0)
    k_mek_deg = k.get('k_mek_deg', 0.4)
    
    res_pMEK = dy_dt[:, 5] - (
        k_mek_act * pCRAF * (1.0 - Tram_effect) * (1.0 - AKT_to_RAF_inhibition)
        + AKT_to_MEK_promotion  # AKT can promote MEK under some conditions
        - k_mek_deg * pMEK
        - MEK_substrate_inhibition * pMEK  # Product inhibition
    )
    
    # d(pERK)/dt
    # Activation: pMEK
    # Inhibition: DUSP6 (negative feedback), degradation
    k_erk_act = k.get('k_erk_act', 1.2)
    k_erk_deg = k.get('k_erk_deg', 0.45)
    
    res_pERK = dy_dt[:, 6] - (
        k_erk_act * pMEK * (1.0 - DUSP6_inhibition)  # DUSP6 negative feedback
        - k_erk_deg * pERK
    )
    
    # d(DUSP6)/dt
    # Synthesis: ERK-dependent (positive feedback creating negative loop)
    # Degradation: constitutive
    res_pDUSP6 = dy_dt[:, 7] - (
        DUSP6_synthesis  # Cooperative ERK-dependent synthesis
        - k_dusp_deg * DUSP6
    )
    
    # ---------------------- PI3K Pathway ODEs ----------------------
    
    # d(pAKT)/dt
    # Activation: RTK→PI3K, compensatory RAF activation
    # Inhibition: PI3Ki, degradation, mTOR feedbacks
    k_akt_act = k.get('k_akt_act', 1.0)
    k_akt_deg = k.get('k_akt_deg', 0.4)
    
    res_pAKT = dy_dt[:, 8] - (
        k_akt_act * PI3K_total_input * (1.0 - PI3Ki_effect)
        - k_akt_deg * pAKT
        - mTOR_total_feedback * pAKT  # S6K negative feedback
    )
    
    # d(pS6K)/dt
    # Activation: pAKT (via mTORC1)
    # Inhibition: degradation, 4EBP1 competition
    k_s6k_act = k.get('k_s6k_act', 0.9)
    k_s6k_deg = k.get('k_s6k_deg', 0.5)
    
    # mTOR activity proportional to AKT, modulated by 4EBP1 competition
    mTOR_activity = pAKT * (1.0 - mTOR_4EBP1_competition)
    
    res_pS6K = dy_dt[:, 9] - (
        k_s6k_act * mTOR_activity
        - k_s6k_deg * pS6K
    )
    
    # d(p4EBP1)/dt
    # Activation: pAKT (via mTORC1)
    # Inhibition: degradation
    k_4ebp1_act = k.get('k_4ebp1_act', 0.85)
    k_4ebp1_deg = k.get('k_4ebp1_deg', 0.45)
    
    res_p4EBP1 = dy_dt[:, 10] - (
        k_4ebp1_act * pAKT
        - k_4ebp1_deg * p4EBP1
    )
    
    # ==================================================================
    # WEIGHTED PHYSICS LOSS
    # ==================================================================
    
    # Weight residuals by biological importance and measurement reliability
    weights = {
        'pEGFR': k.get('w_egfr', 1.0),
        'HER2': k.get('w_her2', 1.0),
        'HER3': k.get('w_her3_w', 1.0),
        'IGF1R': k.get('w_igf1r', 1.0),
        'pCRAF': k.get('w_craf', 1.2),
        'pMEK': k.get('w_mek', 1.8),   # Key drug target
        'pERK': k.get('w_erk', 2.5),   # Master regulator, highly measurable
        'DUSP6': k.get('w_dusp6', 1.5), # Important feedback regulator
        'pAKT': k.get('w_akt', 2.5),   # Master regulator, key drug target
        'pS6K': k.get('w_s6k', 1.5),   # Important mTOR readout
        'p4EBP1': k.get('w_4ebp1', 1.3), # mTOR target
    }
    
    # ==================================================================
    # 1. RECEPTOR DYNAMICS (RTK Phosphorylation)
    # ==================================================================
    
    # Negative feedback from ERK to Receptors (from MATLAB ERK_EGFR_effect etc.)
    k_erk_rtk = k.get('k_erk_rtk', 0.1)
    ERK_feedback = (k_erk_rtk * pERK) / (k.get('Km_erk_rtk', 0.5) + pERK + 1e-8)
    
    # d(pEGFR)/dt
    # Activation depends on time (stimulus) and baseline EGFR
    # Inhibition from ERK feedback and dephosphorylation
    k_egfr_phos = k.get('k_egfr_phos', 0.5)
    k_egfr_dephos = k.get('k_egfr_dephos', 0.2)
    res_pEGFR = dy_dt[:, 0] - (
        k_egfr_phos * (1.0 - pEGFR) - (k_egfr_dephos + ERK_feedback) * pEGFR
    )
    
    # d(HER2)/dt and d(HER3)/dt
    k_her_phos = k.get('k_her_phos', 0.4)
    k_her_dephos = k.get('k_her_dephos', 0.15)
    res_HER2 = dy_dt[:, 1] - (k_her_phos * (1.0 - HER2) - (k_her_dephos + ERK_feedback) * HER2)
    res_HER3 = dy_dt[:, 2] - (k_her_phos * (1.0 - HER3) - (k_her_dephos + ERK_feedback) * HER3)
    
    # d(IGF1R)/dt
    k_igf_phos = k.get('k_igf_phos', 0.3)
    k_igf_dephos = k.get('k_igf_dephos', 0.2)
    res_IGF1R = dy_dt[:, 3] - (k_igf_phos * (1.0 - IGF1R) - (k_igf_dephos + ERK_feedback) * IGF1R)
    
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
    """
    Enhanced biological constraints with pathway balance.
    """
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    
    # Non-negativity constraint
    neg_penalty = torch.mean(torch.relu(-y))
    
    # ==================================================================
    # PATHWAY BALANCE CONSTRAINTS
    # ==================================================================
    
    pCRAF = y[:, 4]
    pMEK = y[:, 5]
    pERK = y[:, 6]
    DUSP6 = y[:, 7]
    pAKT = y[:, 8]
    pS6K = y[:, 9]
    p4EBP1 = y[:, 10]
    
    # MAPK cascade ordering: RAF → MEK → ERK
    # Soft constraint: signal should flow downward
    # Allow some violations due to feedback, but penalize large discrepancies
    mapk_flow_1 = torch.mean(torch.relu(pMEK - 2.5 * pCRAF)**2)
    mapk_flow_2 = torch.mean(torch.relu(pERK - 2.5 * pMEK)**2)
    
    # DUSP6 should correlate with ERK (they form feedback loop)
    dusp6_erk_correlation = torch.mean((DUSP6 - 0.8 * pERK)**2)
    
    # PI3K cascade: AKT → mTOR → S6K/4EBP1
    # Downstream should not exceed upstream by too much
    pi3k_flow_1 = torch.mean(torch.relu(pS6K - 2.0 * pAKT)**2)
    pi3k_flow_2 = torch.mean(torch.relu(p4EBP1 - 2.0 * pAKT)**2)
    
    # S6K and 4EBP1 should be somewhat correlated (both from mTOR)
    mtor_balance = torch.mean((pS6K - p4EBP1)**2)
    
    # ==================================================================
    # FEEDBACK LOOP CONSTRAINTS
    # ==================================================================
    
    # When ERK is high, DUSP6 should eventually be high (feedback)
    # Soft constraint to encourage this relationship
    erk_dusp_feedback = torch.mean(torch.relu(pERK - DUSP6 - 0.5)**2)
    
    # When S6K is high, it should eventually suppress upstream (PI3K/AKT)
    # This is captured in the ODEs, but we can add soft constraint
    
    # Combine all conservation losses
    conservation_loss = (
        neg_penalty +
        0.02 * mapk_flow_1 +
        0.02 * mapk_flow_2 +
        0.015 * dusp6_erk_correlation +
        0.02 * pi3k_flow_1 +
        0.02 * pi3k_flow_2 +
        0.01 * mtor_balance +
        0.015 * erk_dusp_feedback
    )
    
    return conservation_loss


def compute_crosstalk_loss(y_pred_norm, scalers, k_params):
    """
    Explicit loss term to enforce known crosstalk relationships and feedback dynamics.
    """
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    
    pCRAF = y[:, 4]
    pERK = y[:, 6]
    DUSP6 = y[:, 7]
    pAKT = y[:, 8]
    pS6K = y[:, 9]
    
    # ==================================================================
    # NEGATIVE FEEDBACK RELATIONSHIPS
    # ==================================================================
    
    # Feedback 1: ERK-DUSP6 negative feedback loop
    # High ERK should correlate with high DUSP6 (eventually)
    # But high DUSP6 should suppress ERK
    erk_dusp_feedback = torch.mean((pERK * DUSP6 - k_params.get('erk_dusp_product', 0.4))**2)
    
    # Feedback 2: S6K should suppress PI3K pathway
    # When S6K is very high, AKT should be relatively suppressed
    s6k_suppresses_akt = torch.mean(torch.relu(pAKT * pS6K - k_params.get('akt_s6k_product', 0.6))**2)
    
    # ==================================================================
    # CROSSTALK RELATIONSHIPS
    # ==================================================================
    
    # Crosstalk 1: ERK-AKT mutual inhibition
    # When ERK is high, AKT should be relatively low (and vice versa)
    # But not too extreme (some co-activation is possible)
    erk_akt_tradeoff = torch.mean((pERK * pAKT - k_params.get('erk_akt_baseline', 0.5))**2)
    
    # Crosstalk 2: AKT suppresses RAF
    # High AKT with high RAF is inconsistent
    akt_suppresses_raf = torch.mean(torch.relu(pCRAF * pAKT - k_params.get('raf_akt_product', 0.7))**2)
    
    # ==================================================================
    # COMPENSATORY ACTIVATION
    # ==================================================================
    
    # When MAPK pathway is low, PI3K pathway may be elevated (compensation)
    # When PI3K pathway is low, MAPK pathway may be elevated
    # This creates a homeostatic relationship
    pathway_compensation = torch.mean(
        torch.relu(k_params.get('min_pathway_sum', 0.8) - (pERK + pAKT))**2
    )
    
    # ==================================================================
    # DYNAMIC RANGE CONSTRAINTS
    # ==================================================================
    
    # Prevent pathological states where both pathways are maximally suppressed
    # or maximally activated simultaneously (unless under specific drug conditions)
    max_suppression = torch.mean(
        torch.relu(0.1 - pERK) * torch.relu(0.1 - pAKT)
    )
    
    # Total crosstalk loss
    crosstalk_loss = (
        0.15 * erk_dusp_feedback +
        0.12 * s6k_suppresses_akt +
        0.15 * erk_akt_tradeoff +
        0.12 * akt_suppresses_raf +
        0.1 * pathway_compensation +
        0.08 * max_suppression
    )
    
    return crosstalk_loss


def compute_feedback_strength_loss(y_pred_norm, scalers, k_params):
    """
    OPTIONAL: Additional loss to ensure feedback loops have appropriate strength.
    This helps prevent the network from learning weak or non-functional feedbacks.
    """
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    
    pERK = y[:, 6]
    DUSP6 = y[:, 7]
    pAKT = y[:, 8]
    pS6K = y[:, 9]
    
    # Ensure DUSP6 responds to ERK with sufficient sensitivity
    # If ERK changes, DUSP6 should change
    dusp6_sensitivity = torch.mean((DUSP6 - k_params.get('dusp6_erk_ratio', 0.9) * pERK)**2)
    
    # Ensure S6K is sensitive to AKT (via mTOR)
    s6k_sensitivity = torch.mean((pS6K - k_params.get('s6k_akt_ratio', 0.75) * pAKT)**2)
    
    feedback_strength_loss = 0.1 * (dusp6_sensitivity + s6k_sensitivity)
    
    return feedback_strength_loss


# ==================================================================
# USAGE EXAMPLE
# ==================================================================

def total_loss_function(model, t_data, y_data, drugs_data, t_physics, drugs_physics, 
                        k_params, scalers, lambda_data=1.0, lambda_physics=0.1, 
                        lambda_conservation=0.05, lambda_crosstalk=0.05, lambda_feedback=0.02):
    """
    Complete loss function combining all components.
    
    Args:
        model: PINN model
        t_data, y_data, drugs_data: Training data
        t_physics, drugs_physics: Physics collocation points
        k_params: Dictionary of kinetic parameters
        scalers: Normalization parameters
        lambda_*: Loss weighting hyperparameters
    
    Returns:
        total_loss: Weighted sum of all loss components
        loss_dict: Dictionary with individual loss values for monitoring
    """
    
    # Data fitting loss (MSE)
    y_pred_data = model(t_data, drugs_data)
    data_loss = torch.mean((y_pred_data - y_data)**2)
    
    # Physics-informed loss (ODE residuals)
    physics_loss = compute_physics_loss(model, t_physics, drugs_physics, k_params, scalers)
    
    # Get predictions for conservation and crosstalk losses
    y_pred_physics = model(t_physics, drugs_physics)
    
    # Conservation constraints
    conservation_loss = compute_conservation_loss(y_pred_physics, scalers)
    
    # Crosstalk relationships
    crosstalk_loss = compute_crosstalk_loss(y_pred_physics, scalers, k_params)
    
    # Feedback strength (optional)
    feedback_loss = compute_feedback_strength_loss(y_pred_physics, scalers, k_params)
    
    # Total weighted loss
    total_loss = (
        lambda_data * data_loss +
        lambda_physics * physics_loss +
        lambda_conservation * conservation_loss +
        lambda_crosstalk * crosstalk_loss +
        lambda_feedback * feedback_loss
    )
    
    # Return detailed losses for monitoring
    loss_dict = {
        'total': total_loss.item(),
        'data': data_loss.item(),
        'physics': physics_loss.item(),
        'conservation': conservation_loss.item(),
        'crosstalk': crosstalk_loss.item(),
        'feedback': feedback_loss.item()
    }
    
    return total_loss, loss_dict
