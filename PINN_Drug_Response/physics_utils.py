import torch
import torch.nn as nn

def compute_physics_loss(model, t_physics, drugs, k_params, scalers):
    """
    Computes the ODE residuals for the PINN.
    
    Args:
        model: The PINN model
        t_physics: Normalized time points (batch, 1), requires_grad=True
        drugs: Drug concentrations (batch, 4)
        k_params: Dictionary of learnable or fixed rate constants
        scalers: Dictionary containing scaling factors (y_mean, y_std, t_range)
    """
    # Ensure gradients can be computed
    t_physics.requires_grad_(True)
    
    # Forward pass through model
    y_pred_norm = model(t_physics, drugs)
    
    # 1. Compute time derivatives in normalized space
    dy_dt_norm = torch.autograd.grad(
        outputs=y_pred_norm,
        inputs=t_physics,
        grad_outputs=torch.ones_like(y_pred_norm),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 2. Unnormalize outputs and derivatives to physical units for ODE comparison
    # y_unnorm = y_norm * std + mean
    # dy/dt_phys = (dy/dt_norm * std) / t_range
    
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    dy_dt = (dy_dt_norm * scalers['y_std']) / scalers['t_range']
    
    # Indices Mapping:
    # 0: pEGFR, 1: HER2, 2: HER3, 3: IGF1R, 4: pCRAF, 5: pMEK, 
    # 6: pERK, 7: DUSP6, 8: pAKT, 9: pS6K, 10: p4EBP1
    
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
    
    # Constants from k_params (or defaults)
    k = k_params
    Km = 0.5 # Michaelis-Menten constant
    IC50 = 0.5
    n = 2.0 # Hill coefficient
    
    # Drug Effects (Hill Equation)
    Tram_effect = (Tram**n) / (IC50**n + Tram**n + 1e-8)
    PI3Ki_effect = (PI3Ki**n) / (IC50**n + PI3Ki**n + 1e-8)
    
    # DUSP6 Feedback Effect
    DUSP6_effect = k['k_cat'] * DUSP6
    
    # Crosstalk
    # AKT inhibits RAF
    RAF_inh_by_AKT = k['k13'] * pAKT / (Km + pAKT + 1e-8)
    # ERK inhibits PI3K/IRS1 signaling (simplified as reduction in RTK signal)
    PI3K_inh_by_ERK = k['k14'] * pERK / (Km + pERK + 1e-8)
    
    # RTK Signal
    RTK_signal = (pEGFR + HER2 + IGF1R) * (1.0 - PI3K_inh_by_ERK)
    
    # ODE Residuals (MAPK)
    # d(pMEK)/dt = k1 * pCRAF * (1 - Tram_effect) * (1 - RAF_inh_by_AKT) - k2 * pMEK
    res_pMEK = dy_dt[:, 5] - (k['k1'] * pCRAF * (1.0 - Tram_effect) * (1.0 - RAF_inh_by_AKT) - k['k2'] * pMEK)
    
    # d(pERK)/dt = k3 * pMEK * (1 - DUSP6_effect) - k4 * pERK
    res_pERK = dy_dt[:, 6] - (k['k3'] * pMEK * (1.0 - DUSP6_effect) - k['k4'] * pERK)
    
    # d(DUSP6)/dt = k5 * pERK/(Km + pERK) - k6 * DUSP6
    res_pDUSP6 = dy_dt[:, 7] - (k['k5'] * pERK / (Km + pERK + 1e-8) - k['k6'] * DUSP6)
    
    # ODE Residuals (PI3K)
    # d(pAKT)/dt = k7 * RTK_signal * (1 - PI3Ki_effect) - k8 * pAKT
    res_pAKT = dy_dt[:, 8] - (k['k7'] * RTK_signal * (1.0 - PI3Ki_effect) - k['k8'] * pAKT)
    
    # d(pS6K)/dt = k9 * p4EBP1 * pAKT - k10 * pS6K
    res_pS6K = dy_dt[:, 9] - (k['k9'] * p4EBP1 * pAKT - k['k10'] * pS6K)
    
    # d(p4EBP1)/dt = k11 * pAKT - k12 * p4EBP1
    res_p4EBP1 = dy_dt[:, 10] - (k['k11'] * pAKT - k['k12'] * p4EBP1)
    
    # Combine residuals
    physics_loss = torch.mean(res_pMEK**2 + res_pERK**2 + res_pDUSP6**2 + 
                              res_pAKT**2 + res_pS6K**2 + res_p4EBP1**2)
    
    return physics_loss

def compute_conservation_loss(y_pred_norm, scalers):
    """
    Ensures biological constraints:
    1. Outputs must be non-negative (already handled by softplus, but good for safety)
    2. Protein conservation (optional, placeholder if needed)
    """
    y = y_pred_norm * scalers['y_std'] + scalers['y_mean']
    
    # Simple non-negativity penalty if values go below zero (though softplus is used)
    neg_penalty = torch.mean(torch.relu(-y))
    
    # Placeholder for more complex conservation if known
    return neg_penalty
