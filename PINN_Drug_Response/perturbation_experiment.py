"""
Perturbation Experiment: Multi-Phase PINN → ODE Hybrid Simulation
────────────────────────────────────────────────────────────────
This script implements a robust simulation framework to evaluate therapeutic
interventions (e.g., drug additions, dose changes) in cancer signaling pathways.
Methodology:
 1. Phase 1 (0 to t_switch): Employs the trained Physics-Informed Neural Network
    to estimate the cellular state under a known, baseline regimen (e.g., Vemurafenib
    + Trametinib). This leverages the PINN's high interpolate accuracy.
 2. Phase 2 (t_switch to t_end): Transitions to mechanistic ODE integration
    using kinetically-derived parameters. This allows for reliable extrapolation
    into novel drug combinations and dosage regimes (e.g., adding PI3Ki) where
    pure neural networks often fail due to extrapolation non-linearity.
Usage:
    python3 perturbation_experiment.py --t_switch 80 --pi3k_dose 0.5 --t_end 200
    python3 perturbation_experiment.py --t_switch 60 --pi3k_dose 0.001 --t_end 200
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from inference import load_pinn
from data_utils import SPECIES_ORDER
MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'pinn_model_best.pth'
OUTPUT_DIR = 'analysis_results/perturbation_experiments'
BASE_DRUGS = {
    'vemurafenib':    0.5,
    'trametinib':     0.3,
    'pi3k_inhibitor': 0.0,
    'ras_inhibitor':  0.0,
}
def load_trained_k_params(filepath, device='cpu'):
    """
    Load the optimized kinetic parameters (k_params) that were co-trained
    with the PINN. Returns a plain Python dict {param_name: float_value}.
    """
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    if 'k_params_state_dict' not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'k_params_state_dict'. "
            "Make sure you are using a model trained with train_pinn.py."
        )
    raw = checkpoint['k_params_state_dict']
    k = {name: abs(tensor.cpu().item()) for name, tensor in raw.items()}
    k['hill_coeff'] = float(np.clip(k.get('hill_coeff', 2.0), 1.0, 4.0))
    k['n_dusp']     = float(np.clip(k.get('n_dusp', 2.0), 1.5, 3.5))
    return k
def ode_rhs(t, X, k, drugs):
    """
    Right-hand side of the 10-species ODE system for MAPK/PI3K signaling
    in A375 (BRAF V600E) melanoma cells.
    Pure-NumPy translation of physics_utils.py with ALL feedback loops
    and crosstalk terms, including drug_relief on RTK equations and
    RTK_total in PI3K_input.
    """
    eps = 1e-7
    pEGFR, HER2, HER3, IGF1R, pCRAF, pMEK, pERK, DUSP6, pAKT, p4EBP1 = X
    pEGFR  = max(pEGFR, 0.0)
    HER2   = max(HER2,  0.0)
    HER3   = max(HER3,  0.0)
    IGF1R  = max(IGF1R, 0.0)
    pCRAF  = max(pCRAF, 0.0)
    pMEK   = max(pMEK,  0.0)
    pERK   = max(pERK,  0.0)
    DUSP6  = max(DUSP6, 0.0)
    pAKT   = max(pAKT,  0.0)
    p4EBP1 = max(p4EBP1, 0.0)
    Vem    = drugs['vemurafenib']
    Tram   = drugs['trametinib']
    PI3Ki  = drugs['pi3k_inhibitor']
    RasInh = drugs['ras_inhibitor']
    n = k['hill_coeff']
    def hill(conc, ic50):
        return ((conc + eps)**n) / (abs(ic50)**n + (conc + eps)**n + 1e-8)
    Vem_inhibition = hill(Vem,    k['IC50_vem'])
    Tram_effect    = hill(Tram,   k['IC50_tram'])
    PI3Ki_effect   = hill(PI3Ki,  k['IC50_pi3k'])
    Ras_effect     = hill(RasInh, k['IC50_ras'])
    K_sat_egfr  = k['K_sat_egfr']
    K_sat_her2  = k['K_sat_her2']
    K_sat_her3  = k['K_sat_her3']
    K_sat_igfr  = k['K_sat_igfr']
    K_sat_craf  = k['K_sat_craf']
    K_sat_mek   = k['K_sat_mek']
    K_sat_erk   = k['K_sat_erk']
    K_sat_akt   = k['K_sat_akt']
    K_sat_4ebp1 = k['K_sat_4ebp1']
    Vem_paradox = k['k_paradox'] * Vem * K_sat_craf / (K_sat_craf + pCRAF + 1e-8)
    ERK_feedback = k['k_erk_rtk'] * pERK / (k['Km_rtk'] + pERK + 1e-8)
    drug_relief = k['k_up'] * (Vem_inhibition + Tram_effect + PI3Ki_effect)
    ERK_to_SOS = k['k_erk_sos'] * pERK / (k['Km_sos'] + pERK + 1e-8)
    AKT_to_RTK = k['k_akt_rtk'] * pAKT / (k['Km_artk'] + pAKT + 1e-8)
    RTK_total  = pEGFR + HER2 + 1.5 * HER3 + IGF1R
    RAS_GTP    = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK) * (1.0 - Ras_effect)
    RAF_to_PI3K = k['k_raf_pi3k'] * pCRAF / (k['Km_raf_pi3k'] + pCRAF + 1e-8)
    ERK_to_PI3K = k['k_erk_pi3k'] * pERK  / (k['Km_erk_pi3k'] + pERK + 1e-8)
    PI3K_input  = RTK_total * (1.0 - ERK_to_PI3K) + RAF_to_PI3K
    AKT_RAF_inhib = k['k_akt_raf'] * pAKT / (k['Km_akt_raf'] + pAKT + 1e-8)
    DUSP6_activity = k['k_dusp_cat'] * DUSP6 / (k['Km_dusp'] + DUSP6 + 1e-8)
    n_dusp = k['n_dusp']
    DUSP6_induction = (k['k_dusp_synth'] * ((pERK + eps)**n_dusp) /
                       (k['Km_dusp_s']**n_dusp + (pERK + eps)**n_dusp + 1e-8))
    dXdt = np.zeros(10)
    dXdt[0] = (k['k_egfr'] * (1.0 + drug_relief) * K_sat_egfr / (K_sat_egfr + pEGFR + 1e-8)
               - (k['k_egfr_deg'] + ERK_feedback) * pEGFR)
    dXdt[1] = (k['k_her2'] * (1.0 + drug_relief) * K_sat_her2 / (K_sat_her2 + HER2 + 1e-8)
               - (k['k_her2_deg'] + ERK_feedback) * HER2)
    dXdt[2] = (k['k_her3'] * (1.0 + 2.0 * drug_relief) * K_sat_her3 / (K_sat_her3 + HER3 + 1e-8)
               - (k['k_her3_deg'] + ERK_feedback) * HER3)
    dXdt[3] = (k['k_igf'] * (1.0 + drug_relief) * K_sat_igfr / (K_sat_igfr + IGF1R + 1e-8)
               - (k['k_igf_deg'] + ERK_feedback) * IGF1R)
    dXdt[4] = (k['k_craf'] * RAS_GTP * (1.0 - Vem_inhibition) * K_sat_craf / (K_sat_craf + pCRAF + 1e-8)
               + Vem_paradox
               - (k['k_craf_deg'] + AKT_RAF_inhib) * pCRAF)
    dXdt[5] = (k['k_mek'] * pCRAF * (1.0 - Tram_effect) * K_sat_mek / (K_sat_mek + pMEK + 1e-8)
               - k['k_mek_deg'] * pMEK)
    dXdt[6] = (k['k_erk'] * pMEK * K_sat_erk / (K_sat_erk + pERK + 1e-8)
               - (k['k_erk_deg'] + DUSP6_activity) * pERK)
    dXdt[7] = DUSP6_induction - k['k_dusp_deg'] * DUSP6
    dXdt[8] = (k['k_akt'] * PI3K_input * (1.0 - PI3Ki_effect) * K_sat_akt / (K_sat_akt + pAKT + 1e-8)
               - k['k_akt_deg'] * pAKT)
    dXdt[9] = (k['k_4ebp1'] * pAKT * K_sat_4ebp1 / (K_sat_4ebp1 + p4EBP1 + 1e-8)
               - k['k_4ebp1_deg'] * p4EBP1)
    return dXdt
def run_perturbation(
    model, scalers, k_params_dict, device,
    base_drugs, perturbed_drugs,
    t_switch=80.0, t_end=200.0,
    n_phase1=300, n_phase2=500,
):
    """
    Phase 1: PINN prediction under base_drugs (0 → t_switch).
             Reliable because this is a trained condition.
    Phase 2: ODE integration under perturbed_drugs (t_switch → t_end).
             Uses learned kinetic parameters — generalizes to novel conditions.
    """
    print(f"  Phase 1: PINN prediction  t = 0 → {t_switch}h ...")
    t_phase1 = np.linspace(0, t_switch, n_phase1)
    y_phase1 = model.predict(
        t_phase1, base_drugs, scalers, device=device, normalized=False
    )
    y0 = y_phase1[-1, :]
    print(f"  Initial conditions at t={t_switch}h (from PINN):")
    for i, sp in enumerate(SPECIES_ORDER):
        print(f"    {sp:>8s} = {y0[i]:.4f} A.U.")
    print(f"\n  Phase 2: ODE integration  t = {t_switch} → {t_end}h ...")
    print(f"           Using {len(k_params_dict)} trained kinetic parameters")
    print(f"           Drug condition: {perturbed_drugs}")
    t_eval_phase2 = np.linspace(t_switch, t_end, n_phase2)
    sol = solve_ivp(
        fun=lambda t, X: ode_rhs(t, X, k_params_dict, perturbed_drugs),
        t_span=(t_switch, t_end),
        y0=y0,
        method='RK45',
        t_eval=t_eval_phase2,
        rtol=1e-6,
        atol=1e-9,
        max_step=0.2,
    )
    if not sol.success:
        print(f"  ⚠ ODE solver: {sol.message}")
    else:
        print(f"  ✓ ODE integration successful ({sol.nfev} evaluations)")
    t_phase2 = sol.t
    y_phase2 = sol.y.T
    t_full = np.concatenate([t_phase1, t_phase2[1:]])
    y_full = np.concatenate([y_phase1, y_phase2[1:]], axis=0)
    return t_full, y_full, t_phase1, y_phase1, t_phase2, y_phase2
def run_control(model, scalers, device, base_drugs, t_end=200.0, n_points=800):
    """
    Control: Pure PINN prediction with base_drugs for the full time range.
    No ODE handoff — gives a smooth, continuous baseline since this is
    a trained condition where the PINN is reliable.
    """
    print(f"\n  Control: PINN prediction with base drugs t = 0 → {t_end}h ...")
    t_full = np.linspace(0, t_end, n_points)
    y_full = model.predict(
        t_full, base_drugs, scalers, device=device, normalized=False
    )
    return t_full, y_full
def plot_perturbation(
    t_pert, y_pert, t_ctrl, y_ctrl,
    t_switch, base_drugs, perturbed_drugs,
    output_dir, experiment_label,
):
    """2×5 panel figure — perturbed vs control for all 10 species."""
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    axes = axes.flatten()
    c_ctrl = '#636e72'
    c_pre  = '#0984e3'
    c_post = '#d63031'
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        ax.plot(t_ctrl, y_ctrl[:, i],
                color=c_ctrl, lw=1.5, ls='--', alpha=0.55,
                label='Control')
        mask_pre = t_pert <= t_switch
        ax.plot(t_pert[mask_pre], y_pert[mask_pre, i],
                color=c_pre, lw=2, label='Phase 1 (PINN)')
        mask_post = t_pert >= t_switch
        ax.plot(t_pert[mask_post], y_pert[mask_post, i],
                color=c_post, lw=2, label='Phase 2 (ODE)')
        ax.axvline(t_switch, color='#fdcb6e', lw=1.8, ls=':', alpha=0.9)
        ax.set_title(species, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (h)', fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel('Intensity (A.U.)', fontsize=10)
        ax.grid(True, ls='-', color='#e0e0e0', alpha=0.6)
        ax.set_xlim(0, t_pert[-1])
    base_str = f'Vem {base_drugs["vemurafenib"]} + Tram {base_drugs["trametinib"]} μM'
    legend_elements = [
        Line2D([0], [0], color=c_ctrl, ls='--', lw=1.5,
               label=f'Control: {base_str} (PINN, continuous)'),
        Line2D([0], [0], color=c_pre, lw=2,
               label=f'Phase 1: {base_str} (PINN)'),
        Line2D([0], [0], color=c_post, lw=2,
               label=f'Phase 2: {experiment_label} (ODE)'),
        Line2D([0], [0], color='#fdcb6e', ls=':', lw=1.8,
               label=f'Drug change @ t = {t_switch:.0f}h'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f'In-silico Perturbation — {experiment_label} at t = {t_switch:.0f}h\n'
        f'Phase 1: PINN (trained condition)  |  '
        f'Phase 2: ODE with learned kinetic parameters',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    safe_label = experiment_label.replace(' ', '_').replace('+', '_').replace('μ', 'u')
    for ext in ['pdf', 'png']:
        fname = os.path.join(
            output_dir,
            f'perturbation_{safe_label}_at_{t_switch:.0f}h.{ext}'
        )
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved → {fname}")
    plt.close(fig)
def plot_ode_from_ic(
    t_phase2, y_phase2, y0, t_switch,
    perturbed_drugs, output_dir, experiment_label,
):
    """
    Figure 2: Pure ODE simulation starting from collected initial conditions.
    Shows only the Phase 2 trajectory with ICs annotated.
    """
    fig, axes = plt.subplots(2, 5, figsize=(26, 10))
    axes = axes.flatten()
    c_ode = '#d63031'
    c_ic  = '#2d3436'
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        ax.plot(t_phase2, y_phase2[:, i],
                color=c_ode, lw=2.5, label='ODE Simulation')
        ax.scatter([t_switch], [y0[i]], color=c_ic, s=120, marker='D',
                   zorder=5, edgecolors='white', linewidths=2,
                   label=f'IC = {y0[i]:.3f}')
        ax.set_title(f'{species}  (IC = {y0[i]:.3f})', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (h)', fontsize=10)
        if i % 5 == 0:
            ax.set_ylabel('Intensity (A.U.)', fontsize=10)
        ax.grid(True, ls='-', color='#e0e0e0', alpha=0.6)
        ax.set_xlim(t_phase2[0], t_phase2[-1])
    drug_str = ', '.join(
        f'{name} {dose}μM' for name, dose in perturbed_drugs.items() if dose > 0
    )
    legend_elements = [
        Line2D([0], [0], color=c_ode, lw=2.5,
               label=f'ODE (trained kinetic parameters)'),
        Line2D([0], [0], color=c_ic, marker='D', ls='none', markersize=10,
               markeredgecolor='white', markeredgewidth=2,
               label=f'Initial Conditions (from PINN @ t={t_switch:.0f}h)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f'ODE Simulation from Collected ICs — {experiment_label}\n'
        f'ICs collected at t = {t_switch:.0f}h (PINN under Vem+Tram)  |  '
        f'Drugs: {drug_str}',
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    safe_label = experiment_label.replace(' ', '_').replace('+', '_').replace('μ', 'u')
    for ext in ['pdf', 'png']:
        fname = os.path.join(
            output_dir,
            f'ode_from_ic_{safe_label}_at_{t_switch:.0f}h.{ext}'
        )
        fig.savefig(fname, dpi=200, bbox_inches='tight')
        print(f"  ✓ Saved → {fname}")
    plt.close(fig)
def export_csv(t_pert, y_pert, t_ctrl, y_ctrl, t_switch, experiment_label, output_dir):
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    safe_label = experiment_label.replace(' ', '_').replace('+', '_').replace('μ', 'u')
    df = pd.DataFrame(y_pert, columns=SPECIES_ORDER)
    df.insert(0, 'time_h', t_pert)
    df.insert(1, 'phase', np.where(t_pert <= t_switch, 'Phase1_PINN', 'Phase2_ODE'))
    fname = os.path.join(output_dir, f'perturbation_{safe_label}_perturbed.csv')
    df.to_csv(fname, index=False)
    print(f"  ✓ Saved perturbed CSV → {fname}")
    df_c = pd.DataFrame(y_ctrl, columns=SPECIES_ORDER)
    df_c.insert(0, 'time_h', t_ctrl)
    df_c.insert(1, 'phase', np.where(t_ctrl <= t_switch, 'Phase1_PINN', 'Phase2_ODE'))
    fname_c = os.path.join(output_dir, f'perturbation_{safe_label}_control.csv')
    df_c.to_csv(fname_c, index=False)
    print(f"  ✓ Saved control CSV   → {fname_c}")
def print_key_params(k):
    """Print the most biologically interpretable trained parameters."""
    print(f"\n{'═'*60}")
    print(f"  TRAINED KINETIC PARAMETERS (used for ODE integration)")
    print(f"{'═'*60}")
    groups = {
        'Drug IC50s': ['IC50_vem', 'IC50_tram', 'IC50_pi3k', 'IC50_ras'],
        'Hill': ['hill_coeff'],
        'MAPK rates': ['k_craf', 'k_craf_deg', 'k_mek', 'k_mek_deg',
                       'k_erk', 'k_erk_deg'],
        'Feedback':   ['k_erk_rtk', 'k_erk_sos', 'k_akt_raf', 'k_paradox', 'k_up'],
    }
    for group_name, params in groups.items():
        print(f"  {group_name}:")
        for p in params:
            if p in k:
                print(f"    {p:>15s} = {k[p]:.4f}")
            else:
                print(f"    {p:>15s} = N/A")
    print(f"{'═'*60}")
def main():
    parser = argparse.ArgumentParser(
        description='PINN→ODE Drug Perturbation Experiment'
    )
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='Path to trained PINN checkpoint')
    parser.add_argument('--t_switch', type=float, default=80.0,
                        help='Time (hours) to introduce the perturbation')
    parser.add_argument('--t_end', type=float, default=200.0,
                        help='End time (hours)')
    parser.add_argument('--pi3k_dose', type=float, default=0.0,
                        help='PI3K inhibitor dose in Phase 2 (μM)')
    parser.add_argument('--tram_dose', type=float, default=None,
                        help='Override Trametinib dose in Phase 2 (μM)')
    parser.add_argument('--vem_dose', type=float, default=None,
                        help='Override Vemurafenib dose in Phase 2 (μM)')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.model):
        print(f"✗ Error: Model not found at '{args.model}'.")
        return
    print(f"Loading PINN model from {args.model} ...")
    model, scalers = load_pinn(args.model, device)
    print("Loading trained kinetic parameters ...")
    k = load_trained_k_params(args.model, device)
    print_key_params(k)
    perturbed_drugs = BASE_DRUGS.copy()
    if args.pi3k_dose > 0:
        perturbed_drugs['pi3k_inhibitor'] = args.pi3k_dose
    if args.tram_dose is not None:
        perturbed_drugs['trametinib'] = args.tram_dose
    if args.vem_dose is not None:
        perturbed_drugs['vemurafenib'] = args.vem_dose
    changes = []
    if perturbed_drugs['pi3k_inhibitor'] != BASE_DRUGS['pi3k_inhibitor']:
        changes.append(f"PI3Ki {perturbed_drugs['pi3k_inhibitor']}μM")
    if perturbed_drugs['trametinib'] != BASE_DRUGS['trametinib']:
        changes.append(f"Tram {perturbed_drugs['trametinib']}μM")
    if perturbed_drugs['vemurafenib'] != BASE_DRUGS['vemurafenib']:
        changes.append(f"Vem {perturbed_drugs['vemurafenib']}μM")
    experiment_label = 'Add ' + ' + '.join(changes) if changes else 'No Change'
    print(f"\n{'═'*60}")
    print(f"  PINN → ODE PERTURBATION EXPERIMENT")
    print(f"{'═'*60}")
    print(f"  Phase 1 (t = 0 → {args.t_switch}h):   PINN prediction")
    print(f"    Vemurafenib    = {BASE_DRUGS['vemurafenib']} μM")
    print(f"    Trametinib     = {BASE_DRUGS['trametinib']} μM")
    print(f"  Phase 2 (t = {args.t_switch} → {args.t_end}h):  ODE integration")
    for drug, dose in perturbed_drugs.items():
        marker = '  ← CHANGED' if dose != BASE_DRUGS[drug] else ''
        print(f"    {drug:15s} = {dose} μM{marker}")
    print(f"  Experiment: {experiment_label}")
    print(f"{'═'*60}\n")
    t_pert, y_pert, t_phase1, y_phase1, t_phase2, y_phase2 = run_perturbation(
        model, scalers, k, device,
        base_drugs=BASE_DRUGS,
        perturbed_drugs=perturbed_drugs,
        t_switch=args.t_switch,
        t_end=args.t_end,
    )
    y0_collected = y_phase1[-1, :]
    t_ctrl, y_ctrl = run_control(
        model, scalers, device,
        base_drugs=BASE_DRUGS,
        t_end=args.t_end,
    )
    print("\nGenerating Figure 1: Full perturbation plot ...")
    plot_perturbation(
        t_pert, y_pert, t_ctrl, y_ctrl,
        args.t_switch, BASE_DRUGS, perturbed_drugs,
        args.output_dir, experiment_label,
    )
    print("Generating Figure 2: ODE from initial conditions ...")
    plot_ode_from_ic(
        t_phase2, y_phase2, y0_collected, args.t_switch,
        perturbed_drugs, args.output_dir, experiment_label,
    )
    export_csv(t_pert, y_pert, t_ctrl, y_ctrl,
               args.t_switch, experiment_label, args.output_dir)
    print(f"\n{'─'*60}")
    print(f"  Δ Species at t = {args.t_end}h  (Perturbed − Control)")
    print(f"{'─'*60}")
    for i, sp in enumerate(SPECIES_ORDER):
        delta = y_pert[-1, i] - y_ctrl[-1, i]
        pct = 100 * delta / (abs(y_ctrl[-1, i]) + 1e-8)
        arrow = '↑' if delta > 0 else '↓'
        print(f"  {sp:>8s}:  {delta:+.4f} A.U.  ({pct:+.1f}%)  {arrow}")
    print(f"{'─'*60}")
    print(f"\n✓ All outputs saved to: {args.output_dir}/")
if __name__ == '__main__':
    main()
