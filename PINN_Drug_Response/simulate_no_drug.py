"""
No-Drug ODE Simulation from Zero/Low Initial Conditions
========================================================
Numerically integrates the MAPK/PI3K ODE system from physics_utils.py
using the **trained kinetic parameters** but with NO drugs and
ZERO (or low) initial conditions for all species.
This simulates a BRAF V600E melanoma cell "booting up" its signaling
from scratch — showing how constitutive pathway activity builds to
steady state purely from the learned physics.
Biology:
  - BRAF V600E drives constitutive RAS-RAF-MEK-ERK signaling.
  - Without drugs, the pathway should converge to a high basal steady
    state (especially pMEK, pERK).
  - RTKs provide upstream input; DUSP6 is induced by ERK as negative
    feedback; pAKT/p4EBP1 are driven by PI3K arm.
Outputs:
  1. Per-species trajectories from zero IC → steady state (PDF)
  2. Key species summary panel (PDF)
  3. Steady-state vs experimental t=0 bar chart (PDF)
  4. CSV of the full trajectory
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import solve_ivp
from inference import load_pinn
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
plt.switch_backend('Agg')
MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
OUTPUT_DIR = 'analysis_results/no_drug_baseline'
T_MAX = 200
T_POINTS = 1000
INITIAL_CONDITIONS = np.array([
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
    0.01,
])
KEY_SPECIES = ['pERK', 'pMEK', 'pAKT', 'pCRAF', 'DUSP6', 'p4EBP1']
ALL_SPECIES = SPECIES_ORDER
def load_kinetic_params(model_path):
    """Extract the learned k_params from the saved checkpoint."""
    if not os.path.exists(model_path):
        fallback = 'pinn_model_best.pth'
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {fallback}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if 'k_params_state_dict' in checkpoint:
        k_raw = checkpoint['k_params_state_dict']
        k = {name: val.item() for name, val in k_raw.items()}
        print(f"Loaded {len(k)} learned kinetic parameters from {model_path}")
    else:
        print("WARNING: No k_params found in checkpoint — using defaults from train_pinn.py")
        k = get_default_params()
    return k
def get_default_params():
    """Fallback defaults matching train_pinn.py initialization."""
    return {
        'IC50_vem': 0.8, 'IC50_tram': 0.3, 'IC50_pi3k': 0.5, 'IC50_ras': 0.5,
        'hill_coeff': 2.0, 'k_paradox': 0.25,
        'k_egfr': 0.5, 'k_egfr_deg': 0.2, 'k_her2': 0.4, 'k_her2_deg': 0.15,
        'k_her3': 0.4, 'k_her3_deg': 0.15, 'k_igf': 0.3, 'k_igf_deg': 0.2,
        'k_erk_rtk': 0.1, 'Km_rtk': 0.5, 'k_up': 0.3,
        'k_erk_sos': 0.4, 'Km_sos': 0.5, 'k_akt_rtk': 0.15, 'Km_artk': 0.5,
        'k_craf': 1.2, 'k_craf_deg': 0.35, 'k_mek': 1.0, 'k_mek_deg': 0.4,
        'k_erk': 1.2, 'k_erk_deg': 0.45, 'k_dusp_synth': 0.8, 'k_dusp_deg': 0.5,
        'k_dusp_cat': 0.6, 'Km_dusp': 0.4, 'Km_dusp_s': 0.4, 'n_dusp': 2.0,
        'k_raf_pi3k': 0.2, 'Km_raf_pi3k': 0.5, 'k_erk_pi3k': 0.45, 'Km_erk_pi3k': 0.5,
        'k_akt': 1.0, 'k_akt_deg': 0.4,
        'k_4ebp1': 0.85, 'k_4ebp1_deg': 0.45,
        'k_4ebp1_comp': 0.25, 'Km_4ebp1': 0.5, 'k_akt_raf': 0.5, 'Km_akt_raf': 0.5,
        'K_sat_egfr': 1.0, 'K_sat_her2': 2.0, 'K_sat_her3': 2.0, 'K_sat_igfr': 1.5,
        'K_sat_craf': 3.0, 'K_sat_mek': 2.5, 'K_sat_erk': 3.5,
        'K_sat_akt': 1.0, 'K_sat_4ebp1': 1.0,
    }
def ode_system(t, y, k):
    """
    RHS of the 10-species ODE system with ALL DRUGS = 0.
    Since all drug concentrations are zero:
      - Vem_inhibition = 0
      - Tram_effect    = 0
      - PI3Ki_effect   = 0
      - Ras_effect     = 0
      - drug_relief    = 0
      - Vem_paradox    = 0
    The system reduces to constitutive RTK→RAS→RAF→MEK→ERK signaling
    with DUSP6 negative feedback and cross-talk to PI3K/AKT/p4EBP1.
    """
    pEGFR  = max(y[0], 0.0)
    HER2   = max(y[1], 0.0)
    HER3   = max(y[2], 0.0)
    IGF1R  = max(y[3], 0.0)
    pCRAF  = max(y[4], 0.0)
    pMEK   = max(y[5], 0.0)
    pERK   = max(y[6], 0.0)
    DUSP6  = max(y[7], 0.0)
    pAKT   = max(y[8], 0.0)
    p4EBP1 = max(y[9], 0.0)
    eps = 1e-8
    def p(name, default=0.0):
        return abs(k.get(name, default))
    K_sat_egfr  = p('K_sat_egfr', 1.0)
    K_sat_her2  = p('K_sat_her2', 2.0)
    K_sat_her3  = p('K_sat_her3', 2.0)
    K_sat_igfr  = p('K_sat_igfr', 1.5)
    K_sat_craf  = p('K_sat_craf', 3.0)
    K_sat_mek   = p('K_sat_mek', 2.5)
    K_sat_erk   = p('K_sat_erk', 3.5)
    K_sat_akt   = p('K_sat_akt', 1.0)
    K_sat_4ebp1 = p('K_sat_4ebp1', 1.0)
    k_erk_rtk = p('k_erk_rtk', 0.1)
    Km_rtk    = p('Km_rtk', 0.5)
    ERK_feedback = k_erk_rtk * pERK / (Km_rtk + pERK + eps)
    k_erk_sos = p('k_erk_sos', 0.4)
    Km_sos    = p('Km_sos', 0.5)
    k_akt_rtk = p('k_akt_rtk', 0.15)
    Km_artk   = p('Km_artk', 0.5)
    ERK_to_SOS = k_erk_sos * pERK / (Km_sos + pERK + eps)
    AKT_to_RTK = k_akt_rtk * pAKT / (Km_artk + pAKT + eps)
    RTK_total  = pEGFR + HER2 + 1.5 * HER3 + IGF1R
    RAS_GTP    = RTK_total * (1.0 - ERK_to_SOS) * (1.0 - AKT_to_RTK)
    k_raf_pi3k  = p('k_raf_pi3k', 0.2)
    Km_raf_pi3k = p('Km_raf_pi3k', 0.5)
    k_erk_pi3k  = p('k_erk_pi3k', 0.45)
    Km_erk_pi3k = p('Km_erk_pi3k', 0.5)
    RAF_to_PI3K = k_raf_pi3k * pCRAF / (Km_raf_pi3k + pCRAF + eps)
    ERK_to_PI3K = k_erk_pi3k * pERK / (Km_erk_pi3k + pERK + eps)
    PI3K_input  = RTK_total * (1.0 - ERK_to_PI3K) + RAF_to_PI3K
    k_akt_raf  = p('k_akt_raf', 0.5)
    Km_akt_raf = p('Km_akt_raf', 0.5)
    AKT_RAF_inhib = k_akt_raf * pAKT / (Km_akt_raf + pAKT + eps)
    k_dusp_cat = p('k_dusp_cat', 0.6)
    Km_dusp    = p('Km_dusp', 0.4)
    DUSP6_activity = k_dusp_cat * DUSP6 / (Km_dusp + DUSP6 + eps)
    dydt = np.zeros(10)
    k_egfr     = p('k_egfr', 0.5)
    k_egfr_deg = p('k_egfr_deg', 0.2)
    dydt[0] = k_egfr * 1.0 * K_sat_egfr / (K_sat_egfr + pEGFR + eps)              - (k_egfr_deg + ERK_feedback) * pEGFR
    k_her2     = p('k_her2', 0.4)
    k_her2_deg = p('k_her2_deg', 0.15)
    dydt[1] = k_her2 * 1.0 * K_sat_her2 / (K_sat_her2 + HER2 + eps)              - (k_her2_deg + ERK_feedback) * HER2
    k_her3     = p('k_her3', 0.4)
    k_her3_deg = p('k_her3_deg', 0.15)
    dydt[2] = k_her3 * 1.0 * K_sat_her3 / (K_sat_her3 + HER3 + eps)              - (k_her3_deg + ERK_feedback) * HER3
    k_igf     = p('k_igf', 0.3)
    k_igf_deg = p('k_igf_deg', 0.2)
    dydt[3] = k_igf * 1.0 * K_sat_igfr / (K_sat_igfr + IGF1R + eps)              - (k_igf_deg + ERK_feedback) * IGF1R
    k_craf     = p('k_craf', 1.2)
    k_craf_deg = p('k_craf_deg', 0.35)
    dydt[4] = k_craf * RAS_GTP * 1.0 * K_sat_craf / (K_sat_craf + pCRAF + eps)              - (k_craf_deg + AKT_RAF_inhib) * pCRAF
    k_mek     = p('k_mek', 1.0)
    k_mek_deg = p('k_mek_deg', 0.4)
    dydt[5] = k_mek * pCRAF * 1.0 * K_sat_mek / (K_sat_mek + pMEK + eps)              - k_mek_deg * pMEK
    k_erk     = p('k_erk', 1.2)
    k_erk_deg = p('k_erk_deg', 0.45)
    dydt[6] = k_erk * pMEK * K_sat_erk / (K_sat_erk + pERK + eps)              - (k_erk_deg + DUSP6_activity) * pERK
    n_dusp       = min(max(k.get('n_dusp', 2.0), 1.5), 3.5)
    k_dusp_synth = p('k_dusp_synth', 0.8)
    Km_dusp_s    = p('Km_dusp_s', 0.4)
    k_dusp_deg   = p('k_dusp_deg', 0.5)
    DUSP6_induction = k_dusp_synth * (pERK**n_dusp) / (Km_dusp_s**n_dusp + pERK**n_dusp + eps)
    dydt[7] = DUSP6_induction - k_dusp_deg * DUSP6
    k_akt     = p('k_akt', 1.0)
    k_akt_deg = p('k_akt_deg', 0.4)
    dydt[8] = k_akt * PI3K_input * 1.0 * K_sat_akt / (K_sat_akt + pAKT + eps)              - k_akt_deg * pAKT
    k_4ebp1     = p('k_4ebp1', 0.85)
    k_4ebp1_deg = p('k_4ebp1_deg', 0.45)
    dydt[9] = k_4ebp1 * pAKT * K_sat_4ebp1 / (K_sat_4ebp1 + p4EBP1 + eps)              - k_4ebp1_deg * p4EBP1
    return dydt
def run_ode_simulation(k_params, y0, t_max, t_points):
    """Integrate the ODE from y0 over [0, t_max] using scipy."""
    t_eval = np.linspace(0, t_max, t_points)
    sol = solve_ivp(
        fun=lambda t, y: ode_system(t, y, k_params),
        t_span=(0, t_max),
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        max_step=0.5,
        rtol=1e-8,
        atol=1e-10
    )
    if not sol.success:
        print(f"WARNING: ODE solver returned: {sol.message}")
    rows = []
    for i, t_val in enumerate(sol.t):
        row = {'time': t_val}
        for sp_idx, sp in enumerate(ALL_SPECIES):
            row[sp] = sol.y[sp_idx, i]
        rows.append(row)
    return pd.DataFrame(rows), sol
def get_experimental_t0_values():
    """
    Collect t=0 values from all training conditions.
    These represent the pre-treatment basal level and serve as
    the biological ground truth for the untreated steady state.
    """
    t0_data = {sp: [] for sp in ALL_SPECIES}
    for exp in TRAINING_DATA_LIST:
        idx_0 = np.where(exp['time_points'] == 0)[0]
        if len(idx_0) > 0:
            idx_0 = idx_0[0]
            for sp in ALL_SPECIES:
                t0_data[sp].append(exp['species'][sp][idx_0])
    t0_mean = {sp: np.mean(vals) for sp, vals in t0_data.items()}
    t0_std  = {sp: np.std(vals) for sp, vals in t0_data.items()}
    return t0_mean, t0_std
def plot_all_species(df):
    """Plot all 10 species building up from zero to steady state."""
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    axes = axes.flatten()
    t0_mean, t0_std = get_experimental_t0_values()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, species in enumerate(ALL_SPECIES):
        ax = axes[i]
        ax.plot(df['time'], df[species], color=colors[i],
                linewidth=2.5, label='ODE (no drug, IC≈0)')
        if species in t0_mean:
            ax.axhline(t0_mean[species], color='red', linestyle='--',
                       alpha=0.7, linewidth=1.5,
                       label=f'Exp t=0 = {t0_mean[species]:.2f}')
            ax.axhspan(t0_mean[species] - t0_std[species],
                       t0_mean[species] + t0_std[species],
                       color='red', alpha=0.08)
        ss_val = df[species].iloc[-1]
        ax.axhline(ss_val, color='blue', linestyle=':', alpha=0.5,
                   label=f'ODE SS = {ss_val:.2f}')
        ax.set_title(species, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Intensity (A.U.)')
        ax.set_xlim(0, T_MAX)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='best')
    for j in range(len(ALL_SPECIES), len(axes)):
        axes[j].axis('off')
    fig.suptitle('No-Drug ODE Simulation — From Zero IC → Constitutive Steady State\n'
                 '(Uses learned kinetic parameters, all drugs = 0)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = os.path.join(OUTPUT_DIR, 'all_species_trajectories.pdf')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
def plot_key_species(df):
    """Focused panel on key MAPK/PI3K species."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    t0_mean, t0_std = get_experimental_t0_values()
    cmap_vals = plt.cm.viridis(np.linspace(0.15, 0.85, len(KEY_SPECIES)))
    for sp_idx, species in enumerate(KEY_SPECIES):
        ax = axes[sp_idx]
        ax.plot(df['time'], df[species], color=cmap_vals[sp_idx],
                linewidth=2.5, label='ODE trajectory')
        if species in t0_mean:
            ax.axhline(t0_mean[species], color='red', linestyle='--',
                       alpha=0.6, linewidth=1.5,
                       label=f'Exp t=0 mean = {t0_mean[species]:.2f}')
            ax.axhspan(t0_mean[species] - t0_std[species],
                       t0_mean[species] + t0_std[species],
                       color='red', alpha=0.08, label='± 1 SD')
        ax.set_title(species, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Intensity (A.U.)')
        ax.set_xlim(0, T_MAX)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Key Species — No Drug, Zero IC → Steady State',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUTPUT_DIR, 'key_species_summary.pdf')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
def plot_steady_state_comparison(df):
    """Bar chart: ODE steady state vs experimental t=0."""
    t0_mean, t0_std = get_experimental_t0_values()
    ss_pred = {sp: df[sp].iloc[-1] for sp in ALL_SPECIES}
    x = np.arange(len(ALL_SPECIES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2,
           [t0_mean[sp] for sp in ALL_SPECIES], width,
           label='Experimental t=0 (mean ± SD)',
           color='#E74C3C', alpha=0.85,
           yerr=[t0_std[sp] for sp in ALL_SPECIES],
           capsize=4, edgecolor='black', linewidth=0.6)
    ax.bar(x + width / 2,
           [ss_pred[sp] for sp in ALL_SPECIES], width,
           label='ODE Steady State (no drug, IC≈0)',
           color='#3498DB', alpha=0.85,
           edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_SPECIES, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Intensity (A.U.)', fontsize=12)
    ax.set_title('Steady-State Comparison: Experimental Baseline vs ODE (No Drug)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'steady_state_comparison.pdf')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("No-Drug ODE Simulation  (Zero IC → Constitutive Steady State)")
    print("=" * 65)
    print(f"All drugs = 0.0 µM")
    print(f"Initial conditions: {INITIAL_CONDITIONS}")
    print(f"Time range: 0–{T_MAX}h ({T_POINTS} points)\n")
    k_params = load_kinetic_params(MODEL_PATH)
    print("\n  Key learned parameters:")
    for name in ['k_erk', 'k_mek', 'k_craf', 'k_akt',
                 'K_sat_erk', 'K_sat_mek', 'K_sat_craf',
                 'k_erk_deg', 'k_mek_deg', 'k_dusp_cat']:
        val = k_params.get(name, 'N/A')
        if isinstance(val, float):
            print(f"    {name:>15s} = {val:.4f}")
    print("\nIntegrating ODE system from zero IC...")
    df, sol = run_ode_simulation(k_params, INITIAL_CONDITIONS, T_MAX, T_POINTS)
    t0_mean, _ = get_experimental_t0_values()
    print(f"\n  {'Species':>8s} | {'ODE SS':>8s} | {'Exp t=0':>8s} | {'Ratio':>6s}")
    print("  " + "-" * 40)
    for sp in ALL_SPECIES:
        ss = df[sp].iloc[-1]
        exp = t0_mean.get(sp, float('nan'))
        ratio = ss / exp if exp > 0.01 else float('nan')
        print(f"  {sp:>8s} | {ss:8.3f} | {exp:8.3f} | {ratio:6.2f}")
    csv_path = os.path.join(OUTPUT_DIR, 'simulation_no_drug_ode.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved trajectory to {csv_path} ({len(df)} rows)")
    print("\nGenerating plots...")
    plot_all_species(df)
    plot_key_species(df)
    plot_steady_state_comparison(df)
    print("\n" + "=" * 65)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("=" * 65)
if __name__ == "__main__":
    main()
