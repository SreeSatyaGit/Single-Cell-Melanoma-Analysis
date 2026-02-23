"""
Drug Dosage Simulation: Vemurafenib + PI3K Inhibitor Combinations
=================================================================
Simulates pathway trajectories across a grid of Vem/PI3Ki dosages
using the trained PINN global model.

Outputs:
  1. Per-species trajectory plots across all dosage combos (PDF)
  2. Dose-response heatmaps at key timepoints (PDF)
  3. Full prediction table (CSV)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from itertools import product

from inference import load_pinn
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST

plt.switch_backend('Agg')

# ────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────

MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
OUTPUT_DIR = 'analysis_results/dosage_simulation_vem_pi3k'

# Dosage grid for Vem + PI3Ki
VEM_DOSES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
PI3KI_DOSES = [0.0, 0.1, 0.25, 0.5, 1.0]

T_MAX = 80
T_POINTS = 300

HEATMAP_TIMEPOINTS = [1, 4, 8, 24, 48]

KEY_SPECIES = ['pERK', 'pMEK', 'pAKT', 'pCRAF', 'DUSP6', 'pS6K']


def load_model(model_path):
    if not os.path.exists(model_path):
        fallback = 'pinn_model_best.pth'
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"No model found at {model_path} or {fallback}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers = load_pinn(model_path, device)
    print(f"Loaded model from {model_path}")
    return model, scalers, device


def simulate_all_combinations(model, scalers, device):
    t_sim = np.linspace(0, T_MAX, T_POINTS)
    
    all_results = []
    for vem, pi3ki in product(VEM_DOSES, PI3KI_DOSES):
        drugs = {
            'vemurafenib': vem,
            'trametinib': 0.0,
            'pi3k_inhibitor': pi3ki,
            'ras_inhibitor': 0.0
        }
        
        y_pred = model.predict(t_sim, drugs, scalers, device=device, normalized=False)
        
        for t_idx, t_val in enumerate(t_sim):
            row = {
                'time': t_val,
                'vemurafenib': vem,
                'pi3k_inhibitor': pi3ki,
                'label': f'V{vem}+PI3Ki{pi3ki}'
            }
            for sp_idx, sp in enumerate(SPECIES_ORDER):
                row[sp] = y_pred[t_idx, sp_idx]
            all_results.append(row)
    
    return pd.DataFrame(all_results), t_sim





def find_experimental_data(vem, pi3ki):
    """Find matching experimental data for a specific drug combo."""
    for exp in TRAINING_DATA_LIST:
        d = exp['drugs']
        if abs(d['vemurafenib'] - vem) < 1e-4 and \
           abs(d['pi3k_inhibitor'] - pi3ki) < 1e-4 and \
           abs(d['trametinib'] - 0.0) < 1e-4 and \
           abs(d['ras_inhibitor'] - 0.0) < 1e-4:
            return exp
    return None


def plot_species_trajectories(df, t_sim):
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(VEM_DOSES)))
    
    for species in SPECIES_ORDER:
        fig, axes = plt.subplots(1, len(PI3KI_DOSES), figsize=(4 * len(PI3KI_DOSES), 4), sharey=True)
        if len(PI3KI_DOSES) == 1:
            axes = [axes]
        
        for ax_idx, pi3ki in enumerate(PI3KI_DOSES):
            ax = axes[ax_idx]
            
            for v_idx, vem in enumerate(VEM_DOSES):
                mask = (df['vemurafenib'] == vem) & (df['pi3k_inhibitor'] == pi3ki)
                subset = df[mask].sort_values('time')
                ax.plot(subset['time'], subset[species], color=colors[v_idx],
                        linewidth=2, label=f'Vem {vem}')
                
                # --- Superimpose Experimental Data ---
                exp = find_experimental_data(vem, pi3ki)
                if exp:
                    # Highlight specific datasets with bright colors and labels
                    if exp['name'] == 'Vemurafenib Only (0.5)':
                        ax.scatter(exp['time_points'], exp['species'][species], 
                                   color='gold', s=60, edgecolors='black', 
                                   linewidths=1.2, zorder=12, label='Exp: Vem Only')
                    elif exp['name'] == 'Vem + PI3Ki Combo':
                        ax.scatter(exp['time_points'], exp['species'][species], 
                                   color='orange', s=60, edgecolors='black', 
                                   linewidths=1.2, zorder=12, label='Exp: Vem+PI3Ki')
                    else:
                        ax.scatter(exp['time_points'], exp['species'][species], 
                                   color=colors[v_idx], s=45, edgecolors='white', 
                                   linewidths=1.2, zorder=10, alpha=0.8)
                
            
            ax.set_title(f'PI3Ki = {pi3ki} µM', fontsize=11, fontweight='bold')
            ax.set_xlabel('Time (h)')
            ax.set_xlim(0, T_MAX)
            ax.grid(alpha=0.3)
            if ax_idx == 0:
                ax.set_ylabel(f'{species} (A.U.)')
        
            # Add legend to each subplot to show experimental labels if they exist
            ax.legend(fontsize=7, loc='upper right')
        fig.suptitle(f'{species} — Vem + PI3Ki Dosage Simulation', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        
        path = os.path.join(OUTPUT_DIR, f'trajectory_{species}.pdf')
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved {path}")


def plot_key_species_summary(df):
    combos = list(product(VEM_DOSES, PI3KI_DOSES))
    total_doses = [v + p for v, p in combos]
    norm = plt.Normalize(min(total_doses), max(total_doses))
    cmap = plt.cm.coolwarm
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for sp_idx, species in enumerate(KEY_SPECIES):
        ax = axes[sp_idx]
        
        for vem, pi3ki in combos:
            mask = (df['vemurafenib'] == vem) & (df['pi3k_inhibitor'] == pi3ki)
            subset = df[mask].sort_values('time')
            color = cmap(norm(vem + pi3ki))
            ax.plot(subset['time'], subset[species], color=color, linewidth=1.5, alpha=0.7)
            
            # --- Superimpose Experimental Data ---
            exp = find_experimental_data(vem, pi3ki)
            if exp is not None:
                # Check for specific highlights
                if exp['name'] == 'Vemurafenib Only (0.5)':
                    ax.scatter(exp['time_points'], exp['species'][species], 
                               color='gold', s=65, edgecolors='black', 
                               linewidths=1.2, zorder=15, label='Experimental: Vem Only')
                elif exp['name'] == 'Vem + PI3Ki Combo':
                    ax.scatter(exp['time_points'], exp['species'][species], 
                               color='orange', s=65, edgecolors='black', 
                               linewidths=1.2, zorder=15, label='Experimental: Vem+PI3Ki')
                else:
                    ax.scatter(exp['time_points'], exp['species'][species], 
                               color=color, s=35, edgecolors='black', 
                               linewidths=0.7, zorder=10)
        
        # Add legend to show experimental labels once
        if sp_idx == 0:
            ax.legend(fontsize=9, loc='upper right', frameon=True)
        
        ax.set_title(species, fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Intensity (A.U.)')
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-0.1, 3.5)
        ax.grid(alpha=0.3)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label('Total Drug Load (Vem + PI3Ki µM)', fontsize=11)
    
    fig.suptitle('Key Species Response — Vem + PI3Ki Dosage Sweep', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    
    path = os.path.join(OUTPUT_DIR, 'key_species_summary.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_dose_response_heatmaps(df):
    fig, axes = plt.subplots(len(KEY_SPECIES), len(HEATMAP_TIMEPOINTS),
                             figsize=(3.5 * len(HEATMAP_TIMEPOINTS), 3 * len(KEY_SPECIES)))
    
    for sp_idx, species in enumerate(KEY_SPECIES):
        for t_idx, t_snap in enumerate(HEATMAP_TIMEPOINTS):
            ax = axes[sp_idx, t_idx]
            
            t_sim_vals = df['time'].unique()
            t_closest = t_sim_vals[np.argmin(np.abs(t_sim_vals - t_snap))]
            snapshot = df[np.abs(df['time'] - t_closest) < 1e-4]
            
            heatmap = np.zeros((len(PI3KI_DOSES), len(VEM_DOSES)))
            for v_i, vem in enumerate(VEM_DOSES):
                for p_i, pi3ki in enumerate(PI3KI_DOSES):
                    row = snapshot[(snapshot['vemurafenib'] == vem) & (snapshot['pi3k_inhibitor'] == pi3ki)]
                    if len(row) > 0:
                        heatmap[p_i, v_i] = row[species].values[0]
            
            im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis',
                          extent=[min(VEM_DOSES), max(VEM_DOSES), min(PI3KI_DOSES), max(PI3KI_DOSES)])
            
            if sp_idx == 0:
                ax.set_title(f't = {t_snap}h', fontsize=11, fontweight='bold')
            if t_idx == 0:
                ax.set_ylabel(f'{species}\nPI3Ki (µM)', fontsize=9)
            else:
                ax.set_ylabel('')
            if sp_idx == len(KEY_SPECIES) - 1:
                ax.set_xlabel('Vem (µM)', fontsize=9)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    fig.suptitle('Dose-Response Heatmaps — Vem × PI3Ki Grid', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = os.path.join(OUTPUT_DIR, 'dose_response_heatmaps.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("PINN Drug Dosage Simulation: Vemurafenib + PI3K Inhibitor")
    print("=" * 60)
    print(f"Vem doses:   {VEM_DOSES}")
    print(f"PI3Ki doses: {PI3KI_DOSES}")
    print(f"Total combinations: {len(VEM_DOSES) * len(PI3KI_DOSES)}")
    print(f"Time range: 0–{T_MAX}h ({T_POINTS} points)\n")
    
    model, scalers, device = load_model(MODEL_PATH)
    
    print("\nSimulating all dosage combinations...")
    df, t_sim = simulate_all_combinations(model, scalers, device)
    
    csv_path = os.path.join(OUTPUT_DIR, 'simulation_vem_pi3ki_all.csv')
    df.to_csv(csv_path, index=False)
    print(f"  Saved simulation data to {csv_path} ({len(df)} rows)")
    
    print("\nGenerating per-species trajectory plots...")
    plot_species_trajectories(df, t_sim)
    
    print("\nGenerating key species summary...")
    plot_key_species_summary(df)
    
    print("\nGenerating dose-response heatmaps...")
    plot_dose_response_heatmaps(df)
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
