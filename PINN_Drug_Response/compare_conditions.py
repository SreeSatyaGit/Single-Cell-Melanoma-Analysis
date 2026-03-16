import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from inference import load_pinn
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'pinn_model_best.pth'
OUTPUT_DIR = 'analysis_results/condition_reports'
T_MAX = 80
T_POINTS = 300
SPECIES_TO_PLOT = ['pERK', 'pMEK', 'pAKT', 'pCRAF', 'DUSP6', 'p4EBP1']
def plot_all_conditions(experiments, model, scalers, device, output_dir):
    """Generates a multi-panel plot showing all conditions for key species."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    t_sim = np.linspace(0, T_MAX, T_POINTS)
    cond_style = {
        'No Drug (Basal Steady State)': {'color': 'black', 'lw': 3, 'label': 'No Drug (Baseline)'},
        'Vemurafenib Only (0.5)': {'color': 'tab:orange', 'lw': 1.5, 'label': 'Vemurafenib 0.5 $\mu$M'},
        'Trametinib Only (0.3)': {'color': 'tab:purple', 'lw': 1.5, 'label': 'Trametinib 0.3 $\mu$M'},
        'Vem + Tram Combo': {'color': 'tab:green', 'lw': 1.5, 'label': 'Vem 0.5 + Tram 0.3'},
        'Vem + PI3Ki Combo': {'color': '#4dc8b9', 'lw': 1.5, 'label': 'Vem 0.5 + PI3Ki 0.5'},
        'Vem + panRAS Combo': {'color': '#f5b041', 'lw': 1.5, 'label': 'Vem 0.5 + panRAS 0.5'}
    }
    predictions = {}
    for exp in experiments:
        y_pred = model.predict(t_sim, exp['drugs'], scalers, device=device, normalized=False)
        predictions[exp['name']] = y_pred
    for i, species in enumerate(SPECIES_TO_PLOT):
        ax = axes[i]
        species_idx = SPECIES_ORDER.index(species)
        for exp in experiments:
            cond_name = exp['name']
            y_pred = predictions[cond_name]
            style = cond_style.get(cond_name, {'color': 'gray', 'lw': 1.5, 'label': cond_name})
            ax.plot(t_sim, y_pred[:, species_idx], color=style['color'], linewidth=style['lw'], alpha=0.9, label=style['label'])
        ax.set_title(species, fontsize=14, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Intensity (A.U.)')
        ax.set_xlim(0, T_MAX)
        ax.grid(True, linestyle='-', color='#e0e0e0', alpha=0.7)
        if species == 'pERK':
            ax.legend(fontsize=8, loc='upper right')
    fig.suptitle('No-Drug Baseline vs Treated Conditions — Key Species', fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = os.path.join(output_dir, 'no_drug_vs_treated_key_species.pdf')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  - Saved report to {output_path}")
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Please train the model first.")
        return
    print(f"Loading model from {MODEL_PATH}...")
    model, scalers = load_pinn(MODEL_PATH, device)
    print(f"Generating combined report for {len(TRAINING_DATA_LIST)} conditions...")
    plot_all_conditions(TRAINING_DATA_LIST, model, scalers, device, OUTPUT_DIR)
    print(f"\n✓ Combined condition report saved to: {OUTPUT_DIR}/")
if __name__ == "__main__":
    main()
