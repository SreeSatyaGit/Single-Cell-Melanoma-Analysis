import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from inference import load_pinn, predict_new_combination
from data_utils import SPECIES_ORDER

def perform_vem_dose_response(model_path, tram_conc=0.1, vem_range=[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]):
    """
    Performs predictions for a range of Vemurafenib concentrations at a fixed Trametinib concentration.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    model, scalers = load_pinn(model_path, device)
    
    # Selection of species to plot for clarity (primary regulators)
    key_species = ['pERK', 'pMEK', 'pAKT', 'pCRAF', 'DUSP6', 'pS6K']
    
    # Prepare plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Store all results for potential CSV export
    all_results = []
    
    # Colors for different doses
    colors = plt.cm.viridis(np.linspace(0, 1, len(vem_range)))
    
    for i, vem_conc in enumerate(vem_range):
        print(f"Predicting for Vem={vem_conc}, Tram={tram_conc}...")
        
        drugs = {
            'vemurafenib': vem_conc,
            'trametinib': tram_conc,
            'pi3k_inhibitor': 0.0,
            'ras_inhibitor': 0.0
        }
        
        # Predict 0-48 hours
        results = predict_new_combination(model, drugs, scalers, device=device)
        results['vemurafenib'] = vem_conc
        all_results.append(results)
        
        # Plot key species
        for j, species in enumerate(key_species):
            ax = axes[j]
            ax.plot(results['time'], results[species], color=colors[i], 
                    label=f'Vem: {vem_conc} uM', linewidth=2.5)
            ax.set_title(f'{species} Response', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (h)')
            ax.set_ylabel('Normalized Intensity')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)

    for ax in axes:
        ax.legend(fontsize=9, loc='upper right')
        
    plt.suptitle(f'Dose Response: Varying Vemurafenib at Fixed Trametinib ({tram_conc} uM)', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save artifacts
    save_dir = 'analysis_results'
    os.makedirs(save_dir, exist_ok=True)
    
    plot_path = os.path.join(save_dir, f'vem_dose_response_tram{tram_conc}.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Saved dose-response plot to {plot_path}")
    
    # Save CSV data
    mega_df = pd.concat(all_results)
    csv_path = os.path.join(save_dir, f'vem_dose_response_tram{tram_conc}.csv')
    mega_df.to_csv(csv_path, index=False)
    print(f"Saved prediction data to {csv_path}")
    
    plt.show()

if __name__ == "__main__":
    MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'pinn_model_best.pth'
        
    if os.path.exists(MODEL_PATH):
        # We'll run a few variations if requested, but start with 0.1 uM Trametinib
        perform_vem_dose_response(MODEL_PATH, tram_conc=0.1)
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
