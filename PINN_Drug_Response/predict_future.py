import torch
import numpy as np
import pandas as pd
import os
from inference import load_pinn
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

def predict_future_timestamps(model_path, drugs_dict, future_times, condition_label):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model from {model_path}...")
    model, scalers = load_pinn(model_path, device)
    
    # 0. Find experimental data for this condition
    exp_match = None
    for exp in TRAINING_DATA_LIST:
        # Check drug match (approximate)
        match = True
        for drug, val in drugs_dict.items():
            if abs(exp['drugs'][drug] - val) > 1e-4:
                match = False
                break
        if match:
            exp_match = exp
            break
    
    # Perform prediction for original range + future to see the curve
    t_full = np.linspace(0, 80, 240)
    y_full = model.predict(t_full, drugs_dict, scalers, device=device, normalized=False)
    
    # Predict specific future points requested
    t_eval = np.array(future_times, dtype=np.float32)
    predictions = model.predict(t_eval, drugs_dict, scalers, device=device, normalized=False)
    
    # 1. Create Data Table
    df = pd.DataFrame(predictions, columns=SPECIES_ORDER, index=future_times)
    df.index.name = 'Time (h)'
    print(f"\n--- Future Predictions for {condition_label} ---")
    print(df)
    
    # 2. Create Figure
    fig, axes = plt.subplots(4, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        # Plot full trajectory
        ax.plot(t_full, y_full[:, i], 'b-', linewidth=2, label='PINN Projection')
        
        # Mark future points
        ax.scatter(future_times, predictions[:, i], color='red', s=80, marker='x', zorder=5, label='Future Points')
        
        # Plot experimental points if found
        if exp_match:
            ax.scatter(exp_match['time_points'], exp_match['species'][species], 
                       color='green', s=60, edgecolors='black', label='Exp Data', zorder=4)
        
        # Reference line for training end
        ax.axvline(x=48, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title(species, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('Intensity (Raw)')
        ax.set_xlim(0, 80)
        ax.set_ylim(-0.1, 3.0)
        ax.grid(alpha=0.3)
        if i == 0: ax.legend()

    # Hide extra subplots
    for i in range(len(SPECIES_ORDER), len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(f'Long-term Projection vs Experimental Data: {condition_label}\n(Vem: {drugs_dict["vemurafenib"]}, Tram: {drugs_dict["trametinib"]}, PI3K: {drugs_dict["pi3k_inhibitor"]})', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 3. Save Artifacts
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, f'future_predictions_{condition_label.replace(" ", "_")}.csv')
    plot_path = os.path.join(output_dir, f'future_projection_{condition_label.replace(" ", "_")}.pdf')
    
    df.to_csv(csv_path)
    plt.savefig(plot_path)
    
    print(f"\nSaved predictions to {csv_path}")
    print(f"Saved figure to {plot_path}")
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = 'results/nature_submission/pinn_model_global.pth'
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'pinn_model_best.pth'
        
    # Condition: Vem + Tram Combo (from data_utils.py)
    vem_tram_drugs = {
        'vemurafenib': 0.5,
        'trametinib': 0.3,
        'pi3k_inhibitor': 0.0,
        'ras_inhibitor': 0.0
    }
    
    future_times = [50, 55, 60, 65, 70, 75, 80]
    
    if os.path.exists(MODEL_PATH):
        predict_future_timestamps(MODEL_PATH, vem_tram_drugs, future_times, "Vem + Tram Combo")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")
