import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pinn_model import PINN
from data_utils import SPECIES_ORDER, VEM_TRAM_DATA, VEM_ONLY_DATA, TRAM_ONLY_DATA
import seaborn as sns

def load_pinn_with_data(filepath, device='cpu'):
    """Load model and train/test data splits."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Auto-detect hidden_size from checkpoint
    state_dict = checkpoint['model_state_dict']
    hidden_size = state_dict['input_layer.weight'].shape[0]
    
    model = PINN(input_size=6, hidden_size=hidden_size, output_size=11).to(device)
    model.load_state_dict(state_dict)
    scalers = checkpoint['scalers']
    train_data = checkpoint.get('train_data', None)
    test_data = checkpoint.get('test_data', None)
    return model, scalers, train_data, test_data

def plot_extrapolation_results(model_path='pinn_model_best.pth', save_path='extrapolation_results.png'):
    """
    Plot training fit (Single Agents) and extrapolation to held-out Combination therapy.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data = load_pinn_with_data(model_path, device)
    model.eval()
    
    # 1. Prediction for Combination (the Test Set)
    combo_drugs = VEM_TRAM_DATA['drugs']
    t_smooth = np.linspace(0, 48, 200)
    y_combo_pred = model.predict(t_smooth, combo_drugs, scalers, device)
    
    # 2. Prediction for Single Agents (the Train Set)
    y_vem_pred = model.predict(t_smooth, VEM_ONLY_DATA['drugs'], scalers, device)
    y_tram_pred = model.predict(t_smooth, TRAM_ONLY_DATA['drugs'], scalers, device)

    # Compute R² for the combo (generalization performance)
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    # We need predictions at specific exp time points for R2
    t_exp = VEM_TRAM_DATA['time_points']
    y_combo_exp_pred = model.predict(t_exp, combo_drugs, scalers, device)
    
    # Extract test_data['y_norm'] specifically for Combo if possible, 
    # but test_data might contain the whole test set. 
    # In our new script, test_data = Combination.
    r2_combo = compute_r2(test_data['y_norm'], y_combo_exp_pred)
    
    # Create plot
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        
        # Plot model prediction for COMBO
        ax.plot(t_smooth, y_combo_pred[:, i], 'r-', linewidth=2.5, label='Combo Pred', zorder=4)
        
        # Plot model predictions for Single Agents (faded)
        ax.plot(t_smooth, y_vem_pred[:, i], 'g--', linewidth=1, label='Vem Only', alpha=0.5)
        ax.plot(t_smooth, y_tram_pred[:, i], 'b--', linewidth=1, label='Tram Only', alpha=0.5)
        
        # Plot experimental dots for COMBO (the "test" data)
        ax.scatter(test_data['t'], test_data['y_norm'][:, i], 
                  color='red', s=100, marker='s', 
                  label=f'Combo Exp (R²={r2_combo[i]:.2f})', 
                  zorder=5, edgecolors='darkred', linewidths=1)
        
        ax.set_title(f'{species}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Normalized Conc.', fontsize=10)
        ax.grid(alpha=0.2, linestyle=':')
        if i == 0:
            ax.legend(fontsize=8, loc='best')
    
    # Hide extra subplot
    if len(axes) > len(SPECIES_ORDER):
        axes[-1].axis('off')
    
    plt.suptitle('Inductive Bias: Predicting Combination Response from Single-Agent Training', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved extrapolation plot to {save_path}")
    return fig

def plot_training_history(history_file='training_history.csv', save_path='training_test_history.png'):
    """Plot training and test loss over epochs."""
    if not os.path.exists(history_file):
        print(f"Warning: {history_file} not found.")
        return None
        
    history = pd.read_csv(history_file)
    has_test_data = 'l_test' in history.columns and history['l_test'].notna().any()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All loss components
    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], label='Total Loss', linewidth=2)
    ax.plot(history['epoch'], history['l_data'], label='Data Loss (Train Agents)', linewidth=2)
    if has_test_data:
        ax.plot(history['epoch'], history['l_test'], label='Gen. Loss (Combo)', linewidth=2, linestyle='--')
    ax.plot(history['epoch'], history['l_physics'], label='Physics Loss', linewidth=2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Train vs Test comparison
    ax = axes[1]
    ax.plot(history['epoch'], history['l_data'], label='Train (Sgl Agents)', linewidth=2, color='green')
    if has_test_data:
        ax.plot(history['epoch'], history['l_test'], label='Test (Combo)', linewidth=2, color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Generalization: Single -> Combo', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history to {save_path}")
    return fig

def generate_prediction_table(model_path='pinn_model_best.pth', save_path='predictions_table.csv'):
    """Generate detailed prediction table for the Combination therapy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data = load_pinn_with_data(model_path, device)
    
    combo_drugs = VEM_TRAM_DATA['drugs']
    t_exp = VEM_TRAM_DATA['time_points']
    
    # Get predictions
    y_pred_norm = model.predict(t_exp, combo_drugs, scalers, device, normalized=True)
    y_pred_raw = model.predict(t_exp, combo_drugs, scalers, device, normalized=False)
    
    # Create table
    results = []
    for t_idx, t_val in enumerate(t_exp):
        for species_idx, species in enumerate(SPECIES_ORDER):
            true_norm = test_data['y_norm'][t_idx, species_idx]
            pred_norm = y_pred_norm[t_idx, species_idx]
            
            results.append({
                'Time (hrs)': t_val,
                'Species': species,
                'True Value (Norm)': true_norm,
                'Predicted Value (Norm)': pred_norm,
                'Error': true_norm - pred_norm,
                'Percent Error': 100 * (true_norm - pred_norm) / (true_norm + 1e-6),
                'Dataset': 'Combination (Test)'
            })
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions table to {save_path}")
    return df

if __name__ == "__main__":
    import os
    if os.path.exists('pinn_model_best.pth'):
        print("Generating extrapolation analysis...")
        plot_extrapolation_results()
        plot_training_history()
        generate_prediction_table()
        print("\n✓ Visualizations saved.")
    else:
        print("Error: pinn_model_best.pth not found.")
