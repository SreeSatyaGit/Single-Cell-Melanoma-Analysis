import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pinn_model import PINN
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
import seaborn as sns
import os

def load_pinn_with_data(filepath, device='cpu'):
    """Load model and train/test data splits along with condition metadata."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    # Auto-detect hidden_size from checkpoint
    state_dict = checkpoint['model_state_dict']
    # Use the first residual block to detect hidden size
    hidden_size = state_dict['res_blocks.0.fc1.weight'].shape[0]
    
    model = PINN(input_size=5, hidden_size=hidden_size, output_size=11).to(device)
    model.load_state_dict(state_dict)
    
    scalers = checkpoint['scalers']
    train_data = checkpoint.get('train_data', None)
    test_data = checkpoint.get('test_data', None)
    
    # Try to reconstruct drugs from train_data if not explicitly saved
    drugs_dict = None
    if train_data is not None and 'drugs' in train_data:
        drugs_vec = train_data['drugs'][0]
        drugs_dict = {
            'vemurafenib': float(drugs_vec[0]),
            'trametinib': float(drugs_vec[1]),
            'pi3k_inhibitor': float(drugs_vec[2]),
            'ras_inhibitor': float(drugs_vec[3])
        }
        
    return model, scalers, train_data, test_data, drugs_dict

def plot_extrapolation_results(model_path='pinn_model_best.pth', save_path='extrapolation_results.png', drugs_dict_override=None):
    """
    Plot training fit for a specific condition.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data, drugs_dict = load_pinn_with_data(model_path, device)
    model.eval()
    
    # Check for test data availability
    has_test_data = test_data is not None and len(test_data.get('t', [])) > 0
    
    # Use override if provided, otherwise use checkpoint or list default
    if drugs_dict_override is not None:
        final_drugs = drugs_dict_override
    elif drugs_dict is not None:
        final_drugs = drugs_dict
    else:
        final_drugs = TRAINING_DATA_LIST[0]['drugs']
        
    condition_name = os.path.basename(save_path).replace('fit_', '').replace('.png', '').replace('_', ' ')
    
    # Generate smooth predictions from 0 to 48 hours
    t_smooth = np.linspace(0, 48, 200)
    y_smooth = model.predict(t_smooth, final_drugs, scalers, device)
    
    # Get predictions at training points
    # Need to filter train_data for THIS specific drug combination if it's a global model
    if train_data is not None:
        # Match drug vector (approximate due to float)
        target_vec = np.array([final_drugs['vemurafenib'], final_drugs['trametinib'], 
                              final_drugs['pi3k_inhibitor'], final_drugs['ras_inhibitor']])
        
        # Check matching rows in train_data['drugs']
        mask = np.all(np.abs(train_data['drugs'] - target_vec) < 1e-4, axis=1)
        
        if np.any(mask):
            t_plot = train_data['t'][mask]
            y_plot = train_data['y_norm'][mask]
        else:
            # If no matches (e.g. prediction on new dose), don't plot data points
            t_plot, y_plot = None, None
    else:
        t_plot, y_plot = None, None

    # Get predictions at the plot points for R2
    y_train_pred = model.predict(t_plot, final_drugs, scalers, device) if t_plot is not None else None
    
    # Get predictions at test points (if any)
    if has_test_data:
        y_test_pred = model.predict(test_data['t'], drugs_dict, scalers, device)
    
    # Compute R² for training and test
    def compute_r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
        return 1 - (ss_res / (ss_tot + 1e-8))
    
    r2_train = compute_r2(y_plot, y_train_pred) if y_plot is not None else None
    r2_test = compute_r2(test_data['y_norm'], y_test_pred) if has_test_data else None
    
    # Create plot
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        
        # Plot smooth model curve
        ax.plot(t_smooth, y_smooth[:, i], 'b-', linewidth=2, label='PINN', alpha=0.8)
        
        if t_plot is not None:
            # Determine label safely
            r2_val = r2_train[i] if r2_train is not None else 0.0
            label_str = f'Data (R²={r2_val:.2f})'
            
            ax.scatter(t_plot, y_plot[:, i], 
                      color='green', s=100, marker='o', 
                      label=label_str, 
                      zorder=5, edgecolors='darkgreen', linewidths=2)
        
        # Plot test data points if a holdout exists
        if has_test_data:
            ax.scatter(test_data['t'], test_data['y_norm'][:, i], 
                      color='red', s=100, marker='s', 
                      label=f'Test (R²={r2_test[i]:.3f})', 
                      zorder=5, edgecolors='darkred', linewidths=2)
            
            # Add vertical line at train/test boundary
            ax.axvline(x=8, color='gray', linestyle='--', alpha=0.5, label='Train/Test Split')
        
        ax.set_title(f'{species}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Concentration (AU)', fontsize=10)
        ax.set_ylim(-0.1, 1.1)
        ax.grid(alpha=0.3, linestyle=':')
        ax.legend(fontsize=8, loc='best')
        
        # Highlight extrapolation region
        if has_test_data:
            ax.axvspan(8, 48, alpha=0.1, color='red', label='Extrapolation')
    
    # Hide extra subplot
    if len(axes) > len(SPECIES_ORDER):
        axes[-1].axis('off')
    
    title = f'PINN Global Model - Fit for: {condition_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved extrapolation plot to {save_path}")
    plt.close()
    
    # Print summary statistics
    if has_test_data:
        print("\n=== EXTRAPOLATION PERFORMANCE SUMMARY ===")
        if r2_train is not None:
            print(f"Training R² (mean): {np.mean(r2_train):.3f} ± {np.std(r2_train):.3f}")
        print(f"Test R² (mean): {np.mean(r2_test):.3f} ± {np.std(r2_test):.3f}")
        print("\nPer-species Test R²:")
        for species, r2 in zip(SPECIES_ORDER, r2_test):
            print(f"  {species:10s}: {r2:.3f}")
    else:
        print("\n=== TRAINING FIT SUMMARY ===")
        if r2_train is not None:
            print(f"Training R² (mean): {np.mean(r2_train):.3f} ± {np.std(r2_train):.3f}")
        else:
            print("No training data available for this condition (pure extrapolation).")

def plot_training_history(history_file='training_history.csv', save_path='training_test_history.png'):
    """Plot training and test loss over epochs."""
    history = pd.read_csv(history_file)
    has_test_data = history['l_test'].notna().any()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All loss components
    ax = axes[0]
    ax.plot(history['epoch'], history['loss'], label='Total Loss', linewidth=2)
    ax.plot(history['epoch'], history['l_data'], label='Data Loss (Train)', linewidth=2)
    if has_test_data:
        ax.plot(history['epoch'], history['l_test'], label='Data Loss (Test)', linewidth=2, linestyle='--')
    ax.plot(history['epoch'], history['l_physics'], label='Physics Loss', linewidth=2, alpha=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training History', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Train vs Test comparison
    ax = axes[1]
    ax.plot(history['epoch'], history['l_data'], label='Train Loss', linewidth=2, color='green')
    if has_test_data:
        ax.plot(history['epoch'], history['l_test'], label='Test Loss (Extrapolation)', linewidth=2, color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('MSE Loss', fontsize=12)
    ax.set_title('Train vs Test Performance', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history to {save_path}")
    plt.close()

def generate_prediction_table(model_path='pinn_model_best.pth', save_path='predictions_table.csv'):
    """Generate detailed prediction table for a specific model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data, drugs_dict = load_pinn_with_data(model_path, device)
    
    if drugs_dict is None:
        drugs_dict = TRAINING_DATA_LIST[0]['drugs']
    
    # Combine train and test data
    has_test_data = test_data is not None and len(test_data.get('t', [])) > 0
    
    # Create DataFrame for predictions at all unique timepoints
    all_times = np.concatenate([train_data['t'], test_data['t']]) if has_test_data else train_data['t']
    all_y_true_norm = np.concatenate([train_data['y_norm'], test_data['y_norm']]) if has_test_data else train_data['y_norm']
    all_y_true_raw = np.concatenate([train_data['y_raw'], test_data['y_raw']]) if has_test_data else train_data['y_raw']
    
    # Get predictions (normalized and raw)
    y_pred_norm = model.predict(all_times, drugs_dict, scalers, device, normalized=True)
    y_pred_raw = model.predict(all_times, drugs_dict, scalers, device, normalized=False)
    
    # Create table
    results = []
    for t_idx, t_val in enumerate(all_times):
        is_train = t_val <= 8 if has_test_data else True
        for species_idx, species in enumerate(SPECIES_ORDER):
            true_norm = all_y_true_norm[t_idx, species_idx]
            pred_norm = y_pred_norm[t_idx, species_idx]
            
            results.append({
                'Time (hrs)': t_val,
                'Species': species,
                'True Value': true_norm,        # Keep standard name for 0-1 range
                'Predicted Value': pred_norm,   # Keep standard name for 0-1 range
                'True Value (Raw)': all_y_true_raw[t_idx, species_idx],
                'Predicted Value (Raw)': y_pred_raw[t_idx, species_idx],
                'Error': true_norm - pred_norm,
                'Percent Error': 100 * (true_norm - pred_norm) / (true_norm + 1e-6),
                'Dataset': 'Train' if is_train else 'Test'
            })
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions table to {save_path}")
    return df

if __name__ == "__main__":
    import os
    
    # Check default location from config, then fallback
    default_model = 'results/nature_submission/pinn_model_global.pth'
    if not os.path.exists(default_model):
        default_model = 'pinn_model_best.pth'
    
    if os.path.exists(default_model):
        print(f"Generating extrapolation analysis using {default_model}...")
        
        # Determine history file path
        if 'pinn_model_global.pth' in default_model:
            history_file = default_model.replace('pinn_model_global.pth', 'history_global.csv')
        else:
            history_file = 'training_history.csv'
            
        plot_extrapolation_results(model_path=default_model)
        
        if os.path.exists(history_file):
            plot_training_history(history_file=history_file)
            
        df = generate_prediction_table(model_path=default_model)
        print("\n✓ All visualizations complete!")
    else:
        print(f"Error: Model not found at {default_model}. Train the model first.")
