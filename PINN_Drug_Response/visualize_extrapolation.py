import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Dict

from pinn_model import PINN
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
import seaborn as sns
import os
def load_pinn_with_data(filepath, device='cpu'):
    """Load model and train/test data splits along with condition metadata."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    hidden_size = state_dict['res_blocks.0.fc1.weight'].shape[0]
    output_size = state_dict['output_layer.bias'].shape[0]
    print(f"DEBUG: Loading model with hidden_size={hidden_size}, output_size={output_size}")
    model = PINN(input_size=5, hidden_size=hidden_size, output_size=output_size).to(device)
    model.load_state_dict(state_dict)
    scalers = checkpoint['scalers']
    train_data = checkpoint.get('train_data', None)
    test_data  = checkpoint.get('test_data', None)

    # Derive drug vector from saved condition_name, not from row 0 of train_data
    condition_name = checkpoint.get('condition_name', None)
    drugs_dict = None
    if condition_name and condition_name != 'global':
        match = [e for e in TRAINING_DATA_LIST if e['name'] == condition_name]
        if match:
            drugs_dict = match[0]['drugs']
    
    # Fallback: if global model or condition not found, extract unique drug
    # vectors from train_data and use the one with highest drug sum
    # (most treated condition, avoids picking the No Drug row)
    if drugs_dict is None and train_data is not None and 'drugs' in train_data:
        unique_drugs = np.unique(train_data['drugs'], axis=0)
        drug_sums = unique_drugs.sum(axis=1)
        best = unique_drugs[np.argmax(drug_sums)]
        drugs_dict = {
            'vemurafenib':  float(best[0]),
            'trametinib':   float(best[1]),
            'pi3k_inhibitor': float(best[2]),
            'ras_inhibitor':  float(best[3])
        }

    return model, scalers, train_data, test_data, drugs_dict
def plot_extrapolation_results(model_path='pinn_model_best.pth', save_path='extrapolation_results.png', drugs_dict_override=None):
    """
    Plot training fit for a specific condition.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data, drugs_dict = load_pinn_with_data(model_path, device)
    model.eval()
    has_test_data = test_data is not None and len(test_data.get('t', [])) > 0
    if drugs_dict_override is not None:
        final_drugs = drugs_dict_override
    elif drugs_dict is not None:
        final_drugs = drugs_dict
    else:
        final_drugs = TRAINING_DATA_LIST[0]['drugs']
    condition_name = os.path.basename(save_path).replace('fit_', '').replace('.png', '').replace('_', ' ')
    t_smooth = np.linspace(0, 48, 200)
    y_smooth = model.predict(t_smooth, final_drugs, scalers, device)
    if train_data is not None:
        target_vec = np.array([final_drugs['vemurafenib'], final_drugs['trametinib'],
                              final_drugs['pi3k_inhibitor'], final_drugs['ras_inhibitor']])
        mask = np.all(np.abs(train_data['drugs'] - target_vec) < 1e-4, axis=1)
        if np.any(mask):
            t_plot = train_data['t'][mask]
            y_plot = train_data['y_norm'][mask]
        else:
            t_plot, y_plot = None, None
    else:
        t_plot, y_plot = None, None
    y_train_pred = model.predict(t_plot, final_drugs, scalers, device) if t_plot is not None else None
    cond_test_t, cond_test_y = None, None
    if has_test_data:
        target_vec = np.array([final_drugs['vemurafenib'], final_drugs['trametinib'],
                              final_drugs['pi3k_inhibitor'], final_drugs['ras_inhibitor']])
        test_mask = np.all(np.abs(test_data['drugs'] - target_vec) < 1e-4, axis=1)
        if np.any(test_mask):
            cond_test_t = test_data['t'][test_mask]
            cond_test_y = test_data['y_norm'][test_mask]
            y_test_pred = model.predict(cond_test_t, final_drugs, scalers, device)
        else:
            has_test_data = False
    def compute_r2(y_true, y_pred):
        if y_true.shape[0] <= 1:
            return np.full(y_true.shape[1], np.nan)
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
        return 1 - (ss_res / (ss_tot + 1e-8))
    r2_train = compute_r2(y_plot, y_train_pred) if y_plot is not None else None
    r2_test = compute_r2(cond_test_y, y_test_pred) if has_test_data and cond_test_y is not None else None
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()
    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        ax.plot(t_smooth, y_smooth[:, i], 'b-', linewidth=2, label='PINN', alpha=0.8)
        if t_plot is not None:
            r2_val = r2_train[i] if r2_train is not None else 0.0
            label_str = f'Data (R²={r2_val:.2f})'
            ax.scatter(t_plot, y_plot[:, i],
                      color='green', s=100, marker='o',
                      label=label_str,
                      zorder=5, edgecolors='darkgreen', linewidths=2)
        if has_test_data and cond_test_y is not None:
            ax.scatter(cond_test_t, cond_test_y[:, i],
                      color='red', s=100, marker='s',
                      label=f'Test (R²={r2_test[i]:.3f})',
                      zorder=5, edgecolors='darkred', linewidths=2)
    if len(axes) > len(SPECIES_ORDER):
        axes[-1].axis('off')
    title = f'PINN Global Model - Fit for: {condition_name}'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved extrapolation plot to {save_path}")
    plt.close()
    if has_test_data:
        print("\n=== EXTRAPOLATION PERFORMANCE SUMMARY ===")
        if r2_train is not None:
            valid = r2_train[~np.isnan(r2_train)]
            if len(valid) > 0:
                print(f"Training R² (mean): {np.mean(valid):.3f} ± {np.std(valid):.3f}")
            else:
                print("Training R² (mean): N/A (single data point)")
        if r2_test is not None:
            print(f"Test R² (mean): {np.mean(r2_test[~np.isnan(r2_test)]):.3f} ± {np.std(r2_test[~np.isnan(r2_test)]):.3f}")
            print("\nPer-species Test R²:")
            for species, r2 in zip(SPECIES_ORDER, r2_test):
                val_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                print(f"  {species:10s}: {val_str}")
    else:
        print("\n=== TRAINING FIT SUMMARY ===")
        if r2_train is not None:
            valid = r2_train[~np.isnan(r2_train)]
            if len(valid) > 0:
                print(f"Training R² (mean): {np.mean(valid):.3f} ± {np.std(valid):.3f}")
            else:
                print("Training R² (mean): N/A (single data point)")
        else:
            print("No training data available for this condition (pure extrapolation).")
def plot_training_history(history_file='training_history.csv', save_path='training_test_history.png'):
    """Plot training and test loss over epochs."""
    history = pd.read_csv(history_file)
    has_test_data = history['l_test'].notna().any()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
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
def generate_prediction_table(
    model_path: str = 'pinn_model_best.pth',
    save_path: str = 'predictions_table.csv',
    drugs_dict_override: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Generate detailed prediction table for a specific model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers, train_data, test_data, drugs_dict_loaded = load_pinn_with_data(model_path, device)
    
    final_drugs = drugs_dict_override if drugs_dict_override is not None else drugs_dict_loaded
    if final_drugs is None:
        raise ValueError("No drug vector available. Pass drugs_dict_override explicitly.")

    target_vec = np.array([
        final_drugs['vemurafenib'],
        final_drugs['trametinib'],
        final_drugs['pi3k_inhibitor'],
        final_drugs['ras_inhibitor'],
    ], dtype=np.float32)

    has_test_data = test_data is not None and len(test_data.get('t', [])) > 0

    # Filter train_data to this condition only
    train_mask = np.all(np.abs(train_data['drugs'] - target_vec) < 1e-4, axis=1)
    t_train_cond      = train_data['t'][train_mask]
    y_train_norm_cond = train_data['y_norm'][train_mask]
    y_train_raw_cond  = train_data['y_raw'][train_mask]

    # Filter test_data to this condition only
    if has_test_data:
        test_mask = np.all(np.abs(test_data['drugs'] - target_vec) < 1e-4, axis=1)
        if np.any(test_mask):
            t_test_cond      = test_data['t'][test_mask]
            y_test_norm_cond = test_data['y_norm'][test_mask]
            y_test_raw_cond  = test_data['y_raw'][test_mask]
        else:
            t_test_cond      = np.array([])
            y_test_norm_cond = np.empty((0, 10))
            y_test_raw_cond  = np.empty((0, 10))
    else:
        t_test_cond      = np.array([])
        y_test_norm_cond = np.empty((0, 10))
        y_test_raw_cond  = np.empty((0, 10))

    test_times_set = set(np.round(t_test_cond, 4))

    if len(t_test_cond) > 0:
        all_times  = np.concatenate([t_train_cond, t_test_cond])
        all_y_true_norm = np.concatenate([y_train_norm_cond, y_test_norm_cond])
        all_y_true_raw  = np.concatenate([y_train_raw_cond,  y_test_raw_cond])
    else:
        all_times  = t_train_cond
        all_y_true_norm = y_train_norm_cond
        all_y_true_raw  = y_train_raw_cond

    if len(all_times) == 0:
        print(f"Warning: no data found for drug vector {target_vec}. Skipping.")
        return pd.DataFrame()

    y_pred_norm = model.predict(all_times, final_drugs, scalers, device, normalized=True)
    y_pred_raw = model.predict(all_times, final_drugs, scalers, device, normalized=False)
    
    results = []
    for t_idx, t_val in enumerate(all_times):
        split_label = 'Test' if round(float(t_val), 4) in test_times_set else 'Train'
        for species_idx, species in enumerate(SPECIES_ORDER):
            true_norm = all_y_true_norm[t_idx, species_idx]
            pred_norm = y_pred_norm[t_idx, species_idx]
            results.append({
                'Time (hrs)': t_val,
                'Species': species,
                'True Value': true_norm,
                'Predicted Value': pred_norm,
                'True Value (Raw)': all_y_true_raw[t_idx, species_idx],
                'Predicted Value (Raw)': y_pred_raw[t_idx, species_idx],
                'Error': true_norm - pred_norm,
                'Percent Error': 100 * (true_norm - pred_norm) / (true_norm + 1e-6),
                'Dataset': split_label
            })
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Saved predictions table to {save_path}")
    return df
if __name__ == "__main__":
    import os
    default_model = 'results/nature_submission/pinn_model_global.pth'
    if not os.path.exists(default_model):
        default_model = 'pinn_model_best.pth'
    if os.path.exists(default_model):
        print(f"Generating extrapolation analysis using {default_model}...")
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
