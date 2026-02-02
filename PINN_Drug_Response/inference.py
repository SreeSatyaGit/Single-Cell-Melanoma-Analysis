import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pinn_model import PINN
from data_utils import SPECIES_ORDER, TRAINING_DATA_RAW
from physics_utils import compute_physics_loss
import seaborn as sns

def load_pinn(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    # Auto-detect hidden_size from checkpoint
    state_dict = checkpoint['model_state_dict']
    hidden_size = state_dict['input_layer.weight'].shape[0]
    model = PINN(input_size=5, hidden_size=hidden_size, output_size=11).to(device)
    model.load_state_dict(state_dict)
    scalers = checkpoint['scalers']
    return model, scalers

def predict_new_combination(model, drugs_dict, scalers, t_range=(0, 48), n_points=200, device='cpu', normalized=True):
    """
    Generate predictions for a new drug combination.
    """
    t_eval = np.linspace(t_range[0], t_range[1], n_points)
    y_pred = model.predict(t_eval, drugs_dict, scalers, device, normalized=normalized)
    
    # Store in DataFrame
    results = pd.DataFrame(y_pred, columns=SPECIES_ORDER)
    results['time'] = t_eval
    return results

def plot_training_fit(model, scalers, device='cpu'):
    """
    Plots model predictions vs experimental training data.
    """
    t_points = TRAINING_DATA_RAW['time_points']
    drugs_train = TRAINING_DATA_RAW['drugs']
    
    # Generate smooth predictions for plotting
    t_smooth = np.linspace(0, 48, 200)
    y_smooth = model.predict(t_smooth, drugs_train, scalers, device)
    
    # Get predictions at experimental time points for R2
    y_at_exp = model.predict(t_points, drugs_train, scalers, device)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    # Prepare experimental data (normalize it for plotting)
    y_exp_norm = {}
    for species in SPECIES_ORDER:
        raw_vals = TRAINING_DATA_RAW['species'][species]
        # Match data_utils normalization: divide by global max
        y_exp_norm[species] = raw_vals / (np.max(raw_vals) + 1e-8)

    for i, species in enumerate(SPECIES_ORDER):
        ax = axes[i]
        exp_vals = y_exp_norm[species]
        
        # Plot experimental points
        ax.scatter(t_points, exp_vals, color='red', label='Exp Data', zorder=5)
        
        # Plot model line (normalized by default)
        ax.plot(t_smooth, y_smooth[:, i], color='blue', label='PINN Fit')
        
        # Compute R2
        ss_res = np.sum((exp_vals - y_at_exp[:, i])**2)
        ss_tot = np.sum((exp_vals - np.mean(exp_vals))**2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        ax.set_title(f"{species} (R²={r2:.3f})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel("Normalized Intensity")
        ax.grid(alpha=0.3)
        if i == 0: ax.legend()

    # Hide extra subplot
    if len(axes) > len(SPECIES_ORDER):
        axes[-1].axis('off')
        
    plt.tight_layout()
    plt.savefig('model_fit.png', dpi=300)
    plt.close()

def plot_predictions(train_preds, new_preds, species_to_plot=None):
    """
    Plots comparison between training condition and new drug condition.
    """
    if species_to_plot is None:
        species_to_plot = SPECIES_ORDER
        
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, species in enumerate(species_to_plot):
        ax = axes[i]
        ax.plot(train_preds['time'], train_preds[species], '--', color='gray', label='Vem+Tram (Train)')
        ax.plot(new_preds['time'], new_preds[species], '-', color='green', label='Vem+PI3Ki (New)')
        
        ax.set_title(species)
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.2)
        if i == 0: ax.legend()

    if len(axes) > len(species_to_plot):
        axes[-1].axis('off')
        
    plt.suptitle("Comparison: Training (Vem+Tram) vs Prediction (Vem+PI3Ki)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('prediction_comparison.png', dpi=300)
    plt.close()

def plot_history(history_file):
    history = pd.read_csv(history_file)
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['loss'], label='Total Loss')
    plt.plot(history['epoch'], history['l_data'], label='Data Loss')
    plt.plot(history['epoch'], history['l_physics'], label='Physics Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers = load_pinn('pinn_model_best.pth', device)
    
    # 1. Plot Training History
    if os.path.exists('training_history.csv'):
        plot_history('training_history.csv')
    
    # 2. Plot Training Fit
    plot_training_fit(model, scalers, device)
    
    # 3. Predict for new combination
    prediction_drugs = {
        'vemurafenib': 0.5,
        'trametinib': 0.0,
        'pi3k_inhibitor': 0.3,
        'ras_inhibitor': 0.0
    }
    
    new_results = predict_new_combination(model, prediction_drugs, scalers, device=device)
    new_results.to_csv('predictions_pi3ki_vem.csv', index=False)
    
    # Generate Training predictions for comparison
    train_results = predict_new_combination(model, TRAINING_DATA_RAW['drugs'], scalers, device=device)
    
    # 4. Plot Comparisons
    plot_predictions(train_results, new_results)
    
    print("Inference and visualization complete.")
