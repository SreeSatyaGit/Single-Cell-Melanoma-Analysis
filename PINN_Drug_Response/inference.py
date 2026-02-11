import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pinn_model import PINN
from data_utils import SPECIES_ORDER, TRAINING_DATA_LIST
from physics_utils import compute_physics_loss
import seaborn as sns
import os


class LegacyBlock(nn.Module):
    def __init__(self, hidden_size):
        super(LegacyBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        return x + self.fc2(self.activation(self.fc1(x)))

class LegacyPINN(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, output_size=11, num_hidden=4):
        super(LegacyPINN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.res_blocks = nn.ModuleList([LegacyBlock(hidden_size) for _ in range(num_hidden)])
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        
    def forward(self, t, drugs):
        x = torch.cat([t, drugs], dim=1)
        x = torch.tanh(self.input_layer(x))
        for block in self.res_blocks:
            x = block(x)
        return self.softplus(self.output_layer(x))
    
    def predict(self, t_np, drugs_dict, scalers, device='cpu', normalized=True):
        # Re-use the predict logic from PINN (assumes same scalers structure)
        return PINN.predict(self, t_np, drugs_dict, scalers, device, normalized)

def load_pinn(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    
    # 1. Try initializing the current model
    try:
        # Detect hidden size if possible from res_blocks.0.fc1
        if 'res_blocks.0.fc1.weight' in state_dict:
             hidden_size = state_dict['res_blocks.0.fc1.weight'].shape[0]
        else:
             hidden_size = 256 # Default fallback
             
        model = PINN(input_size=5, hidden_size=hidden_size, output_size=11).to(device)
        model.load_state_dict(state_dict)
        print("Loaded modern PINN architecture.")
        
    except RuntimeError as e:
        # 2. Check for legacy architecture (input_layer)
        if 'input_layer.weight' in state_dict:
            print("Warning: Detected legacy model architecture. Loading as LegacyPINN...")
            hidden_size = state_dict['input_layer.weight'].shape[0]
            
            # Detect depth
            depth = 0
            while f'res_blocks.{depth}.fc1.weight' in state_dict:
                depth += 1
            
            print(f"Legacy model depth: {depth} blocks, hidden: {hidden_size}")
            model = LegacyPINN(input_size=5, hidden_size=hidden_size, output_size=11, num_hidden=depth).to(device)
            model.load_state_dict(state_dict)
        else:
            raise e
            
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
    t_points = TRAINING_DATA_LIST[0]['time_points']
    drugs_train = TRAINING_DATA_LIST[0]['drugs']
    
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
        raw_vals = TRAINING_DATA_LIST[0]['species'][species]
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

def plot_predictions(train_preds, new_preds, species_to_plot=None, filename='prediction_comparison.png', label_new='New Condition'):
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
        ax.plot(new_preds['time'], new_preds[species], '-', color='green', label=label_new)
        
        ax.set_title(species)
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.2)
        if i == 0: ax.legend()

    if len(axes) > len(species_to_plot):
        axes[-1].axis('off')
        
    plt.suptitle(f"Comparison: Training vs {label_new}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename, dpi=300)
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
    import argparse
    import sys

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PINN Drug Response Inference')
    parser.add_argument('--model', type=str, default='results/nature_submission/pinn_model_global.pth', help='Path to trained model checkpoint')
    parser.add_argument('--vemurafenib', type=float, help='Vemurafenib concentration (uM)')
    parser.add_argument('--trametinib', type=float, help='Trametinib concentration (uM)')
    parser.add_argument('--pi3k', type=float, help='PI3K Inhibitor concentration (uM)')
    parser.add_argument('--ras', type=float, help='RAS Inhibitor concentration (uM)')
    parser.add_argument('--output', type=str, default='prediction_custom.png', help='Output plot filename')
    parser.add_argument('--csv', type=str, default='prediction_custom.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if os.path.exists(args.model):
        print(f"Loading model from {args.model}...")
        model, scalers = load_pinn(args.model, device)
    else:
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)

    # Check if custom dosages are provided
    if args.vemurafenib is not None or args.trametinib is not None or args.pi3k is not None or args.ras is not None:
        # Set defaults to 0.0 if not provided
        vem = args.vemurafenib if args.vemurafenib is not None else 0.0
        tram = args.trametinib if args.trametinib is not None else 0.0
        pi3k = args.pi3k if args.pi3k is not None else 0.0
        ras = args.ras if args.ras is not None else 0.0
        
        print(f"Predicting for combination: Vem={vem}, Tram={tram}, PI3K={pi3k}, RAS={ras}")
        
        drugs_custom = {
            'vemurafenib': vem,
            'trametinib': tram,
            'pi3k_inhibitor': pi3k,
            'ras_inhibitor': ras
        }
        
        # Predict
        results = predict_new_combination(model, drugs_custom, scalers, device=device)
        
        # Save CSV
        results.to_csv(args.csv, index=False)
        print(f"Saved predictions to {args.csv}")
        
        # Plot
        # We can reuse plot_predictions but we need a baseline. 
        # Or just plot the single prediction.
        # Let's create a simple single plot function inline or use existing.
        
        plt.figure(figsize=(15, 12))
        for i, species in enumerate(SPECIES_ORDER):
            plt.subplot(4, 3, i+1)
            plt.plot(results['time'], results[species], 'g-', linewidth=2, label='Prediction')
            plt.title(species)
            plt.xlabel('Time (h)')
            plt.ylabel('Concentration')
            plt.grid(alpha=0.3)
            plt.ylim(-0.1, 1.1)
            if i == 0: plt.legend()
            
        plt.suptitle(f"Predicted Response: Vem={vem}, Tram={tram}, PI3K={pi3k}, RAS={ras}")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(args.output, dpi=300)
        plt.close()
        print(f"Saved plot to {args.output}")
        
    else:
        # Default behavior (Demo)
        print("No custom dosages provided. Running default demo...")
        
        # 1. Plot Training History
        history_file = 'training_history.csv'
        if 'pinn_model_global.pth' in args.model:
             history_file = args.model.replace('pinn_model_global.pth', 'history_global.csv')

        if os.path.exists(history_file):
            plot_history(history_file)
        
        # 2. Plot Training Fit
        plot_training_fit(model, scalers, device)
        
        # 3. Predict for new combination: Vem + PI3Ki
        vemp_pi3ki_drugs = {
            'vemurafenib': 0.5,
            'trametinib': 0.0,
            'pi3k_inhibitor': 0.3, # Assuming typical dose
            'ras_inhibitor': 0.0
        }
        
        print("Predicting Vem (0.5) + PI3Ki (0.3)...")
        vem_pi3ki_results = predict_new_combination(model, vemp_pi3ki_drugs, scalers, device=device)
        vem_pi3ki_results.to_csv('predictions_vem_pi3ki.csv', index=False)
        
        # 4. Predict for new combination: Vem + PanRAS
        vem_panras_drugs = {
            'vemurafenib': 0.5,
            'trametinib': 0.0,
            'pi3k_inhibitor': 0.0,
            'ras_inhibitor': 0.3   # Assuming typical dose
        }
        
        print("Predicting Vem (0.5) + RASi (0.3)...")
        vem_panras_results = predict_new_combination(model, vem_panras_drugs, scalers, device=device)
        vem_panras_results.to_csv('predictions_vem_panras.csv', index=False)
    
        # Generate Training predictions for comparison
        train_results = predict_new_combination(model, TRAINING_DATA_LIST[0]['drugs'], scalers, device=device)
        
        # 5. Plot Comparisons
        plot_predictions(train_results, vem_pi3ki_results, filename='comparison_vem_pi3ki.png', label_new='Vem+PI3Ki')
        plot_predictions(train_results, vem_panras_results, filename='comparison_vem_panras.png', label_new='Vem+PanRAS')
        
        print("Inference and visualization complete.")
