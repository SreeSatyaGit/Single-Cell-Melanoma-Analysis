import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import SPECIES_ORDER, TRAINING_DATA_RAW
from pinn_model import PINN

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

def load_drug_combinations(csv_path):
    required_cols = ['vemurafenib', 'trametinib', 'pi3k_inhibitor', 'ras_inhibitor']
    combos = pd.read_csv(csv_path)
    missing = [col for col in required_cols if col not in combos.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(missing)}. "
            f"Expected columns: {', '.join(required_cols)} (optional: name)."
        )
    return combos

def _sanitize_label(label):
    sanitized = re.sub(r'[^A-Za-z0-9_-]+', '_', str(label)).strip('_')
    return sanitized or 'combo'

def predict_combinations_to_csv(
    model,
    combos_df,
    scalers,
    t_range=(0, 48),
    n_points=200,
    device='cpu',
    normalized=True,
    output_dir='predictions',
    return_results=False
):
    os.makedirs(output_dir, exist_ok=True)
    combined_rows = []
    results_by_label = []
    for idx, row in combos_df.iterrows():
        label = row['name'] if 'name' in combos_df.columns else f"combo_{idx + 1}"
        label_safe = _sanitize_label(label)
        drugs_dict = {
            'vemurafenib': float(row['vemurafenib']),
            'trametinib': float(row['trametinib']),
            'pi3k_inhibitor': float(row['pi3k_inhibitor']),
            'ras_inhibitor': float(row['ras_inhibitor'])
        }
        results = predict_new_combination(
            model,
            drugs_dict,
            scalers,
            t_range=t_range,
            n_points=n_points,
            device=device,
            normalized=normalized
        )
        results.to_csv(os.path.join(output_dir, f"predictions_{label_safe}.csv"), index=False)
        long_results = results.melt(id_vars=['time'], var_name='species', value_name='value')
        long_results['combo'] = label
        combined_rows.append(long_results)
        results_by_label.append((label, label_safe, results))

    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, 'predictions_all_combos.csv'), index=False)

    if return_results:
        return results_by_label
    return None

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
    parser = argparse.ArgumentParser(description="Run inference for new drug combinations.")
    parser.add_argument('--model', default='pinn_model_best.pth', help='Path to trained model checkpoint.')
    parser.add_argument('--combos-csv', default=None, help='CSV file with drug combinations and dosages.')
    parser.add_argument('--output-dir', default='predictions', help='Directory to save prediction CSVs.')
    parser.add_argument('--t-min', type=float, default=0.0, help='Minimum time for prediction range.')
    parser.add_argument('--t-max', type=float, default=48.0, help='Maximum time for prediction range.')
    parser.add_argument('--n-points', type=int, default=200, help='Number of time points to evaluate.')
    parser.add_argument('--unnormalized', action='store_true', help='Output predictions in raw A.U. scale.')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plot generation for training fit.')
    parser.add_argument('--plot-comparisons', action='store_true', help='Generate comparison plots for combos CSV.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, scalers = load_pinn(args.model, device)

    if os.path.exists('training_history.csv') and not args.skip_plots:
        plot_history('training_history.csv')

    if not args.skip_plots:
        plot_training_fit(model, scalers, device)

    normalized = not args.unnormalized
    t_range = (args.t_min, args.t_max)

    if args.combos_csv:
        combos_df = load_drug_combinations(args.combos_csv)
        combo_results = predict_combinations_to_csv(
            model,
            combos_df,
            scalers,
            t_range=t_range,
            n_points=args.n_points,
            device=device,
            normalized=normalized,
            output_dir=args.output_dir,
            return_results=args.plot_comparisons and not args.skip_plots
        )
        if args.plot_comparisons and not args.skip_plots:
            train_results = predict_new_combination(
                model,
                TRAINING_DATA_RAW['drugs'],
                scalers,
                t_range=t_range,
                n_points=args.n_points,
                device=device,
                normalized=normalized
            )
            for label, label_safe, results in combo_results:
                filename = os.path.join(args.output_dir, f"comparison_{label_safe}.png")
                plot_predictions(train_results, results, filename=filename, label_new=label)
        print(f"Saved predictions for {len(combos_df)} combinations to {args.output_dir}.")
    else:
        # Backwards-compatible single predictions + plots
        vemp_pi3ki_drugs = {
            'vemurafenib': 0.5,
            'trametinib': 0.0,
            'pi3k_inhibitor': 0.3,
            'ras_inhibitor': 0.0
        }
        vem_pi3ki_results = predict_new_combination(
            model,
            vemp_pi3ki_drugs,
            scalers,
            t_range=t_range,
            n_points=args.n_points,
            device=device,
            normalized=normalized
        )
        vem_pi3ki_results.to_csv('predictions_vem_pi3ki.csv', index=False)

        vem_panras_drugs = {
            'vemurafenib': 0.5,
            'trametinib': 0.0,
            'pi3k_inhibitor': 0.0,
            'ras_inhibitor': 0.3
        }
        vem_panras_results = predict_new_combination(
            model,
            vem_panras_drugs,
            scalers,
            t_range=t_range,
            n_points=args.n_points,
            device=device,
            normalized=normalized
        )
        vem_panras_results.to_csv('predictions_vem_panras.csv', index=False)

        train_results = predict_new_combination(
            model,
            TRAINING_DATA_RAW['drugs'],
            scalers,
            t_range=t_range,
            n_points=args.n_points,
            device=device,
            normalized=normalized
        )
        if not args.skip_plots:
            plot_predictions(train_results, vem_pi3ki_results, filename='comparison_vem_pi3ki.png', label_new='Vem+PI3Ki')
            plot_predictions(train_results, vem_panras_results, filename='comparison_vem_panras.png', label_new='Vem+PanRAS')

        print("Inference and visualization complete.")
