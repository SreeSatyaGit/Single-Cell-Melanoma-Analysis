import torch
import os
from train_pinn import train_pinn
from visualize_extrapolation import plot_extrapolation_results, plot_training_history, generate_prediction_table

def main():
    # 1. Configuration - OPTIMIZED FOR FULL DATA TRAINING
    config = {
        'train_until_hour': 48,    # Train on the full 0-48hr range
        'num_epochs': 10000,       # More epochs for convergence
        'learning_rate': 0.0005,   # Lower LR for stability
        'lr_decay': 0.98,          # Slower decay
        'batch_size': 6,           # 6 training points (full dataset)
        'hidden_size': 128,        # Wider network
        'num_physics_points': 200, # More physics constraints
        'weight_decay': 1e-4,      # Stronger regularization
        'weights': {
            'data': 1.0,
            'physics': 5.0,        # CRITICAL: High physics weight for extrapolation
            'boundary': 1.0,
            'conservation': 0.5
        }
    }
    
    # 2. Train the model
    print("="*60)
    print("TRAINING: Using Vem+Tram data at [0,1,4,8,24,48]hrs")
    print("TESTING: No held-out time points (full data training)")
    print("="*60)
    
    model, k_params, history, scalers, train_data, test_data = train_pinn(config)
    
    # 3. Generate visualizations
    print("\n" + "="*60)
    print("GENERATING EXTRAPOLATION ANALYSIS")
    print("="*60)
    
    if os.path.exists('pinn_model_best.pth'):
        plot_extrapolation_results()
        plot_training_history()
        df = generate_prediction_table()
        
        print("\n✓ Results saved:")
        print("  - extrapolation_results.png")
        print("  - training_test_history.png")
        print("  - predictions_table.csv")
        print("  - training_history.csv")
        print("  - training_config.json")
    else:
        print("Error: Model checkpoint not found.")

if __name__ == "__main__":
    main()
