import torch
import os
from train_pinn import train_pinn
from inference import load_pinn, plot_training_fit, predict_new_combination, plot_predictions, plot_history
from data_utils import TRAINING_DATA_RAW

def main():
    # 1. Configuration
    config = {
        'num_epochs': 5000,
        'learning_rate': 0.001,
        'lr_decay': 0.95,
        'batch_size': 6,
        'hidden_size': 100,
        'num_physics_points': 100,
        'weight_decay': 1e-5,
        'weights': {
            'data': 1.0,
            'physics': 0.5,
            'boundary': 0.3,
            'conservation': 0.2
        }
    }
    
    # 2. Train the model
    print("Starting PINN training...")
    model, k_params, history, scalers = train_pinn(config)
    
    # 3. Visualization and Inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pinn_model_best.pth'
    
    if os.path.exists(model_path):
        print("Generating visualizations and predictions...")
        model, scalers = load_pinn(model_path, device)
        
        # Training history
        plot_history('training_history.csv')
        
        # Training data fit
        plot_training_fit(model, scalers, device)
        
        # New drug combination prediction
        prediction_drugs = {
            'vemurafenib': 0.5,
            'trametinib': 0.0,
            'pi3k_inhibitor': 0.3,
            'ras_inhibitor': 0.0
        }
        new_results = predict_new_combination(model, prediction_drugs, scalers, device=device)
        new_results.to_csv('predictions_pi3ki_vem.csv', index=False)
        
        # Training condition predictions (for comparison)
        train_results = predict_new_combination(model, TRAINING_DATA_RAW['drugs'], scalers, device=device)
        
        # Final comparison plot
        plot_predictions(train_results, new_results)
        
        print(f"Results saved to {os.getcwd()}")
    else:
        print("Error: Model checkpoint not found.")

if __name__ == "__main__":
    main()
