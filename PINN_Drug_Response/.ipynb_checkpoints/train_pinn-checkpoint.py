import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os

from pinn_model import PINN
from physics_utils import compute_physics_loss, compute_conservation_loss
from data_utils import prepare_training_tensors, get_collocation_points, SignalingDataset, SPECIES_ORDER

def train_pinn(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # 1. Data Preparation with train/test split
    train_until_hour = config.get('train_until_hour', 8)
    train_data, test_data, scalers = prepare_training_tensors(train_until_hour=train_until_hour)
    
    print(f"Training samples: {len(train_data['t'])} (t={train_data['t']})")
    print(f"Test samples: {len(test_data['t'])} (t={test_data['t']})")
    
    dataset = SignalingDataset(train_data['t_norm'], train_data['drugs'], train_data['y_norm'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Prepare test tensors (if any test points exist)
    has_test_data = len(test_data['t']) > 0
    if has_test_data:
        t_test = torch.tensor(test_data['t_norm'], dtype=torch.float32).view(-1, 1).to(device)
        drugs_test = torch.tensor(test_data['drugs'], dtype=torch.float32).to(device)
        y_test = torch.tensor(test_data['y_norm'], dtype=torch.float32).to(device)
    
    # Move scalers to device
    scalers_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in scalers.items()}
    
    # 2. Model and ODE Parameters
    model = PINN(input_size=5, hidden_size=config['hidden_size'], output_size=11).to(device)
    
    # Learnable ODE constants (aligned with compute_physics_loss)
    param_defaults = {
        'Km': 0.5,
        'IC50': 0.5,
        'hill_coeff': 2.0,
        'IC50_vem': 0.8,
        'k_vem_paradox': 0.25,
        'vem_optimal': 0.3,
        'k_dusp_synth': 0.8,
        'k_dusp_deg': 0.5,
        'Km_dusp': 0.4,
        'n_dusp': 2.5,
        'k_dusp_cat': 0.6,
        'k_erk_sos': 0.4,
        'k_mek_inhib': 0.2,
        'k_s6k_irs': 0.7,
        'Km_s6k': 0.5,
        'k_s6k_mtor': 0.3,
        'k_4ebp1_comp': 0.25,
        'k_akt_rtk': 0.15,
        'k_akt_raf': 0.5,
        'k_erk_pi3k': 0.45,
        'k_raf_pi3k': 0.2,
        'k_akt_mek': 0.18,
        'Km_akt_mek': 1.2,
        'k_craf_act': 1.2,
        'k_craf_deg': 0.35,
        'k_mek_act': 1.0,
        'k_mek_deg': 0.4,
        'k_erk_act': 1.2,
        'k_erk_deg': 0.45,
        'k_akt_act': 1.0,
        'k_akt_deg': 0.4,
        'k_s6k_act': 0.9,
        'k_s6k_deg': 0.5,
        'k_4ebp1_act': 0.85,
        'k_4ebp1_deg': 0.45,
        'k_erk_rtk': 0.1,
        'Km_erk_rtk': 0.5,
        'k_egfr_phos': 0.5,
        'k_egfr_dephos': 0.2,
        'k_her_phos': 0.4,
        'k_her_dephos': 0.15,
        'k_igf_phos': 0.3,
        'k_igf_dephos': 0.2
    }
    k_params = nn.ParameterDict({
        name: nn.Parameter(torch.tensor(value, device=device)) for name, value in param_defaults.items()
    })
    
    # 3. Optimizer and Scheduler
    optimizer = optim.Adam(
        list(model.parameters()) + list(k_params.parameters()), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=config['lr_decay'])
    
    # 4. Losses
    mse_loss = nn.MSELoss()
    history = []
    
    # Collocation points for physics loss (only in training time range)
    t_physics_raw, drugs_physics_raw = get_collocation_points(config['num_physics_points'])
    t_physics = t_physics_raw.to(device)
    drugs_physics = drugs_physics_raw.to(device)
    
    # 5. Training Loop
    best_loss = float('inf')
    
    progress_bar = tqdm(range(config['num_epochs']))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # --- (a) Data Loss (Training Set) ---
        t_data, drugs_data_batch, y_exp = next(iter(data_loader))
        t_data, drugs_data_batch, y_exp = t_data.to(device), drugs_data_batch.to(device), y_exp.to(device)
        
        y_pred = model(t_data, drugs_data_batch)
        l_data = mse_loss(y_pred, y_exp)
        
        # --- (b) Physics Loss ---
        l_physics = compute_physics_loss(model, t_physics, drugs_physics, k_params, scalers_device)
        
        # --- (c) Boundary Loss (t=0) ---
        t0 = torch.zeros((1, 1), device=device)
        drugs0 = drugs_data_batch[0:1]
        y0_pred = model(t0, drugs0)
        y0_exp = y_exp[0:1]
        l_boundary = mse_loss(y0_pred, y0_exp)
        
        # --- (d) Conservation/Constraint Loss ---
        l_conservation = compute_conservation_loss(y_pred, scalers_device)
        
        # Total Weighted Loss
        total_loss = (config['weights']['data'] * l_data + 
                      config['weights']['physics'] * l_physics + 
                      config['weights']['boundary'] * l_boundary + 
                      config['weights']['conservation'] * l_conservation)
        
        # Backward Pass
        total_loss.backward()
        optimizer.step()
        optimizer.step()
        scheduler.step()
        
        # Resample physics points every epoch for better coverage of the high-dimensional drug space
        t_physics_raw, drugs_physics_raw = get_collocation_points(config['num_physics_points'])
        t_physics = t_physics_raw.to(device)
        drugs_physics = drugs_physics_raw.to(device)
        
        # Evaluate on test set (without gradients)
        model.eval()
        l_test = torch.tensor(float('nan'), device=device)
        if has_test_data:
            with torch.no_grad():
                y_test_pred = model(t_test, drugs_test)
                l_test = mse_loss(y_test_pred, y_test)
        
        # Logging
        if epoch % 50 == 0:
            history.append({
                'epoch': epoch,
                'loss': total_loss.item(),
                'l_data': l_data.item(),
                'l_physics': l_physics.item(),
                'l_boundary': l_boundary.item(),
                'l_cons': l_conservation.item(),
                'l_test': l_test.item()
            })
            progress_bar.set_postfix({
                'train': f"{l_data.item():.4e}", 
                'test': f"{l_test.item():.4e}" if has_test_data else "n/a"
            })
            
        # Checkpoints/Early Stopping
        # Checkpoints/Early Stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            
            # Save best model
            save_path = os.path.join(os.getcwd(), 'pinn_model_best.pth')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'k_params_state_dict': k_params.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': total_loss.item(),
                    'scalers': scalers,
                    'train_data': train_data,
                    'test_data': test_data
                }, save_path)
            except Exception as e:
                print(f"Warning: Could not save model to {save_path}: {e}")

        if epoch % 500 == 0 and epoch > 0:
            ckpt_path = os.path.join(os.getcwd(), f'checkpoint_{epoch}.pth')
            try:
                torch.save({'model_state_dict': model.state_dict(), 'scalers': scalers}, ckpt_path)
            except Exception as e:
                print(f"Warning: Could not save checkpoint to {ckpt_path}: {e}")

    # Save final artifacts
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Training complete. Best loss: {best_loss:.4e}")
    return model, k_params, history, scalers, train_data, test_data

if __name__ == "__main__":
    config = {
        'train_until_hour': 48,
        'num_epochs': 10000,
        'learning_rate': 0.0005,
        'lr_decay': 0.98,
        'batch_size': 4,
        'hidden_size': 128,
        'num_physics_points': 200,
        'weight_decay': 1e-4,
        'weights': {
            'data': 1.0,
            'physics': 5.0,      # MUCH higher - critical for extrapolation
            'boundary': 1.0,
            'conservation': 0.5
        }
    }
    
    train_pinn(config)
