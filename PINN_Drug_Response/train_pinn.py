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
    
    # 1. Data Preparation
    train_until_hour = config.get('train_until_hour', 8)
    train_data, test_data, scalers = prepare_training_tensors(train_until_hour=train_until_hour)
    
    print(f"Aggregated Training points: {len(train_data['t'])}")
    has_test_data = len(test_data['t']) > 0
    
    dataset = SignalingDataset(train_data['t_norm'], train_data['drugs'], train_data['y_norm'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    if has_test_data:
        t_test = torch.tensor(test_data['t_norm'], dtype=torch.float32).to(device)
        drugs_test = torch.tensor(test_data['drugs'], dtype=torch.float32).to(device)
        y_test = torch.tensor(test_data['y_norm'], dtype=torch.float32).to(device)
    
    scalers_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in scalers.items()}
    
    # 2. Model and ODE Parameters
    model = PINN(hidden_size=config['hidden_size'], num_hidden=6).to(device)
    
    param_defaults = {
        'Km': 0.5, 'IC50': 0.5, 'hill_coeff': 2.5, 'IC50_vem': 0.8,
        'k_vem_paradox': 0.25, 'vem_optimal': 0.3, 'k_dusp_synth': 0.8,
        'k_dusp_deg': 0.5, 'Km_dusp': 0.4, 'n_dusp': 2.5, 'k_dusp_cat': 0.6,
        'k_erk_sos': 0.4, 'k_mek_inhib': 0.2, 'k_s6k_irs': 0.7, 'Km_s6k': 0.5,
        'k_s6k_mtor': 0.3, 'k_4ebp1_comp': 0.25, 'k_akt_rtk': 0.15, 'k_akt_raf': 0.5,
        'k_erk_pi3k': 0.45, 'k_raf_pi3k': 0.2, 'k_akt_mek': 0.18, 'Km_akt_mek': 1.2,
        'k_craf_act': 1.2, 'k_craf_deg': 0.35, 'k_mek_act': 1.0, 'k_mek_deg': 0.4,
        'k_erk_act': 1.2, 'k_erk_deg': 0.45, 'k_akt_act': 1.0, 'k_akt_deg': 0.4,
        'k_s6k_act': 0.9, 'k_s6k_deg': 0.5, 'k_4ebp1_act': 0.85, 'k_4ebp1_deg': 0.45,
        'k_erk_rtk': 0.1, 'Km_erk_rtk': 0.5, 'k_egfr_phos': 0.5, 'k_egfr_dephos': 0.2,
        'k_her_phos': 0.4, 'k_her_dephos': 0.15, 'k_igf_phos': 0.3, 'k_igf_dephos': 0.2
    }
    k_params = nn.ParameterDict({
        name: nn.Parameter(torch.tensor(value, device=device)) for name, value in param_defaults.items()
    })
    
    # 3. Optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(k_params.parameters()), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=config['lr_decay'])
    
    mse_loss = nn.MSELoss()
    history = []
    best_loss = float('inf')
    
    # 4. Training Loop
    progress_bar = tqdm(range(config['num_epochs']))
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # Resample physics points
        t_physics_raw, drugs_physics_raw = get_collocation_points(config['num_physics_points'])
        t_physics = t_physics_raw.to(device)
        drugs_physics = drugs_physics_raw.to(device)
        
        # --- (a) Data Loss with Perturbation Jittering ---
        t_data, drugs_data, y_exp = next(iter(data_loader))
        t_data, drugs_data, y_exp = t_data.to(device), drugs_data.to(device), y_exp.to(device)
        
        # Add tiny jitter to drugs to prevent memorization (Generalization Trick)
        jitter = (torch.rand_like(drugs_data) - 0.5) * 0.02 
        y_pred = model(t_data, drugs_data + jitter)
        l_data = mse_loss(y_pred, y_exp)
        
        # --- (b) Physics Loss with Weight Annealing ---
        # Gradually increase physics pressure as training progresses
        phys_weight_multiplier = min(1.0, (epoch + 1) / (config['num_epochs'] * 0.5))
        current_phys_weight = config['weights']['physics'] * phys_weight_multiplier
        l_physics = compute_physics_loss(model, t_physics, drugs_physics, k_params, scalers_device)
        
        # --- (c) Boundary Loss ---
        t0 = torch.zeros((1, 1), device=device)
        idx0 = (t_data == 0).squeeze()
        if idx0.any():
            y0_pred = model(t_data[idx0], drugs_data[idx0])
            l_boundary = mse_loss(y0_pred, y_exp[idx0])
        else:
            l_boundary = torch.tensor(0.0, device=device)
        
        # --- (d) Conservation & Parameter Sparsity ---
        l_conservation = compute_conservation_loss(y_pred, scalers_device)
        
        # L1 Regularization on k_params (Favors biologically simple models)
        l_sparsity = sum(torch.abs(p) for p in k_params.parameters())
        
        # Total Weighted Loss
        total_loss = (config['weights']['data'] * l_data + 
                      current_phys_weight * l_physics + 
                      config['weights']['boundary'] * l_boundary + 
                      config['weights']['conservation'] * l_conservation +
                      0.001 * l_sparsity) # Small sparsity penalty
        
        if torch.isnan(total_loss):
            print(f"\nFATAL: NaN detected at epoch {epoch}. Terminating.")
            break
            
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                l_test = mse_loss(model(t_test, drugs_test), y_test).item() if has_test_data else 0.0
            
            history.append({
                'epoch': epoch, 'loss': total_loss.item(), 'l_data': l_data.item(),
                'l_physics': l_physics.item(), 'l_boundary': l_boundary.item(), 'l_test': l_test
            })
            progress_bar.set_postfix({'loss': f"{total_loss.item():.2e}", 'data': f"{l_data.item():.2e}"})
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'k_params_state_dict': k_params.state_dict(),
                    'scalers': scalers,
                    'train_data': train_data,
                    'test_data': test_data
                }, 'pinn_model_best.pth')

    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    return model, k_params, history, scalers, train_data, test_data

if __name__ == "__main__":
    config = {
        'train_until_hour': 48,
        'num_epochs': 20000,
        'learning_rate': 0.0001, # Stabilized
        'lr_decay': 0.9,
        'hidden_size': 256,
        'num_physics_points': 500,
        'weight_decay': 1e-5,
        'weights': {
            'data': 10.0,      # Prioritize matching real data
            'physics': 1.0,    # Smooth with physics
            'boundary': 50.0,  # Anchor to t=0
            'conservation': 0.1
        }
    }
    train_pinn(config)
