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
    train_data, test_data, scalers = prepare_training_tensors(train_until_hour=8)
    
    print(f"Training samples: {len(train_data['t'])} (t={train_data['t']})")
    print(f"Test samples: {len(test_data['t'])} (t={test_data['t']})")
    
    dataset = SignalingDataset(train_data['t_norm'], train_data['drugs'], train_data['y_norm'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Prepare test tensors
    t_test = torch.tensor(test_data['t_norm'], dtype=torch.float32).view(-1, 1).to(device)
    drugs_test = torch.tensor(test_data['drugs'], dtype=torch.float32).to(device)
    y_test = torch.tensor(test_data['y_norm'], dtype=torch.float32).to(device)
    
    # Move scalers to device
    scalers_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in scalers.items()}
    
    # 2. Model and ODE Parameters
    model = PINN(input_size=5, hidden_size=config['hidden_size'], output_size=11).to(device)
    
    # Learnable ODE constants
    k_names = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k_cat']
    k_params = nn.ParameterDict({
        name: nn.Parameter(torch.tensor(0.5, device=device)) for name in k_names
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
        scheduler.step()
        
        # Evaluate on test set (without gradients)
        model.eval()
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
                'test': f"{l_test.item():.4e}"
            })
            
        # Checkpoints/Early Stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'k_params_state_dict': k_params.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss.item(),
                'scalers': scalers,
                'train_data': train_data,
                'test_data': test_data
            }, 'pinn_model_best.pth')

        if epoch % 500 == 0 and epoch > 0:
            torch.save({'model_state_dict': model.state_dict(), 'scalers': scalers}, f'checkpoint_{epoch}.pth')

    # Save final artifacts
    pd.DataFrame(history).to_csv('training_history.csv', index=False)
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Training complete. Best loss: {best_loss:.4e}")
    return model, k_params, history, scalers, train_data, test_data

if __name__ == "__main__":
    config = {
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

