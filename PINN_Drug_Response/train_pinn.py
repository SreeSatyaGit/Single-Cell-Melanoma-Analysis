import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict, Any, List, Optional

from pinn_model import PINN
from physics_utils import compute_physics_loss, compute_conservation_loss
from data_utils import prepare_training_tensors, get_collocation_points, SignalingDataset
from config import TrainingConfig
from utils import setup_logger, save_checkpoint, count_parameters

def train_pinn(config: TrainingConfig, condition_name: Optional[str] = None) -> Tuple[nn.Module, nn.ParameterDict, List[Dict], Dict, Dict, Dict]:
    """
    Trains the Physics-Informed Neural Network with the specified configuration.
    
    Args:
        config (TrainingConfig): Configuration object.
        condition_name (str, optional): Specific condition to train on (if None, trains global model).
        
    Returns:
        Tuple containing trained model, kinetic params, history, scalers, train_data, test_data.
    """
    # Setup Logger
    log_file = f"{config.output_dir}/training_{condition_name if condition_name else 'global'}.log"
    logger = setup_logger(log_file=log_file)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_str = f"Condition: {condition_name}" if condition_name else "GLOBAL (All Conditions)"
    logger.info(f"--- Training on {device} | {mode_str} ---")
    
    # 1. Data Preparation
    train_data, test_data, scalers = prepare_training_tensors(
        train_until_hour=config.train_until_hour, 
        condition_name=condition_name
    )
    
    logger.info(f"Total Training points: {len(train_data['t'])}")
    has_test_data = len(test_data['t']) > 0
    
    dataset = SignalingDataset(train_data['t_norm'], train_data['drugs'], train_data['y_norm'])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    # Move test data to device
    t_test = torch.tensor(test_data['t_norm'], dtype=torch.float32).to(device) if has_test_data else None
    drugs_test = torch.tensor(test_data['drugs'], dtype=torch.float32).to(device) if has_test_data else None
    y_test = torch.tensor(test_data['y_norm'], dtype=torch.float32).to(device) if has_test_data else None
    
    scalers_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in scalers.items()}
    
    # 2. Model Initialization
    model = PINN(
        input_size=config.model.input_size, 
        hidden_size=config.model.hidden_size, 
        output_size=config.model.output_size,
        num_hidden=config.model.num_hidden_layers
    ).to(device)
    
    logger.info(f"Model initialized with {count_parameters(model)} parameters.")
    
    # 3. Kinetic Parameter Initialization (Physically Motivated)
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
    
    # 4. Optimization Setup
    optimizer = optim.Adam(
        list(model.parameters()) + list(k_params.parameters()), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config.lr_decay_step, 
        gamma=config.lr_decay_gamma
    )
    
    mse_loss = nn.MSELoss()
    history = []
    best_loss = float('inf')
    
    # 5. Training Loop
    progress_bar = tqdm(range(config.num_epochs), desc="Training")
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # Resample physics points
        t_physics_raw, drugs_physics_raw = get_collocation_points(config.num_physics_points)
        t_physics = t_physics_raw.to(device)
        drugs_physics = drugs_physics_raw.to(device)
        
        # --- (a) Data Loss ---
        t_data_batch, drugs_data_batch, y_exp_batch = next(iter(data_loader))
        t_data_batch = t_data_batch.to(device)
        drugs_data_batch = drugs_data_batch.to(device)
        y_exp_batch = y_exp_batch.to(device)
        
        # Data Jitter for Generalization
        jitter = (torch.rand_like(drugs_data_batch) - 0.5) * 0.02 
        y_pred = model(t_data_batch, drugs_data_batch + jitter)
        l_data = mse_loss(y_pred, y_exp_batch)
        
        # --- (b) Physics Loss (Annealed) ---
        phys_weight_multiplier = min(1.0, (epoch + 1) / (config.num_epochs * 0.5))
        current_phys_weight = config.weights.physics * phys_weight_multiplier
        l_physics = compute_physics_loss(model, t_physics, drugs_physics, k_params, scalers_device)
        
        # --- (c) Boundary Loss ---
        idx0 = (t_data_batch == 0).squeeze()
        if idx0.any():
            y0_pred = model(t_data_batch[idx0], drugs_data_batch[idx0])
            l_boundary = mse_loss(y0_pred, y_exp_batch[idx0])
        else:
            l_boundary = torch.tensor(0.0, device=device)
        
        # --- (d) Regularization ---
        l_conservation = compute_conservation_loss(y_pred, scalers_device)
        l_sparsity = sum(torch.abs(p) for p in k_params.parameters())
        
        # Total Loss Composition
        total_loss = (config.weights.data * l_data + 
                      current_phys_weight * l_physics + 
                      config.weights.boundary * l_boundary + 
                      config.weights.conservation * l_conservation +
                      config.weights.sparsity * l_sparsity)
        
        if torch.isnan(total_loss):
            logger.error(f"NaN detected at epoch {epoch}. Terminating.")
            break
            
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Monitoring
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                l_test = mse_loss(model(t_test, drugs_test), y_test).item() if has_test_data and t_test is not None else 0.0
            
            history.append({
                'epoch': epoch, 
                'loss': total_loss.item(), 
                'l_data': l_data.item(),
                'l_physics': l_physics.item(), 
                'l_boundary': l_boundary.item(), 
                'l_test': l_test
            })
            progress_bar.set_postfix({'loss': f"{total_loss.item():.2e}", 'data': f"{l_data.item():.2e}"})
            
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                save_name = f"{config.output_dir}/pinn_model_{condition_name.replace(' ', '_') if condition_name else 'global'}.pth"
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'k_params_state_dict': k_params.state_dict(),
                    'scalers': scalers,
                    'train_data': train_data,
                    'test_data': test_data,
                    'condition_name': 'global' if condition_name is None else condition_name,
                    'config': config.to_dict()
                }
                save_checkpoint(checkpoint_data, save_name, logger=None) # Logger none to avoid spam

    # Save History
    hist_name = f"{config.output_dir}/history_{condition_name.replace(' ', '_') if condition_name else 'global'}.csv"
    pd.DataFrame(history).to_csv(hist_name, index=False)
    logger.info("Training complete.")
    
    return model, k_params, history, scalers, train_data, test_data

if __name__ == "__main__":
    from config import TrainingConfig
    from utils import set_seed
    
    set_seed(42)
    conf = TrainingConfig()
    train_pinn(conf)
