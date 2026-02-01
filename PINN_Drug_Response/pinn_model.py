import torch
import torch.nn as nn
import torch.nn.functional as F

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Cancer Signaling Pathways.
    
    Inputs:
        - Time (normalized)
        - Vemurafenib concentration
        - Trametinib concentration
        - PI3K inhibitor concentration
        - RAS inhibitor concentration
        
    Outputs:
        - 11 Signaling species (pEGFR, HER2, HER3, IGF1R, pCRAF, pMEK, pERK, DUSP6, pAKT, pS6K, p4EBP1)
    """
    def __init__(self, input_size=5, hidden_size=100, output_size=11, num_hidden=4):
        super(PINN, self).__init__()
        
        layers = []
        # First layer: input -> hidden
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        
        # Middle layers: hidden -> hidden
        for _ in range(num_hidden - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
            
        # Last layer: hidden -> output
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, t, drugs):
        """
        Forward pass.
        t: (batch, 1) - Normalized time
        drugs: (batch, 4) - Drug concentrations
        """
        # Concatenate time and drugs
        x = torch.cat([t, drugs], dim=1)
        out = self.network(x)
        # Apply Softplus to ensure positivity
        return self.softplus(out)

    def predict(self, t_np, drugs_dict, scalers, device='cpu'):
        """
        Utility for inference with numpy inputs.
        """
        self.eval()
        with torch.no_grad():
            # Prepare time
            t_norm = (t_np - scalers['t_min']) / (scalers['t_max'] - scalers['t_min'])
            t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
            
            # Prepare drugs
            drugs_vec = torch.tensor([
                drugs_dict['vemurafenib'],
                drugs_dict['trametinib'],
                drugs_dict['pi3k_inhibitor'],
                drugs_dict['ras_inhibitor']
            ], dtype=torch.float32).view(1, -1).to(device)
            drugs_tensor = drugs_vec.repeat(t_tensor.size(0), 1)
            
            # Forward pass
            y_pred_norm = self.forward(t_tensor, drugs_tensor)
            
            # Unnormalize
            y_pred = y_pred_norm.cpu().numpy()
            
            # Convert scalers to numpy if they are tensors
            y_std = scalers['y_std']
            if torch.is_tensor(y_std):
                y_std = y_std.cpu().numpy()
                
            y_mean = scalers['y_mean']
            if torch.is_tensor(y_mean):
                y_mean = y_mean.cpu().numpy()
            
            y_pred_unnorm = y_pred * y_std + y_mean
            
            return y_pred_unnorm
