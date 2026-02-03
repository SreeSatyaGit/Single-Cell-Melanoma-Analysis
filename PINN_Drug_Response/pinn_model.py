import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish activation - smoother than ReLU, better gradients than Tanh"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class GatedResidualBlock(nn.Module):
    """Gated Residual Block with adaptive weighting for subtle signal capture."""
    def __init__(self, hidden_size):
        super(GatedResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()  # Swish (SiLU) activation
        self.ln = nn.LayerNorm(hidden_size) # Normalize for deep stability
        
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        
        # Gating mechanism: decides how much signal "flows"
        gate_weights = torch.sigmoid(self.gate(x))
        out = out * gate_weights
        
        return self.ln(out + residual)

class PINN(nn.Module):
    """
    Advanced Physics-Informed Neural Network.
    Features: Multi-path input embedding, Gated Residual blocks, and LayerNorm.
    """
    def __init__(self, input_size=5, hidden_size=256, output_size=11, num_hidden=6):
        super(PINN, self).__init__()
        
        # 1. Input Processing Paths
        # Path for Time (Temporal dynamics)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size // 4),
            nn.SiLU()
        )
        # Path for Drugs (Context/Perturbation)
        self.drug_embed = nn.Sequential(
            nn.Linear(4, hidden_size - (hidden_size // 4)),
            nn.SiLU()
        )
        
        # 2. Main Processing Core (Deeper & Gated)
        self.res_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_size) for _ in range(num_hidden)
        ])
        
        # 3. Dedicated pMEK/pERK attention bridge
        # Helps link the hierarchical nature of the MAPK cascade
        self.mapk_attention = nn.Linear(hidden_size, hidden_size)
        
        # 4. Output Logic
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # SPECIAL BIOLOGICAL INITIALIZATION
        # Initialize output bias to 0.5 so predictions start near the biological mean
        # rather than at the Softplus floor (near zero).
        nn.init.constant_(self.output_layer.bias, 0.5)

    def forward(self, t, drugs):
        # Multi-path embedding fusion
        t_feat = self.time_embed(t)
        d_feat = self.drug_embed(drugs)
        x = torch.cat([t_feat, d_feat], dim=1)
        
        # Residual processing
        for block in self.res_blocks:
            x = block(x)
            
        # Subtle non-linear transformation for high-order relations
        x = x + torch.tanh(self.mapk_attention(x))
            
        out = self.output_layer(x)
        return self.softplus(out)

    def predict(self, t_np, drugs_dict, scalers, device='cpu', normalized=True):
        """
        Utility for inference with numpy inputs.
        
        Args:
            t_np: Time points (numpy array)
            drugs_dict: Dictionary of drug concentrations
            scalers: Dictionary containing scaling factors
            device: 'cpu' or 'cuda'
            normalized: If True (default), returns results in [0, 1] range.
                       If False, returns results unnormalized (A.U.).
        """
        self.eval()
        with torch.no_grad():
            # 1. Prepare time (normalized to 0-48 range)
            t_max = scalers['t_range'].item() if torch.is_tensor(scalers['t_range']) else scalers['t_range']
            t_norm = t_np / t_max
            t_tensor = torch.tensor(t_norm, dtype=torch.float32).view(-1, 1).to(device)
            
            # 2. Prepare drugs
            drugs_vec = torch.tensor([
                drugs_dict['vemurafenib'],
                drugs_dict['trametinib'],
                drugs_dict['pi3k_inhibitor'],
                drugs_dict['ras_inhibitor']
            ], dtype=torch.float32).view(1, -1).to(device)
            drugs_tensor = drugs_vec.repeat(t_tensor.size(0), 1)
            
            # 3. Forward pass (outputs are already in [0, 1] due to softplus/training)
            y_pred_norm = self.forward(t_tensor, drugs_tensor)
            y_pred = y_pred_norm.cpu().numpy()
            
            if normalized:
                return y_pred
                
            # 4. Unnormalize if requested
            y_std = scalers['y_std']
            if torch.is_tensor(y_std):
                y_std = y_std.cpu().numpy()
                
            y_mean = scalers['y_mean']
            if torch.is_tensor(y_mean):
                y_mean = y_mean.cpu().numpy()
            
            y_pred_unnorm = y_pred * y_std + y_mean
            return y_pred_unnorm
