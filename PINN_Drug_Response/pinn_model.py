import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """Swish activation - smoother than ReLU, better gradients than Tanh"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow"""
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = Swish()
        
    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return self.activation(out + residual)

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for Cancer Signaling Pathways.
    Uses residual connections and Swish activation for better extrapolation.
    """
    def __init__(self, input_size=5, hidden_size=128, output_size=11, num_hidden=4):
        super(PINN, self).__init__()
        
        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.activation = Swish()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_hidden)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        
        # Initialize weights with smaller values for stability
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t, drugs):
        x = torch.cat([t, drugs], dim=1)
        x = self.activation(self.input_layer(x))
        
        for block in self.res_blocks:
            x = block(x)
            
        out = self.output_layer(x)
        return self.softplus(out)

    def predict(self, t_np, drugs_dict, scalers, device='cpu'):
        """
        Utility for inference with numpy inputs.
        """
        self.eval()
        with torch.no_grad():
            # Prepare time (normalized to 0-48 range)
            t_norm = t_np / scalers['t_range'].item()
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
