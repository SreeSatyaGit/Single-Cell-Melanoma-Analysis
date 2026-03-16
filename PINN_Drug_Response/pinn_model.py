import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, Union
logger = logging.getLogger(__name__)
TIME_EMBED_RATIO = 0.5
BIOLOGICAL_BASELINE = 0.2
class GatedResidualBlock(nn.Module):
    """Gated Residual Block with adaptive weighting for subtle signal capture."""
    def __init__(self, hidden_size: int):
        super(GatedResidualBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.SiLU()
        self.ln = nn.LayerNorm(hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        gate_weights = torch.sigmoid(self.gate(x))
        out = out * gate_weights
        return self.ln(out + residual)
class PINN(nn.Module):
    """
    Advanced Physics-Informed Neural Network.
    Features: Multi-path input embedding, Gated Residual blocks, and LayerNorm.
    """
    DRUG_ORDER = ['vemurafenib', 'trametinib', 'pi3k_inhibitor', 'ras_inhibitor']
    VERSION = "1.1.0"
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 256,
        output_size: int = 10,
        num_hidden: int = 6
    ) -> None:
        super(PINN, self).__init__()
        self.model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'output_size': output_size,
            'num_hidden': num_hidden,
            'version': self.VERSION
        }
        time_dim = int(hidden_size * TIME_EMBED_RATIO)
        drug_dim = hidden_size - time_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU()
        )
        self.drug_embed = nn.Sequential(
            nn.Linear(4, drug_dim),
            nn.SiLU()
        )
        self.res_blocks = nn.ModuleList([
            GatedResidualBlock(hidden_size) for _ in range(num_hidden)
        ])
        self.mapk_attention = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        self._init_weights()
        logger.info(f"PINN v{self.VERSION} initialized: hidden={hidden_size}, layers={num_hidden}, params={sum(p.numel() for p in self.parameters()):,}")
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.output_layer.bias, BIOLOGICAL_BASELINE)
    def forward(self, t: torch.Tensor, drugs: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_embed(t)
        d_feat = self.drug_embed(drugs)
        x = torch.cat([t_feat, d_feat], dim=1)
        for block in self.res_blocks:
            x = block(x)
        x = x + torch.tanh(self.mapk_attention(x))
        out = self.output_layer(x)
        return self.softplus(out)
    def predict(
        self,
        t_np: np.ndarray,
        drugs_dict: Dict[str, float],
        scalers: Dict[str, Any],
        device: str = 'cpu',
        normalized: bool = True
    ) -> np.ndarray:
        """
        High-level inference utility with numpy inputs.
        Args:
            t_np: Time points in hours (numpy array).
            drugs_dict: Dictionary of drug concentrations keyed by canonical drug names.
            scalers: Dictionary containing 't_range', 'y_std', 'y_mean' scaling factors.
            device: 'cpu' or 'cuda'.
            normalized: If True, returns results in normalized range.
                       If False, returns results in original A.U. scale.
        Returns:
            Predictions as a numpy array of shape (len(t_np), 10).
        """
        missing_drugs = set(self.DRUG_ORDER) - set(drugs_dict.keys())
        if missing_drugs:
            raise ValueError(f"Missing drug concentrations: {missing_drugs}. Expected keys: {self.DRUG_ORDER}")
        required_scalers = ['t_range', 'y_std', 'y_mean']
        missing_scalers = set(required_scalers) - set(scalers.keys())
        if missing_scalers:
            raise ValueError(f"Missing scalers: {missing_scalers}")
        self.eval()
        with torch.no_grad():
            t_max = scalers['t_range'].item() if torch.is_tensor(scalers['t_range']) else scalers['t_range']
            t_tensor = torch.as_tensor(t_np / t_max, dtype=torch.float32).view(-1, 1).to(device)
            drugs_vec = torch.tensor(
                [drugs_dict[d] for d in self.DRUG_ORDER],
                dtype=torch.float32
            ).view(1, -1).to(device)
            drugs_tensor = drugs_vec.expand(t_tensor.size(0), -1)
            y_pred_norm = self.forward(t_tensor, drugs_tensor)
            y_pred = y_pred_norm.cpu().numpy()
            if normalized:
                return y_pred
            y_std = scalers['y_std'].cpu().numpy() if torch.is_tensor(scalers['y_std']) else scalers['y_std']
            y_mean = scalers['y_mean'].cpu().numpy() if torch.is_tensor(scalers['y_mean']) else scalers['y_mean']
            return y_pred * y_std + y_mean
