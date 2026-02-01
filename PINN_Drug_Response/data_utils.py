import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Training data from western blot experiments
TRAINING_DATA_RAW = {
    'time_points': np.array([0, 1, 4, 8, 24, 48]),
    'species': {
        'pEGFR': np.array([0.187429057, 0.194104013, 0.273793266, 0.276513687, 0.34600589, 0.372292386]),
        'HER2': np.array([0.306924546, 0.275751955, 0.32171108, 0.23070312, 1.013023288, 1.045536401]),
        'HER3': np.array([0.295284147, 0.285719072, 0.385045943, 0.582261781, 0.751301308, 0.264889608]),
        'IGF1R': np.array([1.180034579, 0.967927178, 0.808905442, 0.781013289, 0.41928501, 0.870763253]),
        'pCRAF': np.array([0.234376572, 0.641878896, 0.567434544, 0.406320223, 0.582899195, 0.25113447]),
        'pMEK': np.array([1.936660577, 0.029380652, 0.01287383, 0.03390921, 0.095155796, 0.944936578]),
        'pERK': np.array([3.273353557, 0.075717978, 0.011570416, 0.00642985, 0.041863585, 0.91621491]),
        'DUSP6': np.array([2.854207662, 2.842703936, 1.163746208, 0.332720449, 0.030434242, 0.094073888]),
        'pAKT': np.array([0.527301325, 0.614645732, 0.95895017, 0.895019432, 0.412820453, 0.269891704]),
        'pS6K': np.array([1.385578651, 1.388228355, 1.286010223, 0.720958901, 0.12299088, 0.028906108]),
        'p4EBP1': np.array([0.793559668, 1.176099875, 1.210864904, 1.415698564, 0.858543042, 0.167293554])
    },
    'drugs': {
        'vemurafenib': 0.5,
        'trametinib': 0.3,
        'pi3k_inhibitor': 0.0,
        'ras_inhibitor': 0.0
    }
}

SPECIES_ORDER = [
    'pEGFR', 'HER2', 'HER3', 'IGF1R', 'pCRAF', 'pMEK', 
    'pERK', 'DUSP6', 'pAKT', 'pS6K', 'p4EBP1'
]

class SignalingDataset(Dataset):
    def __init__(self, t, drugs, y):
        self.t = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        self.drugs = torch.tensor(drugs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        return self.t[idx], self.drugs[idx], self.y[idx]

def prepare_training_tensors():
    """
    Prepares normalized tensors for training.
    """
    t_points = TRAINING_DATA_RAW['time_points']
    num_points = len(t_points)
    
    # 1. Prepare Y (species data)
    y_data = np.zeros((num_points, 11))
    for i, species in enumerate(SPECIES_ORDER):
        y_data[:, i] = TRAINING_DATA_RAW['species'][species]
    
    # 2. Compute scaling factors (z-score for outputs)
    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0) + 1e-6
    y_norm = (y_data - y_mean) / y_std
    
    # 3. Time scaling (min-max)
    t_min = t_points.min()
    t_max = t_points.max()
    t_range = t_max - t_min
    t_norm = (t_points - t_min) / t_range
    
    # 4. Drug concentration (repeated for each time point)
    drugs_raw = TRAINING_DATA_RAW['drugs']
    drugs_vec = np.array([
        drugs_raw['vemurafenib'],
        drugs_raw['trametinib'],
        drugs_raw['pi3k_inhibitor'],
        drugs_raw['ras_inhibitor']
    ])
    drugs_data = np.tile(drugs_vec, (num_points, 1))
    
    # Save scalers for later use
    scalers = {
        'y_mean': torch.tensor(y_mean, dtype=torch.float32),
        'y_std': torch.tensor(y_std, dtype=torch.float32),
        't_min': t_min,
        't_max': t_max,
        't_range': t_range
    }
    
    return t_norm, drugs_data, y_norm, scalers

def get_collocation_points(n_points=100):
    """
    Generates random collocation points for physics loss.
    """
    t_physics = np.linspace(0, 1, n_points).reshape(-1, 1)
    # Use training drugs for collocation points
    drugs_raw = TRAINING_DATA_RAW['drugs']
    drugs_vec = np.array([
        drugs_raw['vemurafenib'],
        drugs_raw['trametinib'],
        drugs_raw['pi3k_inhibitor'],
        drugs_raw['ras_inhibitor']
    ])
    drugs_physics = np.tile(drugs_vec, (n_points, 1))
    
    return torch.tensor(t_physics, dtype=torch.float32), torch.tensor(drugs_physics, dtype=torch.float32)
