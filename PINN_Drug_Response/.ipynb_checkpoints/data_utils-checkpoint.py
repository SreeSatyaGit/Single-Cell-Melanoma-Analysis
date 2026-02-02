import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Training data from western blot experiments
TRAINING_DATA_RAW = {
    'time_points': np.array([0, 1, 4, 8, 24, 48]),
    'species': {
        'pEGFR': np.array([0.222379739,0.622877159,0.629217784,0.533530834,0.022513609,0.010036399]),
        'HER2': np.array([0.306924546,0.275751955,0.32171108,0.23070312,1.013023288,1.045536401]),
        'HER3': np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
        'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
        'pCRAF': np.array([0.234376572,0.641878896,0.567434544,0.406320223,0.582899195,0.25113447]),
        'pMEK': np.array([1.936660577,0.029380652,0.012873835,0.03390921,0.095155796,0.944936578]),
        'pERK': np.array([3.273353557,0.075717978,0.011570416,0.00642985,0.041863585,0.91621491]),
        'DUSP6': np.array([2.854207662,2.842703936,1.163746208,0.332720449,0.030434242,0.094073888]),
        'pAKT': np.array([0.549427631,0.642783939,1.046735362,0.944355203,0.479593107,0.310063914]),
        'pS6K': np.array([0.527301325,0.614645732,0.95895017,0.895019432,0.412820453,0.269891704]),
        'p4EBP1': np.array([0.793559668,1.176099875,1.210864904,1.415698564,0.858543042,0.167293554])
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

def prepare_training_tensors(train_until_hour=8):
    """
    Prepares normalized tensors for training with train/test split.
    
    NORMALIZATION STRATEGY (Optimized for PINNs):
    =============================================
    1. TIME: Normalized to [0, 1] using t_max = 48 hours
    2. SPECIES: Each species normalized to [0, 1] using GLOBAL max
       - Using global max (not training max) prevents test values > 1.0
       - Independent scaling gives equal weight to all proteins in loss
    3. DRUGS: Kept as-is (already in [0, 1] range)
    
    Args:
        train_until_hour: Train only on time points <= this value.
                         Default 8 means train on [0,1,4,8], test on [24,48]
    """
    t_points = TRAINING_DATA_RAW['time_points'].astype(np.float32)
    num_points = len(t_points)
    
    # ==================================================================
    # 1. PREPARE SPECIES DATA
    # ==================================================================
    y_data = np.zeros((num_points, 11), dtype=np.float32)
    for i, species in enumerate(SPECIES_ORDER):
        y_data[:, i] = TRAINING_DATA_RAW['species'][species]
    
    # ==================================================================
    # 2. TRAIN/TEST SPLIT
    # ==================================================================
    train_mask = t_points <= train_until_hour
    t_train = t_points[train_mask]
    y_train = y_data[train_mask]
    
    t_test = t_points[~train_mask]
    y_test = y_data[~train_mask]
    
    # ==================================================================
    # 3. SPECIES NORMALIZATION: [0, 1] using GLOBAL MAX
    # ==================================================================
    # Use global max (all data) to prevent test values from exceeding 1.0
    y_global_max = np.max(y_data, axis=0)  # Max across ALL time points
    y_global_min = np.zeros_like(y_global_max)  # Min is 0 for biological data
    y_scale = y_global_max + 1e-8  # Scale factor = max value
    
    y_train_norm = y_train / y_scale
    y_test_norm = y_test / y_scale
    
    # ==================================================================
    # 4. TIME NORMALIZATION: [0, 1]
    # ==================================================================
    t_max = 48.0
    t_train_norm = (t_train / t_max).reshape(-1, 1)
    t_test_norm = (t_test / t_max).reshape(-1, 1)
    
    # ==================================================================
    # 5. DRUGS: Keep as-is (already [0, 1])
    # ==================================================================
    drugs_raw = TRAINING_DATA_RAW['drugs']
    drugs_vec = np.array([
        drugs_raw['vemurafenib'],
        drugs_raw['trametinib'],
        drugs_raw['pi3k_inhibitor'],
        drugs_raw['ras_inhibitor']
    ], dtype=np.float32)
    drugs_train = np.tile(drugs_vec, (len(t_train), 1))
    drugs_test = np.tile(drugs_vec, (len(t_test), 1))
    
    # ==================================================================
    # 6. SCALERS (for physics loss and inference)
    # ==================================================================
    scalers = {
        'y_mean': torch.tensor(y_global_min, dtype=torch.float32),  # Min = 0
        'y_std': torch.tensor(y_scale, dtype=torch.float32),        # Scale = max
        't_range': torch.tensor(t_max, dtype=torch.float32)         # Time scale = 48
    }
    
    train_data = {
        't': t_train,
        't_norm': t_train_norm,
        'drugs': drugs_train,
        'y': y_train_norm,      # Mapping 'y' to normalized for easy plotting
        'y_norm': y_train_norm, # For backward compatibility
        'y_raw': y_train        # Original A.U. values
    }
    
    test_data = {
        't': t_test,
        't_norm': t_test_norm,
        'drugs': drugs_test,
        'y': y_test_norm,       # Mapping 'y' to normalized
        'y_norm': y_test_norm,  # Compatibility
        'y_raw': y_test         # Original
    }
    
    return train_data, test_data, scalers

def get_collocation_points(n_points=100, extrapolation_weight=2.0):
    """
    Generates collocation points for physics loss.
    CRITICAL: Must cover ENTIRE time domain including extrapolation region.
    
    Args:
        n_points: Total number of collocation points
        extrapolation_weight: How many more points in extrapolation vs training region
    """
    # Split points between training region and extrapolation region
    # Training: 0-8hrs (normalized: 0-0.167)
    # Extrapolation: 8-48hrs (normalized: 0.167-1.0)
    
    n_train_region = int(n_points / (1 + extrapolation_weight))
    n_extrap_region = n_points - n_train_region
    
    # Training region: 0 to 8 hrs (normalized 0 to 8/48 = 0.167)
    t_train_region = np.linspace(0, 8/48, n_train_region)
    
    # Extrapolation region: 8 to 48 hrs (normalized 0.167 to 1.0)
    t_extrap_region = np.linspace(8/48, 1.0, n_extrap_region)
    
    t_physics = np.concatenate([t_train_region, t_extrap_region]).reshape(-1, 1)
    
    # Randomize drug concentrations for physics training
    # This is CRITICAL for the model to learn the effect of drugs it hasn't seen in exp data
    # ranges: [0, 1.0] for all drugs
    
    vemurafenib = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    trametinib = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    pi3k_inhibitor = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    ras_inhibitor = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    
    # Also include specific "pure" conditions to ensure boundaries are well-learned
    # e.g., mix in some cases with 0 drugs, or only 1 drug
    mask_pure = np.random.rand(len(t_physics)) < 0.2
    if np.any(mask_pure):
        vemurafenib[mask_pure] = 0
        trametinib[mask_pure] = 0
        pi3k_inhibitor[mask_pure] = 0
        ras_inhibitor[mask_pure] = 0
        
    drugs_physics = np.hstack([vemurafenib, trametinib, pi3k_inhibitor, ras_inhibitor])
    
    return torch.tensor(t_physics, dtype=torch.float32), torch.tensor(drugs_physics, dtype=torch.float32)
