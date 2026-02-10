import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Multiple training conditions from western blot experiments
# Each entry in the list represents a different drug condition
TRAINING_DATA_LIST = [
    {
        'name': 'Vemurafenib Only (0.5)',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.222379739,0.622877159,0.629217784,0.533530834,0.022513609,0.010036399]),
            'HER2': np.array([0.306924546,0.275751955,0.32171108,0.23070312,1.013023288,1.045536401]),
            'HER3':  np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
            'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
            'pCRAF': np.array([0.234376572,0.641878896,0.567434544,0.406320223,0.582899195,0.25113447]),
            'pMEK': np.array([1.936660577,0.029380652,0.012873835,0.03390921,0.095155796,0.944936578]),
            'pERK': np.array([3.273353557,0.075717978,0.011570416,0.00642985,0.041863585,0.91621491]),
            'DUSP6': np.array([2.854207662,2.842703936,1.163746208,0.332720449,0.030434242,0.094073888]),
            'pAKT': np.array([0.549427631,0.642783939,1.046735362,0.944355203,0.479593107,0.310063914]),
            'pS6K': np.array([0.527301325,0.614645732,0.95895017,0.895019432,0.412820453,0.269891704]),
            'p4EBP1': np.array([0.793559668,1.176099875,1.210864904,1.415698564,0.858543042,0.167293554])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.0, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Trametinib Only (0.3)',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.35120201,0.545885712,0.642780201,0.359436744,0.009710876,0.01311083]), 
            'HER2':  np.array([0.328412641,0.302493913,0.32569075,0.254737487,1.050977922,1.026829681]),
            'HER3':  np.array([0.297534144,0.20792856,0.227078284,0.297261013,0.966331288,0.97673174]),
            'IGF1R': np.array([1.193833833,1.170396382,1.060073053,0.983950361,0.525472084,0.298723901]),
            'pCRAF': np.array([0.233819043,0.451384259,0.355935204,0.786954922,0.946694075,0.26220219]),
            'pMEK':  np.array([2.024063616,0.501888368,0.442535778,0.508820348,0.671938823,0.983710872]),
            'pERK':  np.array([3.487158237,0.114217722,0.022162422,0.03726381,0.40696761,1.204092044]),
            'DUSP6': np.array([3.054033606,3.118384551,1.178142126,0.367382855,0.031574513,0.092390719]),
            'pAKT':  np.array([0.549427631,0.642783939,1.046735362,0.944355203,0.479593107,0.310063914]),
            'pS6K':  np.array([1.188057499,1.407252052,1.24589939,1.009394508,0.076418566,0.011264414]),
            'p4EBP1': np.array([0.984913436,1.263725259,1.355812598,1.380725265,0.941991484,0.281153536])
        },
        'drugs': {'vemurafenib': 0.0, 'trametinib': 0.3, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vem + Tram Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.222379739,0.622877159,0.629217784,0.533530834,0.022513609,0.010036399]), # Placeholder values
            'HER2':  np.array([0.306924546,0.275751955,0.32171108,0.23070312,1.013023288,1.045536401]),
            'HER3':  np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
            'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
            'pCRAF': np.array([0.234376572,0.641878896,0.567434544,0.406320223,0.582899195,0.25113447]),
            'pMEK':  np.array([1.936660577,0.029380652,0.012873835,0.03390921,0.095155796,0.944936578]),
            'pERK':  np.array([3.273353557,0.075717978,0.011570416,0.00642985,0.041863585,0.91621491]),
            'DUSP6': np.array([2.854207662,2.842703936,1.163746208,0.332720449,0.030434242,0.094073888]),
            'pAKT':  np.array([0.527301325,0.614645732,0.95895017,0.895019432,0.412820453,0.269891704]),
            'pS6K':  np.array([1.385578651,1.388228355,1.286010223,0.720958901,0.12299088,0.028906108]),
            'p4EBP1': np.array([0.793559668,1.176099875,1.210864904,1.415698564,0.858543042,0.167293554])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.3, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vem + PI3Ki Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.333114406,0.304726551,0.329135261,0.368629137,0.15272344,0.056896369]), 
            'HER2':  np.array([0.8,0.8,0.8,0.3,0.3,0.3]),
            'HER3':  np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
            'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
            'pCRAF': np.array([0.228812616,0.14996033,0.140558542,0.142348739,0.153630718,0.062348623]),
            'pMEK':  np.array([1.33376507,1.368265108,1.368011013,1.367879453,0.064659593,0.098798053]), 
            'pERK':  np.array([1.047555559,0.026391429,0.032430347,0.034839997,0.082959596,1.047391392]),
            'DUSP6': np.array([0.940002873,0.94013042,0.939602496,0.038674106,0.037145377,0.939351267]),
            'pAKT':  np.array([0.052490516,0.002881302,0.004164565,0.009998056,0.009415223,0.012665673]), 
            'pS6K':  np.array([1.385578651,1.388228355,1.286010223,0.720958901,0.12299088,0.028906108]),
            'p4EBP1': np.array([0.989227226,0.85,0.76,0.6,0.06,0.3])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.0, 'pi3k_inhibitor': 0.5, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vem + panRAS Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.5,0.1,0.1,0.3,0.3,0.3]), 
            'HER2':  np.array([0.8,0.8,0.8,0.3,0.3,0.3]),
            'HER3':  np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
            'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
            'pCRAF': np.array([0.228812616,0.14996033,0.140558542,0.142348739,0.153630718,0.062348623]),
            'pMEK':  np.array([1.33,0,0,0.3,0.4,1.33]), 
            'pERK':  np.array([1.04,0,0,0.3,0.4,1.04]),
            'DUSP6': np.array([0.7,0.2,0.3,0.2,0.3,0.7]),
            'pAKT':  np.array([0.1,0.6,0.2,0.7,0.4,0.9]), 
            'pS6K':  np.array([1.385578651,1.388228355,1.286010223,0.720958901,0.12299088,0.028906108]),
            'p4EBP1': np.array([0.989227226,0.85,0.76,0.6,0.76,1.45])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.0, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.5}
    }
]

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
    Aggregates data from multiple experiments and prepares tensors.
    Uses Robust Max-Scaling (1.1 * Max) to ensure headroom for predictions.
    """
    all_t, all_y, all_drugs = [], [], []
    
    # 1. Collect data from ALL experiments
    for exp in TRAINING_DATA_LIST:
        t_points = exp['time_points'].astype(np.float32)
        num_pts = len(t_points)
        
        y_exp = np.zeros((num_pts, 11), dtype=np.float32)
        for i, species in enumerate(SPECIES_ORDER):
            y_exp[:, i] = exp['species'][species]
            
        drug_vec = np.array([
            exp['drugs']['vemurafenib'],
            exp['drugs']['trametinib'],
            exp['drugs']['pi3k_inhibitor'],
            exp['drugs']['ras_inhibitor']
        ], dtype=np.float32)
        drug_mat = np.tile(drug_vec, (num_pts, 1))
        
        all_t.append(t_points)
        all_y.append(y_exp)
        all_drugs.append(drug_mat)
        
    t_data = np.concatenate(all_t)
    y_data = np.concatenate(all_y)
    drugs_data = np.concatenate(all_drugs)
    
    # 2. Train/Test Split logic
    train_mask = t_data <= train_until_hour
    
    # 3. Robust Normalization Factors
    # Using 1.1x Max provides "numerical headroom" so predictions don't hit the Softplus ceiling
    y_scale = np.max(y_data, axis=0) * 1.1 + 1e-8
    t_max = 48.0
    
    # 4. Prepare Outputs
    train_data = {
        't': t_data[train_mask],
        't_norm': (t_data[train_mask] / t_max).reshape(-1, 1),
        'drugs': drugs_data[train_mask],
        'y_norm': y_data[train_mask] / y_scale,
        'y_raw': y_data[train_mask]
    }
    
    test_data = {
        't': t_data[~train_mask],
        't_norm': (t_data[~train_mask] / t_max).reshape(-1, 1),
        'drugs': drugs_data[~train_mask],
        'y_norm': y_data[~train_mask] / y_scale,
        'y_raw': y_data[~train_mask]
    }
    
    scalers = {
        'y_mean': torch.zeros(11),
        'y_std': torch.tensor(y_scale, dtype=torch.float32),
        't_range': torch.tensor(t_max, dtype=torch.float32)
    }
    
    return train_data, test_data, scalers
    
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
