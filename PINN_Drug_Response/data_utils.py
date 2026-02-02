import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# =========================================================================================
# EXPERIMENTAL DATASETS (A375 Melanoma Lines)
# Sourced from Matlab/Modeling/WB1/*.m files
# =========================================================================================

# Time points in hours
TIME_POINTS = np.array([0, 1, 4, 8, 24, 48])

# 1. COMBINATION THERAPY (VEMURAFENIB + TRAMETINIB)
# Source: VemandTram.m
VEM_TRAM_DATA = {
    'name': 'Vem_Tram_Combo',
    'time_points': TIME_POINTS,
    'species': {
        'pEGFR': np.array([0.222379739,0.622877159,0.629217784,0.533530834,0.022513609,0.010036399]),
        'HER2': np.array([0.306924546,0.275751955,0.32171108,0.23070312,1.013023288,1.045536401]),
        'HER3': np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
        'pDGFR': np.array([0.583361128,0.585284809,0.785897279,1.208147444,2.298226921,2.387788835]), # Replaces IGF1R
        'pCRAF': np.array([0.234376572,0.641878896,0.567434544,0.406320223,0.582899195,0.25113447]),
        'pMEK': np.array([1.936660577,0.029380652,0.012873835,0.03390921,0.095155796,0.944936578]),
        'pERK': np.array([3.273353557,0.075717978,0.011570416,0.00642985,0.041863585,0.91621491]),
        'DUSP6': np.array([2.854207662,2.842703936,1.163746208,0.332720449,0.030434242,0.094073888]),
        'pAKT': np.array([0.527301325,0.614645732,0.95895017,0.895019432,0.412820453,0.269891704]),
        'pS6K': np.array([1.385578651,1.388228355,1.286010223,0.720958901,0.12299088,0.028906108]),
        'p4EBP1': np.array([0.793559668,1.176099875,1.210864904,1.415698564,0.858543042,0.167293554])
    },
    'drugs': {
        'vemurafenib': 1.0,  # Normalized concentration (1.0 = High/Treatment dose)
        'trametinib': 1.0,
        'pi3k_inhibitor': 0.0,
        'ras_inhibitor': 0.0
    }
}
TRAINING_DATA_RAW = VEM_TRAM_DATA

# 2. VEMURAFENIB ONLY
# Source: Vemurafenib.m
VEM_ONLY_DATA = {
    'name': 'Vem_Only',
    'time_points': TIME_POINTS,
    'species': {
        'pEGFR': np.array([0.291928893,0.392400458,0.265016688,0.394238749,0.006158316,0.008115099]),
        'HER2': np.array([0.245236744,0.177917339,0.239075259,0.306884773,1.066654783,1.005085151]),
        'HER3': np.array([0.203233765,0.194358998,0.303475212,0.674083831,0.89702403,0.459831389]),
        'pDGFR': np.array([0.474174188,0.492132953,0.743620725,1.266460499,2.514722273,2.482761079]),
        'pCRAF': np.array([0.366397596,0.537106733,0.465541704,0.586732657,1.102322681,0.269181259]),
        'pMEK': np.array([1.75938884,0.170160085,0.095112609,0.201000276,0.219207054,0.502831668]),
        'pERK': np.array([2.903209735,0.207867788,0.303586121,0.805254439,1.408362153,1.847606441]),
        'DUSP6': np.array([2.677161325,2.782754577,1.130758062,0.395642757,0.828575853,0.916618219]),
        'pAKT': np.array([0.513544148,0.613178403,1.03451863,1.113391047,0.535242724,0.538273551]),
        'pS6K': np.array([1.432459522,1.520433646,1.542177411,1.248505245,0.109963216,0.013374136]),
        'p4EBP1': np.array([1.002468056,1.276793699,1.252681407,1.707504483,1.271216967,0.61389625])
    },
    'drugs': {
        'vemurafenib': 1.0,
        'trametinib': 0.0,
        'pi3k_inhibitor': 0.0,
        'ras_inhibitor': 0.0
    }
}

# 3. TRAMETINIB ONLY
# Source: Trametnib.m
TRAM_ONLY_DATA = {
    'name': 'Tram_Only',
    'time_points': TIME_POINTS,
    'species': {
        'pEGFR': np.array([0.35120201,0.545885712,0.642780201,0.359436744,0.009710876,0.01311083]),
        'HER2': np.array([0.245236744,0.177917339,0.239075259,0.306884773,1.066654783,1.005085151]),
        'HER3': np.array([0.203233765,0.194358998,0.303475212,0.674083831,0.89702403,0.459831389]),
        'pDGFR': np.array([0.474174188,0.492132953,0.743620725,1.266460499,2.514722273,2.482761079]),
        'pCRAF': np.array([0.233819043,0.451384259,0.355935204,0.786954922,0.946694075,0.26220219]),
        'pMEK': np.array([2.024063616,0.501888368,0.442535778,0.508820348,0.671938823,0.983710872]),
        'pERK': np.array([3.487158237,0.114217722,0.022162422,0.03726381,0.40696761,1.204092044]),
        'DUSP6': np.array([3.054033606,3.118384551,1.178142126,0.367382855,0.031574513,0.092390719]),
        'pAKT': np.array([0.549427631,0.642783939,1.046735362,0.944355203,0.479593107,0.310063914]),
        'pS6K': np.array([1.432459522,1.520433646,1.542177411,1.248505245,0.109963216,0.013374136]),
        'p4EBP1': np.array([0.984913436,1.263725259,1.355812598,1.380725265,0.941991484,0.281153536])
    },
    'drugs': {
        'vemurafenib': 0.0,
        'trametinib': 1.0,
        'pi3k_inhibitor': 0.0,
        'ras_inhibitor': 0.0
    }
}

# Mapping: IGF1R replaced by pDGFR based on Matlab data availability
SPECIES_ORDER = [
    'pEGFR', 'HER2', 'HER3', 'pDGFR', 'pCRAF', 'pMEK', 
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

def prepare_training_tensors(train_datasets=None, test_datasets=None, train_until_hour=48):
    """
    Prepares normalized tensors for training and testing.
    
    Args:
        train_datasets: List of datasets to use for training.
        test_datasets: List of datasets to use for prediction/testing.
        train_until_hour: If a dataset is in BOTH lists or if only train_datasets provided, 
                         splits by time. If datasets are distinct (Condition split), 
                         this primarily affects time normalization.
    """
    if train_datasets is None:
        raise ValueError("train_datasets must be provided")
    
    # If no test datasets provided, we assume we might be doing time-splitting on train_datasets?
    # But for this specific task (Condition Split), we expect explicit lists.
    # We will handle both cases.
    
    # helper to check if a dataset is in a list
    def is_in(ds, ds_list):
        return any(ds['name'] == d['name'] for d in ds_list)

    # Combine all for global normalization
    all_datasets = train_datasets + (test_datasets if test_datasets else [])
    # Remove duplicates for normalization calculation
    unique_datasets = {d['name']: d for d in all_datasets}.values()
    
    # ==================================================================
    # 1. PREPARE ALL DATA FOR SCALING
    # ==================================================================
    all_y = []
    all_t = []
    
    for dataset in unique_datasets:
        t_points = dataset['time_points'].astype(np.float32)
        num_points = len(t_points)
        y_data = np.zeros((num_points, len(SPECIES_ORDER)), dtype=np.float32)
        for i, species in enumerate(SPECIES_ORDER):
            y_data[:, i] = dataset['species'][species]
        all_t.append(t_points)
        all_y.append(y_data)

    y_all = np.concatenate(all_y, axis=0) if all_y else np.zeros((0, len(SPECIES_ORDER)))
    
    # Global Scales
    y_global_max = np.max(y_all, axis=0)
    y_global_min = np.zeros_like(y_global_max)
    y_scale = y_global_max + 1e-8
    
    t_max = 48.0 # Standardize on 48h
    
    scalers = {
        'y_mean': torch.tensor(y_global_min, dtype=torch.float32),
        'y_std': torch.tensor(y_scale, dtype=torch.float32),
        't_range': torch.tensor(t_max, dtype=torch.float32)
    }

    # ==================================================================
    # 2. BUILD TRAIN / TEST TENSORS
    # ==================================================================
    
    def process_dataset_list(ds_list, is_train_mode):
        t_list, y_list, drugs_list = [], [], []
        if not ds_list:
            return np.array([]), np.array([]), np.array([])
            
        for dataset in ds_list:
            t_points = dataset['time_points'].astype(np.float32)
            y_raw = np.zeros((len(t_points), len(SPECIES_ORDER)), dtype=np.float32)
            for i, sp in enumerate(SPECIES_ORDER):
                y_raw[:, i] = dataset['species'][sp]
            
            # If doing time split (dataset in both, or implicit), apply mask
            # But here we assume explicit lists dictate split usually. 
            # We'll just take ALL points from the provided datasets unless specifically asked to truncate.
            # User request: "train on Vem only... and predictions on vem+tram".
            # This implies taking ALL timepoints for train datasets, and ALL timepoints for test datasets.
            # However, if train_until_hour < 48 is passed, we might want to respect it for training data?
            # Let's respect train_until_hour for TRAIN datasets only.
            
            if is_train_mode:
                mask = t_points <= train_until_hour
            else:
                # For test datasets, we usually want the whole trajectory to see extrapolation
                mask = np.ones_like(t_points, dtype=bool)
                
            t_sub = t_points[mask]
            y_sub = y_raw[mask]
            
            drugs_vec = np.array([
                dataset['drugs']['vemurafenib'],
                dataset['drugs']['trametinib'],
                dataset['drugs']['pi3k_inhibitor'],
                dataset['drugs']['ras_inhibitor']
            ], dtype=np.float32)
            drugs_sub = np.tile(drugs_vec, (len(t_sub), 1))
            
            t_list.append(t_sub)
            y_list.append(y_sub)
            drugs_list.append(drugs_sub)
            
        return (np.concatenate(t_list) if t_list else np.array([]),
                np.concatenate(y_list) if y_list else np.array([]),
                np.concatenate(drugs_list) if drugs_list else np.array([]))

    t_train, y_train, drugs_train = process_dataset_list(train_datasets, is_train_mode=True)
    t_test, y_test, drugs_test = process_dataset_list(test_datasets, is_train_mode=False)
    
    # Normalize
    t_train_norm = (t_train / t_max).reshape(-1, 1) if t_train.size else t_train
    t_test_norm = (t_test / t_max).reshape(-1, 1) if t_test.size else t_test
    
    y_train_norm = y_train / y_scale if y_train.size else y_train
    y_test_norm = y_test / y_scale if y_test.size else y_test
    
    train_data = {
        't': t_train,
        't_norm': t_train_norm,
        'drugs': drugs_train,
        'y_norm': y_train_norm,
        'y_raw': y_train
    }
    
    test_data = {
        't': t_test,
        't_norm': t_test_norm,
        'drugs': drugs_test,
        'y_norm': y_test_norm,
        'y_raw': y_test
    }
    
    return train_data, test_data, scalers

def get_collocation_points(n_points=100, train_until_hour=48, t_max=48.0):
    """
    Generates collocation points for physics loss.
    """
    # Sample time uniformly
    t_physics = np.random.uniform(0, t_max, n_points).reshape(-1, 1)
    
    # Randomize drug concentrations for physics training
    # This is CRITICAL for the model to learn the effect of drugs it hasn't seen in exp data
    # ranges: [0, 1.0] for all drugs
    
    vemurafenib = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    trametinib = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    pi3k_inhibitor = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    ras_inhibitor = np.random.uniform(0, 1.0, size=(len(t_physics), 1))
    
    # Also include specific "pure" conditions to ensure boundaries are well-learned
    # e.g., mix in some cases with 0 drugs, or only 1 drug
    mask_pure = np.random.rand(len(t_physics)) < 0.3
    if np.any(mask_pure):
        # 30% of batch: strictly 0 or 1.0 for drugs to hit boundaries
        r = np.random.rand(np.sum(mask_pure))
        vals_vem = np.where(r < 0.33, 0.0, np.where(r < 0.66, 1.0, np.random.rand(np.sum(mask_pure))))
        vemurafenib[mask_pure] = vals_vem.reshape(-1, 1)
        # Independent random values for Trametinib
        r_tram = np.random.rand(np.sum(mask_pure))
        vals_tram = np.where(r_tram < 0.33, 0.0, np.where(r_tram < 0.66, 1.0, np.random.rand(np.sum(mask_pure))))
        trametinib[mask_pure] = vals_tram.reshape(-1, 1)
        pi3k_inhibitor[mask_pure] = 0 # Mostly focus on Vem/Tram
        ras_inhibitor[mask_pure] = 0
        
    drugs_physics = np.hstack([vemurafenib, trametinib, pi3k_inhibitor, ras_inhibitor])
    
    # Normalize time
    t_physics_norm = t_physics / t_max
    
    return torch.tensor(t_physics_norm, dtype=torch.float32), torch.tensor(drugs_physics, dtype=torch.float32)
