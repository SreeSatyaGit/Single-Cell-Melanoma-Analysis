from typing import Dict, List, Tuple, Optional, Any
import torch
import numpy as np
from torch.utils.data import Dataset

ExperimentalCondition = Dict[str, Any]
SpeciesData = Dict[str, np.ndarray]

SPECIES_ORDER: List[str] = [
    'pEGFR', 'HER2', 'HER3', 'IGF1R', 'pCRAF', 'pMEK', 
    'pERK', 'DUSP6', 'pAKT', 'p4EBP1'
]

def _compute_basal_steady_state() -> Dict[str, np.ndarray]:
    """
    Compute the average t=0 (pre-treatment) protein levels across all
    experimental conditions. These represent the constitutive steady state.
    
    Returns a dict of {species_name: scalar_value} for the basal level.
    """
    _t0_values = {
        'pEGFR': [0.291928893, 0.35120201, 0.222379739, 0.333114406, 0.641846],
        'HER2':  [0.245236744, 0.328412641, 0.306924546, 0.411424, 0.745302],
        'HER3':  [0.203233765, 0.297534144, 0.295284147, 0.411424, 0.745302],
        'IGF1R': [0.96024414, 1.193833833, 1.180034579, 1.365603, 1.179241],
        'pCRAF': [0.366397596, 0.233819043, 0.234376572, 0.989486, 2.764099],
        'pMEK':  [1.75938884, 2.024063616, 1.936660577, 2.018304, 2.462343],
        'pERK':  [2.903209735, 3.487158237, 3.273353557, 3.47456, 4.450806],
        'DUSP6': [2.677161325, 3.054033606, 2.854207662, 1.47878, 1.519079],
        'pAKT':  [1.080071, 0.549427631, 0.527301325, 0.320037, 0.536743],
        'p4EBP1':[1.002468056, 0.984913436, 0.793559668, 0.698339, 0.74627],
    }
    return {sp: np.mean(vals) for sp, vals in _t0_values.items()}

_BASAL_SS = _compute_basal_steady_state()

_NO_DRUG_TIME_POINTS = np.array([0, 1, 4, 8, 24, 48])
_NO_DRUG_SPECIES = {
    sp: np.full(len(_NO_DRUG_TIME_POINTS), _BASAL_SS[sp])
    for sp in SPECIES_ORDER
}

TRAINING_DATA_LIST: List[Dict[str, Any]] = [
    {
        'name': 'No Drug (Basal Steady State)',
        'time_points': _NO_DRUG_TIME_POINTS,
        'species': _NO_DRUG_SPECIES,
        'drugs': {'vemurafenib': 0.0, 'trametinib': 0.0, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vemurafenib Only (0.5)',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.291928893,0.392400458,0.265016688,0.394238749,0.006158316,0.008115099]),
            'HER2': np.array([0.245236744,0.177917339,0.239075259,0.306884773,1.066654783,1.005085151]),
            'HER3':  np.array([0.203233765,0.194358998,0.303475212,0.674083831,0.89702403,0.459831389]),
            'IGF1R': np.array([0.96024414,1.070624757,1.126551098,1.157072001,0.734617533,0.4579226222]),
            'pCRAF': np.array([0.366397596,0.537106733,0.465541704,0.586732657,1.102322681,0.269181259]),
            'pMEK': np.array([1.75938884,0.170160085,0.095112609,0.201000276,0.219207054,0.502831668]),
            'pERK': np.array([2.903209735,0.207867788,0.303586121,0.805254439,1.408362153,1.847606441]),
            'DUSP6': np.array([2.677161325,2.782754577,1.130758062,0.395642757,0.828575853,0.916618219]),
            'pAKT': np.array([1.080071,0.900024,1.066681,0.899337,1.21576,1.333254]),
            'p4EBP1': np.array([1.002468056,1.276793699,1.252681407,1.707504483,1.271216967,0.61389625])
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
            'p4EBP1': np.array([0.984913436,1.263725259,1.355812598,1.380725265,0.941991484,0.281153536])
        },
        'drugs': {'vemurafenib': 0.0, 'trametinib': 0.3, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },

    {
        'name': 'Vem + Tram Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.222379739,0.622877159,0.629217784,0.533530834,0.022513609,0.010036399]),
            'HER2':  np.array([0.306924546,0.275751955,0.32171108,0.23070312,1.013023288,1.045536401]),
            'HER3':  np.array([0.295284147,0.285719072,0.385045943,0.582261781,0.751301308,0.264889608]),
            'IGF1R': np.array([1.180034579,0.967927178,0.808905442,0.781013289,0.41928501,0.870763253]),
            'pCRAF': np.array([0.234376572,0.641878896,0.567434544,0.406320223,0.582899195,0.25113447]),
            'pMEK':  np.array([1.936660577,0.029380652,0.012873835,0.03390921,0.095155796,0.944936578]),
            'pERK':  np.array([3.273353557,0.075717978,0.011570416,0.00642985,0.041863585,0.91621491]),
            'DUSP6': np.array([2.854207662,2.842703936,1.163746208,0.332720449,0.030434242,0.094073888]),
            'pAKT':  np.array([0.527301325,0.614645732,0.95895017,0.895019432,0.412820453,0.269891704]),
            'p4EBP1': np.array([0.793559668,1.176099875,1.210864904,1.415698564,0.858543042,0.167293554])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.3, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vem + PI3Ki Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.333114406,0.304726551,0.329135261,0.368629137,0.15272344,0.056896369]), 
            'HER2':  np.array([0.411424,0.55907,0.526718,0.938131,1.064491,2.521235]),
            'HER3':  np.array([0.411424,0.55907,0.526718,0.938131,1.064491,2.521235]),
            'IGF1R': np.array([1.365603,1.141697,1.536194,1.383503,1.438155,1.746471]),
            'pCRAF': np.array([0.989486,0.600938,0.567638,0.406382,0.228887,0.49824]),
            'pMEK':  np.array([2.018304,0.340262,0.812539,0.878866,0.478651,0.899268]), 
            'pERK':  np.array([3.47456,0,2.286908,2.961433,2.12377,3.14885]),
            'DUSP6': np.array([1.47878,1.186021,0.677175,0.826286,0.453723,1.852728]),
            'pAKT':  np.array([0.320037,0.026226,0.163851,0.05965,0.215918,0.29892]), 
            'p4EBP1': np.array([0.698339,0.175174,0.146055,0.10574,0.198613,0.131825])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.0, 'pi3k_inhibitor': 0.5, 'ras_inhibitor': 0.0}
    },
    {
        'name': 'Vem + panRAS Combo',
        'time_points': np.array([0, 1, 4, 8, 24, 48]),
        'species': {
            'pEGFR': np.array([0.641846,0.40072,0.16665,0.265356,0.338174,0.072136]), 
            'HER2':  np.array([0.745302,0.421645,0.527132,0.349948,0.436277,0.299083]),
            'HER3':  np.array([0.745302,0.421645,0.527132,0.349948,0.436277,0.299083]),
            'IGF1R': np.array([1.179241,0.97553,0.59481,0.710898,0.607603,0.62405]),
            'pCRAF': np.array([2.764099,2.230603,3.58039,2.311266,2.405439,1.47887]),
            'pMEK':  np.array([2.462343,0.156141,0.599231,1.118925,0.964204,0.73809]), 
            'pERK':  np.array([4.450806,0,0.192028,0.213346,0.534783,3.47456]),
            'DUSP6': np.array([1.519079,0.420432,0,0.173392,0.068671,0.181719]),
            'pAKT':  np.array([0.536743,1.171471,1.100104,0.904367,0.415397,1.076115]), 
            'p4EBP1': np.array([0.74627,1.027582,0.89376,0.876078,0.41763,0.215742])
        },
        'drugs': {'vemurafenib': 0.5, 'trametinib': 0.0, 'pi3k_inhibitor': 0.0, 'ras_inhibitor': 0.5}
    }
]

class SignalingDataset(Dataset):
    """
    Standard PyTorch dataset for experimental signaling data.
    Maps (time, drugs) -> protein_expression.
    """
    def __init__(self, t: torch.Tensor, drugs: torch.Tensor, y: torch.Tensor):
        self.t = t
        self.drugs = drugs
        self.y = y

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.t[idx], self.drugs[idx], self.y[idx]

def prepare_training_tensors(
    train_until_hour: float = 48.0, 
    condition_name: Optional[str] = None,
    split_mode: str = "holdout",
    holdout_timepoints: Optional[List[float]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, torch.Tensor]]:
    """
    Prepares training and testing tensors from raw experimental data.

    Args:
        train_until_hour: Cutoff time for training (used in 'cutoff' mode).
        condition_name: If provided, filters to only this condition.
        split_mode: 'cutoff' = train on t <= train_until_hour,
                    'holdout' = leave specific timepoints out for testing.
        holdout_timepoints: List of timepoints (in hours) to hold out for testing.
                           Only used when split_mode='holdout'. Default: [24.0].
        
    Returns:
        train_data: Dict with keys 't', 'drugs', 'y_norm', 'y_raw'
        test_data: Dict with keys 't', 'drugs', 'y_norm', 'y_raw'
        scalers: Dict with normalization factors
    """
    if holdout_timepoints is None:
        holdout_timepoints = [24.0]
    
    all_t, all_y, all_drugs = [], [], []
    
    experiments = TRAINING_DATA_LIST
    if condition_name:
        experiments = [e for e in TRAINING_DATA_LIST if e['name'] == condition_name]
        if not experiments:
            raise ValueError(f"Condition '{condition_name}' not found.")
    
    for exp in experiments:
        t_points = exp['time_points'].astype(np.float32)
        num_pts = len(t_points)
        
        y_exp = np.zeros((num_pts, 10), dtype=np.float32)
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
    
    t_max = 48.0
    
    if split_mode == "holdout":
        holdout_set = set(holdout_timepoints)
        test_mask = np.array([t in holdout_set for t in t_data])
        train_mask = ~test_mask
    else:
        train_mask = t_data <= train_until_hour
        test_mask = ~train_mask
    
    y_train = y_data[train_mask]
    y_min = np.min(y_train, axis=0)
    y_max = np.max(y_train, axis=0)
    y_range = y_max - y_min
    y_range[y_range == 0] = 1.0
    
    y_norm_all = (y_data - y_min) / y_range
    y_norm_all = np.nan_to_num(y_norm_all, nan=0.0, posinf=1.0, neginf=0.0)
    
    def package_data(mask):
        return {
            't': t_data[mask],
            't_norm': (t_data[mask] / t_max).reshape(-1, 1),
            'drugs': drugs_data[mask],
            'y_norm': y_norm_all[mask],
            'y_raw': y_data[mask]
        }
    
    train_data = package_data(train_mask)
    test_data = package_data(test_mask)
    
    scalers = {
        'y_mean': torch.tensor(y_min, dtype=torch.float32),
        'y_std': torch.tensor(y_range, dtype=torch.float32),
        't_range': torch.tensor(t_max, dtype=torch.float32)
    }
    
    return train_data, test_data, scalers

def get_collocation_points(num_points: int = 2000, basal_fraction: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates physics collocation points covering the temporal domain [0, 1] (normalized).
    Drug concentrations are sampled from actual training conditions with
    small perturbations to ensure physics is enforced near real data manifolds.
    
    A guaranteed fraction of points are pure no-drug (basal) to enforce
    the steady-state constraint: dX/dt ≈ 0 when no drug is applied.

    Args:
        num_points: Total number of points to generate.
        basal_fraction: Fraction of points guaranteed to be zero-drug (default 0.2).
        
    Returns:
        t_collocation: Normalized temporal points (num_points, 1).
        drugs_collocation: Drug concentration points (num_points, 4).
    """
    # 1. Sample random time points across the entire domain
    t_collocation = np.random.uniform(0, 1.0, size=(num_points, 1)).astype(np.float32)
    
    # 2. Extract base drug conditions from training data
    base_conditions = np.array([
        [exp['drugs']['vemurafenib'], exp['drugs']['trametinib'],
         exp['drugs']['pi3k_inhibitor'], exp['drugs']['ras_inhibitor']]
        for exp in TRAINING_DATA_LIST
    ], dtype=np.float32)
    
    # 3. Calculate counts for basal vs treated points
    num_basal = int(num_points * basal_fraction)
    num_treated = num_points - num_basal
    
    # 4. Generate treated points with perturbations
    sample_idx = np.random.choice(len(base_conditions), size=num_treated)
    treated_drugs = base_conditions[sample_idx].copy()
    
    # Add Gaussian jitter and clip to ensure positive concentrations
    jitter = np.random.normal(0, 0.05, size=(num_treated, 4)).astype(np.float32)
    treated_drugs = np.clip(treated_drugs + jitter, 0, None)
    
    # 5. Generate basal (no-drug) points
    basal_drugs = np.zeros((num_basal, 4), dtype=np.float32)
    
    # 6. Combine, shuffle and convert to tensors
    drugs_collocation = np.concatenate([basal_drugs, treated_drugs], axis=0)
    
    shuffle_idx = np.random.permutation(num_points)
    t_collocation = t_collocation[shuffle_idx]
    drugs_collocation = drugs_collocation[shuffle_idx]
    
    return torch.tensor(t_collocation), torch.tensor(drugs_collocation)

