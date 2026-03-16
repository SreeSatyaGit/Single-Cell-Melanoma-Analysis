from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class ModelConfig:
    input_size: int = 5
    hidden_size: int = 256
    output_size: int = 10
    num_hidden_layers: int = 6

@dataclass
class LossWeights:
    data: float = 10.0
    physics: float = 1.0
    boundary: float = 5.0
    conservation: float = 0.1
    sparsity: float = 0.001
    steady_state: float = 2.0

@dataclass
class TrainingConfig:
    experiment_name: str = "PINN_Global_Model"
    output_dir: str = "results/nature_submission"
    seed: int = 42
    
    train_until_hour: float = 48.0
    split_mode: str = "holdout"
    holdout_timepoints: List[float] = field(default_factory=lambda: [4.0, 24.0])
    
    num_epochs: int = 5000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_decay_gamma: float = 0.9
    lr_decay_step: int = 1000
    
    num_physics_points: int = 2000
    
    model: ModelConfig = field(default_factory=ModelConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "split_mode": self.split_mode,
            "train_until_hour": self.train_until_hour,
            "holdout_timepoints": self.holdout_timepoints,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "physics_points": self.num_physics_points,
        }
