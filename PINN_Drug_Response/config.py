from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class ModelConfig:
    input_size: int = 5
    hidden_size: int = 256
    output_size: int = 11
    num_hidden_layers: int = 6

@dataclass
class LossWeights:
    data: float = 10.0
    physics: float = 1.0
    boundary: float = 50.0
    conservation: float = 0.1
    sparsity: float = 0.001

@dataclass
class TrainingConfig:
    # Experiment
    experiment_name: str = "PINN_Global_Model"
    output_dir: str = "results/nature_submission"
    seed: int = 42
    
    # Data
    train_until_hour: float = 48.0
    
    # Optimization
    num_epochs: int = 50000
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_decay_gamma: float = 0.9
    lr_decay_step: int = 5000
    
    # Physics
    num_physics_points: int = 2000  # Increased for robust integration
    
    # Sub-configs
    model: ModelConfig = field(default_factory=ModelConfig)
    weights: LossWeights = field(default_factory=LossWeights)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "train_until_hour": self.train_until_hour,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "physics_points": self.num_physics_points,
            # Add other fields as necessary for logging
        }
