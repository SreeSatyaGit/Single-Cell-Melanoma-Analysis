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
    split_mode: str = "partial_condition_holdout"
    holdout_timepoints: List[float] = field(default_factory=lambda: [4.0, 24.0])
    holdout_condition: str = "Vem + PI3Ki Combo"
    partial_condition_train_timepoints: List[float] = field(default_factory=lambda: [0.0, 4.0])
    
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
            "experiment_name":                    self.experiment_name,
            "output_dir":                         self.output_dir,
            "seed":                               self.seed,
            "split_mode":                         self.split_mode,
            "train_until_hour":                   self.train_until_hour,
            "holdout_timepoints":                 self.holdout_timepoints,
            "holdout_condition":                  self.holdout_condition,
            "partial_condition_train_timepoints": self.partial_condition_train_timepoints,
            "num_epochs":                         self.num_epochs,
            "learning_rate":                      self.learning_rate,
            "weight_decay":                       self.weight_decay,
            "lr_decay_gamma":                     self.lr_decay_gamma,
            "lr_decay_step":                      self.lr_decay_step,
            "num_physics_points":                 self.num_physics_points,
            "model": {
                "input_size":        self.model.input_size,
                "hidden_size":       self.model.hidden_size,
                "output_size":       self.model.output_size,
                "num_hidden_layers": self.model.num_hidden_layers,
            },
            "weights": {
                "data":         self.weights.data,
                "physics":      self.weights.physics,
                "boundary":     self.weights.boundary,
                "conservation": self.weights.conservation,
                "sparsity":     self.weights.sparsity,
                "steady_state": self.weights.steady_state,
            },
        }
