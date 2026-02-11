# Physics-Informed Neural Network (PINN) for Cancer Signaling

## Project Overview

This project implements a Physics-Informed Neural Network (PINN) to predict drug combination responses in cancer signaling pathways (MAPK and PI3K). The model is trained on experimental Western Blot data and constrained by biological Ordinary Differential Equations (ODEs).

## Current Focus: Temporal Extrapolation

**Training Setup:**
- Drug Combination: **Vemurafenib (0.5) + Trametinib (0.3)**
- Training Data: Time points **[0, 1, 4, 8 hours]** (4 points)
- Test Data: Time points **[24, 48 hours]** (2 held-out points)
- Objective: Train on early time points and **extrapolate** to late time points

This experimental design tests the model's ability to learn biological dynamics from limited early observations and predict future states.

### Full-Range Training Option

To fit **all available time points (0–48h)** before running inference on new drug combinations, set:

```python
config = {
    'train_until_hour': 48,
    # other hyperparameters...
}
```

This removes the temporal holdout split and trains on the full dataset.

## Project Structure

```
PINN_Drug_Response/
│
├── pinn_model.py              # Neural network architecture
├── physics_utils.py           # ODE constraints and physics loss
├── data_utils.py              # Data loading with train/test split
├── train_pinn.py              # Training loop with extrapolation
├── visualize_extrapolation.py # Specialized visualization for train/test
├── main.py                    # Main execution script
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Biological Context

The model tracks 11 signaling species across MAPK and PI3K pathways:

**MAPK Pathway:**
- Receptors → RAS → RAF(pCRAF) → MEK(pMEK) → ERK(pERK)
- Negative feedback: ERK induces DUSP6 → inhibits ERK

**PI3K Pathway:**
- Receptors → PI3K → AKT(pAKT) → mTOR → S6K(pS6K) / 4EBP1(p4EBP1)
- mTOR feedback: S6K inhibits IRS1

**Crosstalk:**
- AKT inhibits RAF (pathway suppression)
- ERK inhibits PI3K (compensatory activation)
- RAF activates PI3K (paradoxical activation)

**Drug Mechanisms:**
- **Vemurafenib**: BRAF inhibitor (with paradoxical activation at low doses)
- **Trametinib**: MEK inhibitor

## Installation

```bash
cd /Users/bharadwajanandivada/SCMPA/PINN_Drug_Response
pip install -r requirements.txt
```

## Usage


### Basic Training and Extrapolation

```bash
python main.py
```

This will:
1. Train a global PINN model on all available drug conditions.
2. Evaluate performance on any held-out test data.
3. Generate comprehensive visualization plots for each condition.
4. Save the trained model and history to `results/nature_submission/`.

### Making Conceptions (Inference)

To predict pathway response to a **new custom drug combination** without retraining:

```bash
python inference.py --model results/nature_submission/pinn_model_global.pth \
  --vemurafenib 0.5 \
  --trametinib 0.1 \
  --pi3k 0.0 \
  --ras 0.0 \
  --output custom_prediction.png
```

This generates a plot and CSV file for the specified dosage.

### Output Files

Results are saved in `results/nature_submission/`:

- `pinn_model_global.pth` - Best model checkpoint.
- `history_global.png` - Training vs Test loss curves.
- `fit_{condition}.png` - Plots showing model fit vs experimental data.
- `table_{condition}.csv` - Detailed numerical predictions.

## Hyperparameter Tuning

Configuration is managed in `config.py` using dataclasses for type safety and clarity.

To modify training parameters, edit `config.py`:

```python
@dataclass
class TrainingConfig:
    # Experiment
    output_dir: str = "results/nature_submission"
    
    # Optimization
    num_epochs: int = 50000
    learning_rate: float = 1e-4
    
    # Physics
    num_physics_points: int = 2000
    
    # Weights are in the LossWeights class
    weights: LossWeights = field(default_factory=LossWeights)
```

## Next Steps

1. **Current Focus**: Vem+Tram extrapolation → validate model
2. **Future**: Predict new drug combinations (PI3Ki+Vem)
3. **Advanced**: Uncertainty quantification with ensembles

## Citation

If you use this code, please cite the underlying experimental data source and acknowledge the PINN methodology.

## Contact

For questions about this implementation, contact the project maintainer.
