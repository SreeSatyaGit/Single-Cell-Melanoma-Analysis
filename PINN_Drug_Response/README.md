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
1. Train PINN on configured time range
2. Evaluate extrapolation if test points are held out
3. Generate visualizations and predictions

### Output Files

- `pinn_model_best.pth` - Best model checkpoint
- `extrapolation_results.png` - Training fit + test extrapolation
- `training_test_history.png` - Train vs test loss curves
- `predictions_table.csv` - Detailed predictions with errors
- `training_history.csv` - Loss values per epoch
- `training_config.json` - Hyperparameters

### Visualization Only

If you already have a trained model:

```bash
python visualize_extrapolation.py
```

## Model Architecture

**Input Layer (5 features):**
- Time (normalized)
- Vemurafenib concentration
- Trametinib concentration
- PI3K inhibitor concentration
- RAS inhibitor concentration

**Hidden Layers:**
- 4 hidden layers × 100 neurons
- Activation: Hyperbolic tangent (tanh)

**Output Layer (11 species):**
- All phosphorylated proteins
- Activation: Softplus (ensures positivity)

## Loss Function

```
Total Loss = λ₁·L_data + λ₂·L_physics + λ₃·L_boundary + λ₄·L_conservation
```

**Components:**
1. **Data Loss**: MSE between predictions and experimental data (training points only)
2. **Physics Loss**: ODE residuals (∂y/∂t - f(y, drugs, k))
3. **Boundary Loss**: Initial condition enforcement at t=0
4. **Conservation Loss**: Biological constraints (non-negativity, pathway balance)

**Default Weights:**
- Data: 1.0, Physics: 0.5, Boundary: 0.3, Conservation: 0.2

## Physics Constraints

The model enforces these biological ODEs:

**MAPK:**
```
d(pMEK)/dt = k1·pCRAF·(1-Tram)·(1-AKT_inh) + AKT_relief - k2·pMEK
d(pERK)/dt = k3·pMEK·(1-DUSP6) - k4·pERK
d(DUSP6)/dt = k5·pERK²/(Km²+pERK²) - k6·DUSP6
```

**PI3K:**
```
d(pAKT)/dt = k7·RTK·(1-PI3Ki) + RAF_feedfwd - k8·pAKT - mTOR_feedback
d(pS6K)/dt = k9·pAKT·(1+k·p4EBP1) - k10·pS6K
d(p4EBP1)/dt = k11·pAKT - k12·p4EBP1
```

All rate constants (k1-k14, k_cat) are **learned during training** alongside neural network weights.

## Interpreting Results

### Good Extrapolation Indicators:
- **High R² on test set (>0.7)**: Model captures temporal dynamics
- **Smooth predictions**: Physics constraints prevent overfitting
- **Low train-test gap**: Generalization to unseen time points

### Poor Extrapolation Indicators:
- **Oscillations at 24-48hrs**: Overfitting or weak physics constraints
- **R² < 0.5 on test**: Model hasn't learned dynamics
- **Large train-test gap**: Need stronger regularization

## Key Features

1. **Train/Test Temporal Split**: Validates extrapolation capability
2. **Physics-Informed**: Not just curve fitting—enforces biological laws
3. **Learnable Rate Constants**: Infers kinetic parameters from data
4. **Crosstalk Modeling**: Captures compensatory drug resistance mechanisms
5. **Real-Time Monitoring**: Tracks both train and test loss during training

## Hyperparameter Tuning

Edit `main.py` config dictionary:

```python
config = {
    'num_epochs': 5000,        # Increase if underfitting
    'learning_rate': 0.001,    # Typical range: [1e-4, 1e-2]
    'hidden_size': 100,        # Try [50, 100, 200]
    'num_physics_points': 100, # More = stronger physics
    'weights': {
        'physics': 0.5,        # Increase to enforce ODEs more
        'data': 1.0,
    }
}
```

## Next Steps

1. **Current Focus**: Vem+Tram extrapolation → validate model
2. **Future**: Predict new drug combinations (PI3Ki+Vem)
3. **Advanced**: Uncertainty quantification with ensembles

## Citation

If you use this code, please cite the underlying experimental data source and acknowledge the PINN methodology.

## Contact

For questions about this implementation, contact the project maintainer.
