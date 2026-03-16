# Physics-Informed Neural Networks (PINN) for Cancer Signaling Dynamics

## Overview

This repository provides a framework for modeling cancer signaling pathway dynamics using **Physics-Informed Neural Networks (PINNs)**. The model integrates experimental measurements with mechanistic Ordinary Differential Equations (ODEs) to predict the response of melanoma cells to various drug combinations.

The framework is specifically designed to handle **temporal and drug-combination extrapolation**, leveraging biological constraints to generalize beyond the training distribution.

## Key Features

- **Mechanistic Integration**: Embeds 10-species ODE systems (MAPK & PI3K/AKT pathways) directly into the neural network loss function.
- **Global Modeling**: A single model trained across multiple drug conditions (Vemurafenib, Trametinib, PI3Ki, panRASi) to learn universal kinetic parameters.
- **PINN-ODE Hybrid Perturbation**: A robust two-phase simulation framework:
  1. **Phase 1 (PINN)**: Accurately predicts response under known training conditions.
  2. **Phase 2 (ODE)**: Uses learned physics to simulate novel drug perturbations (e.g., adding a third drug at a specific time point).
- **Dose-Response Mapping**: Tools for generating in-silico dose-response surfaces and steady-state analyses.

## Signaling Model

The biological network modeled includes:
- **MAPK Pathway**: RTK → RAS → pCRAF → pMEK → pERK
- **PI3K/AKT Pathway**: RTK → pAKT → p4EBP1
- **Feedback & Crosstalk**: 
  - Dual-specificity phosphatase (DUSP6) induction by ERK (negative feedback).
  - AKT-mediated inhibition of CRAF.
  - ERK-mediated inhibition of upstream signaling (SOS/RTK).
  - Paradoxical activation of RAF by BRAF inhibitors.

## Repository Structure

```text
.
├── pinn_model.py              # PINN architecture with drug-embedding layers
├── physics_utils.py           # Mechanistic ODE system and physics loss terms
├── data_utils.py              # Experimental data management and normalization
├── config.py                  # Dataclass-based training configurations
├── train_pinn.py              # Training loop with physics-informed constraints
├── main.py                    # Pipeline entry point for model training
├── perturbation_experiment.py # PINN-ODE hybrid simulation for novel perturbations
├── inference.py               # Model loading and custom prediction utilities
├── simulate_no_drug.py        # Steady-state analysis from initial conditions
├── simulate_dosages.py        # Systematic dose-sweep simulations (e.g., Vem + Tram)
├── visualize_extrapolation.py # Publication-quality fit diagnostics
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

## Getting Started

### Installation

```bash
pip install -r requirements.txt
```

### 1. Training the Global Model
To train the model on all experimental conditions and generate fit reports:
```bash
python main.py
```

### 2. Running a Perturbation Experiment
To simulate adding a PI3K inhibitor (0.5µM) at $t=80h$ after initial Vem+Tram treatment:
```bash
python perturbation_experiment.py --t_switch 80 --pi3k_dose 0.5 --t_end 200
```
This generates:
- **Phase 1 (PINN)**: Trajectory up to time of perturbation.
- **Phase 2 (ODE)**: Mechanistic extrapolation under the new drug condition.

### 3. Systematic Dose Analysis
To simulate the entire 6x5 dose-response grid for Vemurafenib and Trametinib:
```bash
python simulate_dosages.py
```

## Performance & Validation
The model tracks training loss across several components:
- **Data Loss**: Alignment with experimental Western Blot points.
- **Physics Loss**: Violation of the underlying ODE system.
- **Steady-State Loss**: Constraint for basal stability at $t=0$.

Results and model checkpoints are stored in `results/nature_submission/` (configurable).

## Citation
If you utilize this framework in your research, please cite our study on [insert publication link/DOI].
