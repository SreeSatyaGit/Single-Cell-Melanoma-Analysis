# Physics-Informed Neural Network (PINN) for Cancer Signaling

This project implements a PINN in PyTorch to predict drug combination responses in cancer signaling pathways (MAPK and PI3K). The model is trained on experimental Western Blot data and constrained by biological Ordinary Differential Equations (ODEs).

## Project Structure

- `pinn_model.py`: Architecture of the PINN (4 hidden layers, tanh activation, softplus output).
- `physics_utils.py`: Biological ODE residuals and physics-informed loss computation.
- `data_utils.py`: Data preprocessing, normalization, and custom Dataset classes.
- `train_pinn.py`: Training loop with multi-component loss and learning rate scheduling.
- `inference.py`: Prediction logic and visualization functions.
- `main.py`: Main script to execute the full pipeline.
- `requirements.txt`: Project dependencies.

## Biological Context

The model tracks 11 signaling species across MAPK and PI3K pathways, including crosstalk and feedback loops:
- **MAPK**: RAS -> RAF -> MEK(pMEK) -> ERK(pERK)
- **PI3K**: PI3K -> AKT(pAKT) -> mTOR -> S6K(pS6K) / 4EBP1(p4EBP1)
- **Feedback**: ERK induces DUSP6 which inhibits MEK/ERK.
- **Crosstalk**: AKT inhibits RAF; ERK inhibits PI3K signaling.

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the model and generate predictions:
```bash
python main.py
```

## Outputs

- `pinn_model_best.pth`: The trained model checkpoint.
- `training_history.csv`: Loss values per epoch.
- `training_history.png`: Plots of total, data, and physics losses.
- `model_fit.png`: Comparison of model predictions vs. experimental training data.
- `predictions_pi3ki_vem.csv`: Predictions for the new drug combination (Vemurafenib + PI3K inhibitor).
- `prediction_comparison.png`: Side-by-side comparison of training condition vs. new drug condition.
- `training_config.json`: Hyperparameters used for the run.

## Physics Constraints

The model uses `torch.autograd` to compute time derivatives $\partial y / \partial t$ and minimizes the residual:
$$ \mathcal{L}_{physics} = \mathbb{E} \| \frac{\partial y}{\partial t} - f(y, u, k) \|^2 $$
where $f$ represents the mechanistic biological ODEs.
