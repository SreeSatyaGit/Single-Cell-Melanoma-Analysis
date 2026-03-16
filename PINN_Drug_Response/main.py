import os
import torch
import numpy as np
from typing import Dict, Any
from train_pinn import train_pinn
from visualize_extrapolation import plot_extrapolation_results, plot_training_history, generate_prediction_table
from data_utils import TRAINING_DATA_LIST
from config import TrainingConfig
from utils import set_seed, setup_logger
def main():
    set_seed(42)
    config = TrainingConfig()
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger(log_file=f"{config.output_dir}/main_pipeline.log")
    logger.info("="*60)
    logger.info("PINN TRAINING PIPELINE: GLOBAL MODEL")
    logger.info("Objective: Train a single generalized model on all drug conditions.")
    logger.info("="*60)
    model, k_params, history, scalers, train_data, test_data = train_pinn(config, condition_name=None)
    model_path = f"{config.output_dir}/pinn_model_global.pth"
    history_path = f"{config.output_dir}/history_global.csv"
    logger.info("\n" + "="*60)
    logger.info("GENERATING ANALYSIS PER CONDITION")
    logger.info("="*60)
    if os.path.exists(model_path):
        plot_training_history(
            history_file=history_path,
            save_path=f"{config.output_dir}/history_global.png"
        )
        for exp in TRAINING_DATA_LIST:
            condition = exp['name']
            cond_id = condition.replace(" ", "_").replace("(", "").replace(")", "")
            logger.info(f"Analyzing condition: {condition}")
            drugs_override = exp['drugs']
            plot_extrapolation_results(
                model_path=model_path,
                save_path=f"{config.output_dir}/fit_{cond_id}.png",
                drugs_dict_override=drugs_override
            )
            generate_prediction_table(
                model_path=model_path,
                save_path=f"{config.output_dir}/table_{cond_id}.csv"
            )
        logger.info(f"\n✓ Pipeline complete. All artifacts saved to {config.output_dir}/")
    else:
        logger.error("Global model training failed - checkpoint not found.")
if __name__ == "__main__":
    main()
    print("PIPELINE COMPLETE: All conditions trained and analyzed.")
    print("="*60)
