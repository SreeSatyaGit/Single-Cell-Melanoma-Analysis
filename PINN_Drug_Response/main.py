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
    # 1. Reproducibility
    set_seed(42)
    
    # 2. Configuration Setup
    # Using defaults defined in config.py which are optimized for publication-quality results
    config = TrainingConfig()
    
    # Ensure output directory exists (handled by logger but good practice)
    os.makedirs(config.output_dir, exist_ok=True)
    
    logger = setup_logger(log_file=f"{config.output_dir}/main_pipeline.log")
    logger.info("="*60)
    logger.info("PINN TRAINING PIPELINE: GLOBAL MODEL (Nature Submission Ready)")
    logger.info("Objective: Train a single generalized model on all drug conditions.")
    logger.info("="*60)
    
    # 3. Train the Global Model
    # condition_name=None automatically aggregates all data in data_utils
    model, k_params, history, scalers, train_data, test_data = train_pinn(config, condition_name=None)
    
    model_path = f"{config.output_dir}/pinn_model_global.pth"
    history_path = f"{config.output_dir}/history_global.csv"
    
    # 4. Generate Reproducible Figures
    logger.info("\n" + "="*60)
    logger.info("GENERATING ANALYSIS PER CONDITION")
    logger.info("="*60)
    
    if os.path.exists(model_path):
        # Overall History Plot
        plot_training_history(
            history_file=history_path, 
            save_path=f"{config.output_dir}/history_global.png"
        )
        
        # Individual Condition Fits (Extrapolation & Validation)
        for exp in TRAINING_DATA_LIST:
            condition = exp['name']
            cond_id = condition.replace(" ", "_").replace("(", "").replace(")", "")
            logger.info(f"Analyzing condition: {condition}")
            
            # Identify the correct drug vector from the dataset to ensure accurate plotting
            drugs_override = exp['drugs']
            
            # Generate Fit Plot
            plot_extrapolation_results(
                model_path=model_path, 
                save_path=f"{config.output_dir}/fit_{cond_id}.png",
                drugs_dict_override=drugs_override
            )
            
            # Generate Prediction Table
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
