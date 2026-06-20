import os
import argparse
import yaml
import torch
from datasets import get_datasets
from models.hybrid_model import AdaptiveExplainableHybridModel
from training.engine import train_stage

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_configs(base_config, stage_config):
    """
    Recursively merges stage config into base config.
    """
    merged = base_config.copy()
    for k, v in stage_config.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Stage Waste Classification & Segmentation Trainer")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to base config")
    parser.add_argument("--dry-run", action="store_true", help="Run a quick 1-epoch dry run on synthetic data")
    parser.add_argument("--stage", type=str, default="all", choices=["stage1", "stage2", "stage3", "all"], help="Stage to run")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load base configuration
    base_config = load_config(args.config)
    
    # Apply dry-run updates if active
    if args.dry_run:
        print("!!! DRY RUN MODE ACTIVE !!!")
        base_config['dataset']['synthetic'] = True
        base_config['training']['epochs'] = 1
        base_config['scheduler']['warmup_epochs'] = 0
        base_config['training']['early_stopping_patience'] = 1
        base_config['xai']['sample_interval'] = 1
        base_config['xai']['eval_num_samples'] = 2
        
    device = torch.device(base_config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize the base hybrid model
    model = AdaptiveExplainableHybridModel(base_config)
    
    stages_to_run = []
    if args.stage == "all":
        stages_to_run = ["stage1", "stage2", "stage3"]
    else:
        stages_to_run = [args.stage]
        
    global_epoch_counter = 0
    
    # Stage 1: TrashNet
    if "stage1" in stages_to_run:
        stage_cfg_path = "configs/stage1_trashnet.yaml"
        stage_cfg = load_config(stage_cfg_path)
        config = merge_configs(base_config, stage_cfg)
        
        # In dry run, force synthetic and 1 epoch
        if args.dry_run:
            config['dataset']['synthetic'] = True
            config['training']['epochs'] = 1
            config['scheduler']['warmup_epochs'] = 0
            
        train_loader, val_loader, _ = get_datasets(config, stage="stage1")
        
        # Train Stage 1
        train_stage(model, "stage1", train_loader, val_loader, config, start_epoch=global_epoch_counter)
        global_epoch_counter += config['training']['epochs']
        
    # Stage 2: TACO
    if "stage2" in stages_to_run:
        stage_cfg_path = "configs/stage2_taco.yaml"
        stage_cfg = load_config(stage_cfg_path)
        config = merge_configs(base_config, stage_cfg)
        
        if args.dry_run:
            config['dataset']['synthetic'] = True
            config['training']['epochs'] = 1
            config['scheduler']['warmup_epochs'] = 0
            
        # Load weights from Stage 1 if we didn't just train it in the same run
        checkpoint_path_s1 = os.path.join(config['training']['checkpoint_dir'], f"{config['run_name']}_stage1_best.pt")
        if "stage1" not in stages_to_run and os.path.exists(checkpoint_path_s1):
            print(f"Loading weights from Stage 1 checkpoint: {checkpoint_path_s1}")
            checkpoint = torch.load(checkpoint_path_s1, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        train_loader, val_loader, _ = get_datasets(config, stage="stage2")
        
        # Train Stage 2
        train_stage(model, "stage2", train_loader, val_loader, config, start_epoch=global_epoch_counter)
        global_epoch_counter += config['training']['epochs']
        
    # Stage 3: ZeroWaste
    if "stage3" in stages_to_run:
        stage_cfg_path = "configs/stage3_zerowaste.yaml"
        stage_cfg = load_config(stage_cfg_path)
        config = merge_configs(base_config, stage_cfg)
        
        if args.dry_run:
            config['dataset']['synthetic'] = True
            config['training']['epochs'] = 1
            config['scheduler']['warmup_epochs'] = 0
            
        # Load weights from Stage 2 if we didn't just train it in the same run
        checkpoint_path_s2 = os.path.join(config['training']['checkpoint_dir'], f"{config['run_name']}_stage2_best.pt")
        if "stage2" not in stages_to_run and os.path.exists(checkpoint_path_s2):
            print(f"Loading weights from Stage 2 checkpoint: {checkpoint_path_s2}")
            checkpoint = torch.load(checkpoint_path_s2, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
        train_loader, val_loader, _ = get_datasets(config, stage="stage3")
        
        # Train Stage 3
        train_stage(model, "stage3", train_loader, val_loader, config, start_epoch=global_epoch_counter)
        global_epoch_counter += config['training']['epochs']
        
    print("\nTraining workflow completed successfully!")

if __name__ == "__main__":
    main()
