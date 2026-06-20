import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from datasets import get_datasets
from models.hybrid_model import AdaptiveExplainableHybridModel
from models.baselines import BaselineModel
from evaluation.evaluator import (
    evaluate_model, 
    plot_confusion_matrix, 
    plot_roc_curves, 
    plot_segmentation_predictions
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation & Generalization Suite")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--stage-checkpoint", type=str, default="stage2", choices=["stage1", "stage2", "stage3"], help="Which stage checkpoint to load for testing")
    parser.add_argument("--dry-run", action="store_true", help="Quick run with synthetic dataset")
    parser.add_argument("--run-baselines", action="store_true", help="Train and benchmark against baseline models")
    return parser.parse_args()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()
    config = load_config(args.config)
    
    if args.dry_run:
        config['dataset']['synthetic'] = True
        config['xai']['eval_num_samples'] = 2
        config['training']['epochs'] = 1
    else:
        config['dataset']['synthetic'] = False
        
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation runner target device: {device}")
    
    # 1. Instantiate our hybrid model
    model = AdaptiveExplainableHybridModel(config)
    
    # Map stage checkpoint shortcut to actual stage run name
    stage_to_run_name = {
        "stage1": "stage1_trashnet",
        "stage2": "stage2_taco",
        "stage3": "stage3_zerowaste"
    }
    run_name = stage_to_run_name.get(args.stage_checkpoint, config['run_name'])
    checkpoint_path = os.path.join(
        config['training']['checkpoint_dir'], 
        f"{run_name}_{args.stage_checkpoint}_best.pt"
    )
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Running with initialized weights.")
        
    model = model.to(device)
    
    # Output metrics container
    exp_results = []
    
    # ========================================================
    # Experiment 1: Train TrashNet -> Test TACO (Cross-Dataset Classification)
    # ========================================================
    print("\n--- Running Experiment 1: TrashNet Model -> TACO Test Dataset ---")
    # Load TACO dataset
    _, _, taco_test_loader = get_datasets(config, stage="stage2")
    taco_results = evaluate_model(model, taco_test_loader, device, num_classes=config['dataset']['num_classes'])
    
    if 'cls' in taco_results:
        metrics = taco_results['cls']
        print(f"  TACO Classification Accuracy: {metrics['accuracy']:.4f}")
        print(f"  TACO Macro F1-Score:           {metrics['f1_score']:.4f}")
        
        # Save plots
        plot_confusion_matrix(metrics['confusion_matrix'], config['dataset']['class_names'], "outputs/exp1_taco_confusion_matrix.png")
        plot_roc_curves(taco_results['cls_logits'], taco_results['cls_targets'], config['dataset']['class_names'], "outputs/exp1_taco_roc_curve.png")
        print("  [+] Visualizations exported to 'outputs/'")
        
        exp_results.append({
            "Experiment": "Exp 1: TrashNet -> TACO",
            "Cls Accuracy": metrics['accuracy'],
            "Cls F1-Score": metrics['f1_score'],
            "Seg mIoU": taco_results['seg']['miou'] if 'seg' in taco_results else np.nan,
            "Seg Dice": taco_results['seg']['mean_dice'] if 'seg' in taco_results else np.nan
        })

    # ========================================================
    # Experiment 2: Train TrashNet -> Test ZeroWaste (Cross-Dataset Segmentation)
    # ========================================================
    print("\n--- Running Experiment 2: TrashNet Model -> ZeroWaste Test Dataset ---")
    _, _, zw_test_loader = get_datasets(config, stage="stage3")
    zw_results = evaluate_model(model, zw_test_loader, device, num_classes=config['dataset']['num_classes'])
    
    if 'seg' in zw_results:
        metrics = zw_results['seg']
        print(f"  ZeroWaste Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        print(f"  ZeroWaste mIoU:           {metrics['miou']:.4f}")
        print(f"  ZeroWaste Mean Dice:      {metrics['mean_dice']:.4f}")
        
        # Save segmentation masks overlay
        samples = zw_results['samples']
        plot_segmentation_predictions(
            samples['images'], samples['masks'], samples['preds'],
            config['dataset']['class_names'], "outputs/exp2_zerowaste_segmentations.png"
        )
        print("  [+] Visualizations exported to 'outputs/'")
        
        exp_results.append({
            "Experiment": "Exp 2: TrashNet -> ZeroWaste",
            "Cls Accuracy": zw_results['cls']['accuracy'] if 'cls' in zw_results else np.nan,
            "Cls F1-Score": zw_results['cls']['f1_score'] if 'cls' in zw_results else np.nan,
            "Seg mIoU": metrics['miou'],
            "Seg Dice": metrics['mean_dice']
        })

    # ========================================================
    # Baseline Comparisons
    # ========================================================
    baseline_metrics = []
    # Include parameters count for our hybrid model
    hybrid_params = count_parameters(model)
    print(f"\nModel Parameter Count: {hybrid_params:,} parameters")
    
    baseline_metrics.append({
        "Model": "Ours (ViT + CNN Hybrid)",
        "Params": f"{hybrid_params / 1e6:.2f}M",
        "Cls Acc (TACO)": exp_results[0]['Cls Accuracy'] if len(exp_results) > 0 else 0.0,
        "Seg mIoU (ZeroWaste)": exp_results[1]['Seg mIoU'] if len(exp_results) > 1 else 0.0
    })
    
    if args.run_baselines:
        print("\n--- Running Baseline Benchmarking ---")
        baselines_list = ["resnet50", "vit_b16", "convnext", "segformer"]
        
        for name in baselines_list:
            print(f"Evaluating Baseline: {name}")
            try:
                base_model = BaselineModel(model_name=name, num_classes=config['dataset']['num_classes'], pretrained=False)
                base_model = base_model.to(device)
                
                params = count_parameters(base_model)
                
                # Run evaluation (with initial weights for dry runs)
                taco_res = evaluate_model(base_model, taco_test_loader, device, num_classes=config['dataset']['num_classes'])
                zw_res = evaluate_model(base_model, zw_test_loader, device, num_classes=config['dataset']['num_classes'])
                
                cls_acc = taco_res['cls']['accuracy'] if 'cls' in taco_res else 0.0
                seg_iou = zw_res['seg']['miou'] if 'seg' in zw_res else 0.0
                
                baseline_metrics.append({
                    "Model": name.upper(),
                    "Params": f"{params / 1e6:.2f}M",
                    "Cls Acc (TACO)": cls_acc,
                    "Seg mIoU (ZeroWaste)": seg_iou
                })
            except Exception as e:
                print(f"  Error evaluating baseline {name}: {e}")
                
        # Generate baseline dataframe and print
        df_baselines = pd.DataFrame(baseline_metrics)
        print("\n=== Baseline Comparison Summary Table ===")
        print(df_baselines.to_string(index=False))
        df_baselines.to_csv("outputs/baseline_comparison.csv", index=False)
        
    # Generate overall experiments summary
    if exp_results:
        df_exp = pd.DataFrame(exp_results)
        df_exp.to_csv("outputs/experiment_generalization_summary.csv", index=False)
        print("\n=== Generalization Experiments Table ===")
        print(df_exp.to_string(index=False))

    print("\nEvaluation completed. Results saved in 'outputs/' folder.")

if __name__ == "__main__":
    main()
