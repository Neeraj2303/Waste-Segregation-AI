import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tqdm
from training.losses import JointLoss
from training.lr_scheduler import CosineWarmupScheduler

# Try importing wandb (optional)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    class wandb:
        @staticmethod
        def log(*args, **kwargs): pass
        @staticmethod
        def init(*args, **kwargs): pass

def train_epoch(model, loader, optimizer, scaler, loss_fn, device, use_amp=True):
    """
    Train model for one epoch.
    Supports mixed precision (AMP) and accumulates metrics.
    """
    model.train()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_seg_loss = 0.0
    
    # Track classification accuracy
    correct_cls = 0
    total_cls = 0
    
    # Track segmentation metrics (basic pixel accuracy)
    correct_seg = 0
    total_seg = 0
    
    pbar = tqdm.tqdm(loader, desc="  Training", leave=False)
    for images, masks, labels in pbar:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision
        if use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss_dict = loss_fn(outputs, labels, masks)
                loss = loss_dict['loss']
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_dict = loss_fn(outputs, labels, masks)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            
        epoch_loss += loss.item()
        epoch_cls_loss += loss_dict['cls_loss']
        epoch_seg_loss += loss_dict['seg_loss']
        
        # Track accuracy if classification is active
        if 'cls_logits' in outputs:
            preds = torch.argmax(outputs['cls_logits'], dim=1)
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.size(0)
            
        # Track segmentation pixel accuracy if active
        if 'seg_logits' in outputs:
            seg_preds = torch.argmax(outputs['seg_logits'], dim=1)
            # Exclude background in pixel accuracy or count all
            correct_seg += (seg_preds == masks).sum().item()
            total_seg += masks.numel()
            
        pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    n_batches = len(loader)
    metrics = {
        'loss': epoch_loss / n_batches,
        'cls_loss': epoch_cls_loss / n_batches,
        'seg_loss': epoch_seg_loss / n_batches,
        'cls_acc': correct_cls / max(1, total_cls),
        'seg_pixel_acc': correct_seg / max(1, total_seg)
    }
    return metrics


@torch.no_grad()
def val_epoch(model, loader, loss_fn, device):
    """
    Validate model for one epoch.
    """
    model.eval()
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_seg_loss = 0.0
    
    correct_cls = 0
    total_cls = 0
    
    correct_seg = 0
    total_seg = 0
    
    pbar = tqdm.tqdm(loader, desc="  Validation", leave=False)
    for images, masks, labels in pbar:
        images = images.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss_dict = loss_fn(outputs, labels, masks)
        
        epoch_loss += loss_dict['loss'].item()
        epoch_cls_loss += loss_dict['cls_loss']
        epoch_seg_loss += loss_dict['seg_loss']
        
        # Track classification accuracy
        if 'cls_logits' in outputs:
            preds = torch.argmax(outputs['cls_logits'], dim=1)
            correct_cls += (preds == labels).sum().item()
            total_cls += labels.size(0)
            
        # Track segmentation pixel accuracy
        if 'seg_logits' in outputs:
            seg_preds = torch.argmax(outputs['seg_logits'], dim=1)
            correct_seg += (seg_preds == masks).sum().item()
            total_seg += masks.numel()
            
    n_batches = len(loader)
    metrics = {
        'loss': epoch_loss / n_batches,
        'cls_loss': epoch_cls_loss / n_batches,
        'seg_loss': epoch_seg_loss / n_batches,
        'cls_acc': correct_cls / max(1, total_cls),
        'seg_pixel_acc': correct_seg / max(1, total_seg)
    }
    return metrics


def train_stage(model, stage_name, train_loader, val_loader, config, start_epoch=0):
    """
    Executes a complete stage of multi-stage training (TrashNet, TACO, or ZeroWaste).
    Sets up optimizer, scheduler, logs to TensorBoard/WandB, implements early stopping.
    """
    print(f"\n=== Starting training stage: {stage_name} ===")
    
    # Create output folders
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Support Multi-GPU
    if torch.cuda.device_count() > 1 and config['device'] == 'cuda':
        print(f"INFO: Detected {torch.cuda.device_count()} GPUs. Wrapping model with DataParallel.")
        model = nn.DataParallel(model)
        is_dp = True
    else:
        is_dp = False
        
    # Set up Joint Loss
    loss_fn = JointLoss(config)
    
    # Optimizer and Scheduler
    lr = config['optimizer']['lr']
    weight_decay = config['optimizer']['weight_decay']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    epochs = config['training']['epochs']
    warmup_epochs = config['scheduler']['warmup_epochs']
    min_lr = config['scheduler']['min_lr']
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=warmup_epochs, max_epochs=epochs, min_lr=min_lr)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])
    
    # TensorBoard logger
    tb_writer = None
    if config['training']['tb_logging']:
        tb_writer = SummaryWriter(log_dir=os.path.join(config['training']['log_dir'], f"{config['run_name']}_{stage_name}"))
        
    # WandB Logger
    if config['training']['wandb_logging'] and HAS_WANDB:
        try:
            wandb.init(
                project=config['project_name'],
                name=f"{config['run_name']}_{stage_name}",
                config=config,
                reinit=True
            )
        except Exception as e:
            print(f"WARNING: Wandb initialization failed: {e}. Logging to wandb disabled.")
            
    best_val_loss = float('inf')
    early_stop_patience = config['training']['early_stopping_patience']
    no_improve_epochs = 0
    
    for epoch in range(1, epochs + 1):
        actual_epoch = start_epoch + epoch
        print(f"Epoch {epoch}/{epochs} (Global Epoch {actual_epoch}) [LR: {optimizer.param_groups[0]['lr']:.6f}]")
        
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, loss_fn, device, config['use_amp'])
        val_metrics = val_epoch(model, val_loader, loss_fn, device)
        
        # Step LR scheduler
        scheduler.step()
        
        # Print metrics
        print(f"  Train -> Loss: {train_metrics['loss']:.4f} | Cls Acc: {train_metrics['cls_acc']:.4f} | Seg Pixel Acc: {train_metrics['seg_pixel_acc']:.4f}")
        print(f"  Val   -> Loss: {val_metrics['loss']:.4f} | Cls Acc: {val_metrics['cls_acc']:.4f} | Seg Pixel Acc: {val_metrics['seg_pixel_acc']:.4f}")
        
        # Log to TensorBoard
        if tb_writer:
            for k, v in train_metrics.items():
                tb_writer.add_scalar(f"Train_{stage_name}/{k}", v, epoch)
            for k, v in val_metrics.items():
                tb_writer.add_scalar(f"Val_{stage_name}/{k}", v, epoch)
            tb_writer.add_scalar(f"LR_{stage_name}", optimizer.param_groups[0]['lr'], epoch)
            
        # Log to WandB
        if config['training']['wandb_logging'] and HAS_WANDB:
            wandb.log({
                f"{stage_name}/train_loss": train_metrics['loss'],
                f"{stage_name}/train_cls_loss": train_metrics['cls_loss'],
                f"{stage_name}/train_seg_loss": train_metrics['seg_loss'],
                f"{stage_name}/train_cls_acc": train_metrics['cls_acc'],
                f"{stage_name}/train_seg_acc": train_metrics['seg_pixel_acc'],
                f"{stage_name}/val_loss": val_metrics['loss'],
                f"{stage_name}/val_cls_loss": val_metrics['cls_loss'],
                f"{stage_name}/val_seg_loss": val_metrics['seg_loss'],
                f"{stage_name}/val_cls_acc": val_metrics['cls_acc'],
                f"{stage_name}/val_seg_acc": val_metrics['seg_pixel_acc'],
                f"{stage_name}/epoch": actual_epoch
            })
        
        # Model Checkpointing
        val_loss = val_metrics['loss']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            
            # Unwrap DataParallel model to save weights cleanly
            unwrapped_model = model.module if is_dp else model
            
            checkpoint_path = os.path.join(
                config['training']['checkpoint_dir'], 
                f"{config['run_name']}_{stage_name}_best.pt"
            )
            torch.save({
                'epoch': actual_epoch,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"  [+] Saved new best model checkpoint to {checkpoint_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"  [!] Early stopping triggered. Validation loss has not improved for {early_stop_patience} epochs.")
                break
                
    if tb_writer:
        tb_writer.close()
        
    return best_val_loss
