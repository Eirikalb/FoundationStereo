#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
from omegaconf import OmegaConf
import torchvision.transforms as transforms
from PIL import Image
import random
import imageio
import cv2
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt


# Add parent directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# Import project modules
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from Utils import *
from Utils import vis_disparity
from dataloaders.middlebury_dataset import MiddleburyDataset
from dataloaders.middlebury2021_dataset import Middlebury2021Dataset
from dataloaders.kittistereo_dataset import KITTIStereoDataset
from dataloaders.booster_dataset import BoosterDataset
from dataloaders.layeredflow_dataset import LayeredFlowDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sequence_loss(init_disp, disp_preds, gt_disp, valid_mask, gamma=0.9):
    """
    Calculate the sequence loss over multiple disparity predictions
    Args:
        init_disp: Initial disparity prediction
        disp_preds: List of disparity predictions from the model
        gt_disp: Ground truth disparity
        valid_mask: Valid mask for the ground truth
        gamma: Weight decay for later predictions
    """
    n_predictions = len(disp_preds)
    flow_loss = 0.0
    
    # Print shapes for debugging
    print(f"init_disp shape: {init_disp.shape}")
    print(f"gt_disp shape: {gt_disp.shape}")
    print(f"valid_mask shape: {valid_mask.shape}")
    
    # Ensure all inputs have the correct shape
    if init_disp.shape != gt_disp.shape:
        # Downsample ground truth to match initial prediction
        gt_disp = F.interpolate(gt_disp, size=init_disp.shape[2:], mode='bilinear', align_corners=False)
        valid_mask = F.interpolate(valid_mask.float(), size=init_disp.shape[2:], mode='nearest') > 0.5
    
    # Convert boolean mask to float for loss calculation
    valid_mask = valid_mask.float()
    
    # Clip ground truth values to max_disp range
    max_disp = 416  # Maximum disparity value from args
    gt_disp = torch.clamp(gt_disp, min=0, max=max_disp)
    
    # Normalize disparity values to [0, 1] range
    init_disp = init_disp / max_disp
    gt_disp = gt_disp / max_disp
    disp_preds = [dp / max_disp for dp in disp_preds]
    
    # Print normalized ranges for debugging
    print(f"Normalized init_disp range: [{init_disp.min():.2f}, {init_disp.max():.2f}]")
    print(f"Normalized gt_disp range: [{gt_disp.min():.2f}, {gt_disp.max():.2f}]")
    
    # Calculate loss on initial prediction
    i_loss = F.smooth_l1_loss(init_disp[valid_mask > 0.5], gt_disp[valid_mask > 0.5], reduction='mean')
    flow_loss += i_loss
    
    # Loss on subsequent predictions with decaying weights
    for i in range(n_predictions):
        curr_pred = disp_preds[i]
        i_weight = gamma**(n_predictions - i - 1)
        
        # Ensure prediction matches ground truth shape
        if curr_pred.shape != gt_disp.shape:
            curr_pred = F.interpolate(curr_pred, size=gt_disp.shape[2:], mode='bilinear', align_corners=False)
        
        i_loss = F.smooth_l1_loss(curr_pred[valid_mask > 0.5], gt_disp[valid_mask > 0.5], reduction='mean')
        flow_loss += i_weight * i_loss
        
    # Average the losses
    flow_loss = flow_loss / (n_predictions + 1)
    
    # Check for NaN or infinite values
    if torch.isnan(flow_loss) or torch.isinf(flow_loss):
        print("Warning: Loss is NaN or infinite!")
        print(f"init_disp range: [{init_disp.min():.2f}, {init_disp.max():.2f}]")
        print(f"gt_disp range: [{gt_disp.min():.2f}, {gt_disp.max():.2f}]")
        print(f"valid_mask range: [{valid_mask.min():.2f}, {valid_mask.max():.2f}]")
        print(f"Number of valid pixels: {(valid_mask > 0.5).sum().item()}")
        print(f"Raw loss values: initial={i_loss.item():.4f}, final={flow_loss.item():.4f}")
    
    return flow_loss


def plot_training_batch(left, right, disp_gt, init_disp, disp_preds, padder, args):
    """Plot training batch visualization including input images, ground truth, predictions and error map.
    
    Args:
        left: Left input image tensor
        right: Right input image tensor
        disp_gt: Ground truth disparity tensor
        init_disp: Initial disparity prediction tensor
        disp_preds: List of disparity prediction tensors
        padder: InputPadder instance
        args: Training arguments
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Unpad predictions
    disp_preds = [padder.unpad(dp) for dp in disp_preds]
    init_disp = padder.unpad(init_disp)
    
    # Convert tensors to numpy for visualization
    left_np = left[0].cpu().permute(1, 2, 0).detach().numpy()
    right_np = right[0].cpu().permute(1, 2, 0).detach().numpy()
    disp_gt_np = disp_gt[0, 0].cpu().detach().numpy()
    init_disp_np = init_disp[0, 0].cpu().detach().numpy()
    final_disp_np = disp_preds[-1][0, 0].cpu().detach().numpy()
    
    # Handle zero values in ground truth
    disp_gt_np = np.where(disp_gt_np == 0, np.nan, disp_gt_np)
    error_map = np.abs(final_disp_np - disp_gt_np)
    
    # Get min/max values excluding nan/inf
    vmin = np.min(disp_gt_np[~np.isnan(disp_gt_np) & ~np.isinf(disp_gt_np)])
    vmax = np.max(disp_gt_np[~np.isnan(disp_gt_np) & ~np.isinf(disp_gt_np)])
    
    # Visualize images
    axes[0, 0].imshow(left_np)
    axes[0, 0].set_title('Left Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_np)
    axes[0, 1].set_title('Right Image')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(disp_gt_np, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Ground Truth Disparity')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(init_disp_np, cmap="turbo")
    axes[1, 0].set_title('Initial Predicted Disparity')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(final_disp_np, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title('Final Predicted Disparity')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(error_map, cmap='turbo')
    axes[1, 2].set_title('Error Map')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def train_epoch(model, train_loader, optimizer, epoch, args, writer=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    valid_loss_count = 0  # Count of valid (non-NaN) losses
    nan_loss_count = 0    # Count of NaN losses
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    
    # Set up scaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)
    
    for batch_idx, batch_data in enumerate(pbar):
        # Handle different dataset return formats
   
        # If dataset returns more values, take only what we need
        left, right, disp_gt = batch_data["im2"], batch_data["im3"], batch_data["gt"]
        has_disp = torch.ones(left.shape[0], dtype=torch.bool)  # Assume all samples have disparity

        # Ensure inputs are float tensors
        left = left.cuda().float()
        right = right.cuda().float()

        padder = InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        disp_gt = disp_gt.cuda().float()
        
        # Filter out samples without disparity
        valid_samples = torch.where(has_disp)[0]
        if len(valid_samples) == 0:
            continue
                # Visualize every N batches
        if batch_idx % args.log_interval == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(True):
                    init_disp, disp_preds = model(left, right, iters=args.train_iters)
            plot_training_batch(left, right, disp_gt, init_disp, disp_preds, padder, args)

        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(True):
            init_disp, disp_preds = model(left, right, iters=args.train_iters)
            
            # Unpad outputs
            disp_preds = [padder.unpad(dp) for dp in disp_preds]
            init_disp = padder.unpad(init_disp)
            
            # Create valid mask for loss calculation
            valid_mask = (disp_gt > 0) & (disp_gt < args.max_disp)
            
            # Calculate loss
            loss = sequence_loss(init_disp, disp_preds, disp_gt, valid_mask)
        
        # Check for NaN loss
        if torch.isnan(loss):
            nan_loss_count += 1
            logging.warning(f"NaN loss detected at batch {batch_idx}")
            continue
            
        # Backward pass and optimize with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Log statistics
        total_loss += loss.item()
        valid_loss_count += 1
        
        # Update progress bar with both valid and NaN loss counts
        pbar.set_postfix({
            "Loss": loss.item() if not torch.isnan(loss) else "NaN",
            "Valid": valid_loss_count,
            "NaN": nan_loss_count
        })

    # Calculate average loss only from valid samples
    avg_loss = total_loss / valid_loss_count if valid_loss_count > 0 else float('inf')
    
    # Log final epoch statistics
    if writer:
        wandb.log({
            'Train/Epoch_Avg_Loss': avg_loss,
            'Train/Epoch_Valid_Loss_Count': valid_loss_count,
            'Train/Epoch_NaN_Loss_Count': nan_loss_count,
            'Train/Epoch_Valid_Loss_Ratio': valid_loss_count / (valid_loss_count + nan_loss_count) if (valid_loss_count + nan_loss_count) > 0 else 0,
            'epoch': epoch
        })
    
    return avg_loss


def validate(model, val_loader, epoch, args, writer=None):
    """Validate the model"""
    total_loss = 0
    total_epe = 0
    total_d1 = 0
    count = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} Validation")
        for batch_idx, batch_data in enumerate(pbar):
            # Handle different dataset return formats
            left, right, disp_gt = batch_data["im2"], batch_data["im3"], batch_data["gt"]
            has_disp = torch.ones(left.shape[0], dtype=torch.bool)  # Assume all samples have disparity
            
            # Ensure inputs are float tensors
            left = left.cuda().float()
            right = right.cuda().float()
            disp_gt = disp_gt.cuda().float()
            # Filter out samples without disparity
            valid_samples = torch.where(has_disp)[0]
            if len(valid_samples) == 0:
                continue
                
            # Pad inputs if needed
            padder = InputPadder(left.shape, divis_by=32)
            left, right = padder.pad(left, right)
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                disp_pred = model(left, right, iters=args.valid_iters)
            
            # Unpad outputs
            disp_pred = padder.unpad(disp_pred)
            
            # Create valid mask for loss calculation
            valid_mask = (disp_gt > 0) & (disp_gt < args.max_disp)
            
            # Calculate metrics
            epe = F.l1_loss(disp_pred[valid_mask], disp_gt[valid_mask], reduction='mean')
            d1 = (torch.abs(disp_pred[valid_mask] - disp_gt[valid_mask]) > 3).float().mean()
            
            # Accumulate metrics
            total_epe += epe.item()
            total_d1 += d1.item()
            count += 1
            
            pbar.set_postfix({"EPE": epe.item(), "D1": d1.item()})
    
    # Calculate averages
    avg_epe = total_epe / count if count > 0 else float('inf')
    avg_d1 = total_d1 / count if count > 0 else float('inf')
    
    # Log metrics
    if writer:
        wandb.log({
            'Val/EPE': avg_epe,
            'Val/D1': avg_d1,
            'epoch': epoch
        })
    
    return avg_epe, avg_d1


def save_checkpoint(model, optimizer, epoch, epe, d1, is_best, args):
    """Save model checkpoint"""
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epe': epe,
        'd1': d1,
        'global_step': epoch * args.steps_per_epoch
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_ep{epoch}.pth')
    torch.save(state, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
        torch.save(state, best_path)
        
    # Save config
    cfg_path = os.path.join(args.checkpoint_dir, 'cfg.yaml')
    OmegaConf.save(config=args, f=cfg_path)
    
    logging.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train FoundationStereo')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=False, default="/nfs/data/eirik/stereoanywhere_assets/datasets/booster/train", help='path to dataset directory') # '/nfs/data/eirik/stereoanywhere_assets/datasets/booster/train' "/nfs/data/eirik/stereoanywhere_assets/datasets/data"
    parser.add_argument('--dataset_type', type=str, default='booster', help='dataset type (middlebury, middlebury2021, kitti_stereo, booster, layeredflow)')
    parser.add_argument('--val_percent', type=float, default=0.1, help='percentage of data for validation')
    
    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default='pretrained_models/23-51-11/model_best_bp2.pth', help='path to pretrained model')
    parser.add_argument('--max_disp', type=int, default=416, help='maximum disparity')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--train_iters', type=int, default=12, help='number of iterations during training')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of iterations during evaluation')
    
    # Data augmentation parameters
    parser.add_argument('--do_augmentation', action='store_true', default=False, help='apply data augmentation')
    parser.add_argument('--crop_height', type=int, default=300//2, help='random crop height')
    parser.add_argument('--crop_width', type=int, default=500//2, help='random crop width')

    
    # System parameters
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='save interval')
    parser.add_argument('--mixed_precision', action='store_true', default=True, 
                        help='use mixed precision training (required for FlashAttention)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16'],
                        help='data type for mixed precision (fp16 or bf16)')
    parser.add_argument('--wandb_project', type=str, default='foundation-stereo', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default='eirikalb', help='wandb entity name')
    
    # Architecture parameters (matching those in run_demo.py)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128], help='hidden dimensions')
    parser.add_argument('--n_downsample', type=int, default=2, help='number of downsample layers')
    parser.add_argument('--n_gru_layers', type=int, default=3, help='number of GRU layers')
    parser.add_argument('--corr_levels', type=int, default=2, help='number of correlation levels')
    parser.add_argument('--corr_radius', type=int, default=4, help='correlation radius')
    
    # Add new argument for freezing depth_anything
    parser.add_argument('--freeze_depth_anything', action='store_true', default=True, 
                        help='freeze the depth_anything part of the model')
    
    args = parser.parse_args()

    # Load pretrained model config if specified
    if args.pretrained_model is not None:
        pretrained_dir = os.path.dirname(args.pretrained_model)
        cfg_path = os.path.join(pretrained_dir, 'cfg.yaml')
        if os.path.exists(cfg_path):
            logging.info(f"Loading pretrained model config from {cfg_path}")

            cfg = OmegaConf.load(cfg_path)
            for k in args.__dict__:
                cfg[k] = args.__dict__[k]
            args = OmegaConf.create(cfg)
            logging.info(f"args:\n{args}")
            logging.info(f"Using pretrained model from {pretrained_dir}")
        else:
            logging.warning(f"No cfg.yaml found in pretrained model directory {pretrained_dir}")
    
    # Set seed
    set_seed(args.seed)
    
    # Make sure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.checkpoint_dir, 'training.log'))
        ]
    )
    
    # Initialize wandb
    logging.info("Training arguments:")
    for arg, value in sorted(vars(args).items()):
        logging.info(f"  {arg}: {value}")

    # Convert OmegaConf to regular dict for wandb
    wandb_config = OmegaConf.to_container(args, resolve=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=wandb_config,
        name=f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )    

    # Log model architecture
    model = FoundationStereo(args)
    wandb.watch(model)
    
    # Create train dataset
    aug_params = {
        'crop_size': (args.crop_height, args.crop_width),
        'do_flip': False,
        'min_scale': 0.1,
        'max_scale': .3,
        
    } if args.do_augmentation else None

    # Verify that the data directory exists
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    logging.info(f"Loading {args.dataset_type} dataset from {args.data_dir}")
    
    try:
        if args.dataset_type == 'middlebury':
            train_dataset = MiddleburyDataset(
                datapath=args.data_dir,
                aug_params=aug_params,
                test=False,
                mono=None,
                scale_factor=1.0
            )
        elif args.dataset_type == 'middlebury2021':
            train_dataset = Middlebury2021Dataset(
                datapath=args.data_dir,
                aug_params=aug_params,
                test=False,
                mono=None,
                scale_factor=1.0
            )
        elif args.dataset_type in ['kitti_stereo', 'kitti2015', 'kitti2012']:
            train_dataset = KITTIStereoDataset(
                datapath=args.data_dir,
                aug_params=aug_params,
                test=False,
                mono=None,
                scale_factor=1.0
            )
        elif args.dataset_type == 'booster':
            train_dataset = BoosterDataset(
                datapath=args.data_dir,
                aug_params=aug_params,
                test=False,
                mono=None,
                scale_factor=1.0
            )
        elif args.dataset_type == 'layeredflow':
            train_dataset = LayeredFlowDataset(
                datapath=args.data_dir,
                aug_params=aug_params,
                test=False,
                mono=None,
                scale_factor=1.0
            )
        else:
            raise ValueError(f'Unknown dataset type: {args.dataset_type}')
        
        logging.info(f"Successfully loaded dataset with {len(train_dataset)} samples")
        
        # Print a few sample paths to verify
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            logging.info(f"First sample loaded successfully")
    
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise
    
    # Create validation dataset (either from split or separate dir)
    val_dir = os.path.join(args.data_dir, 'val')
    if os.path.exists(val_dir):
        # Use separate validation folder if it exists
        logging.info(f"Using separate validation directory: {val_dir}")
        try:
            val_dataset = train_dataset.__class__(
                datapath=val_dir,
                aug_params={
                    'crop_size': (args.crop_height, args.crop_width),
                    'do_flip': False,  # No flipping for validation
                },
                test=True,
                mono=None,
                scale_factor=1.0
            )
            logging.info(f"Validation dataset loaded with {len(val_dataset)} samples")
        except Exception as e:
            logging.error(f"Error loading validation dataset: {str(e)}")
            raise
    else:
        # Split training data for validation
        logging.info(f"No separate validation directory found. Splitting training data for validation.")
        train_size = int(len(train_dataset) * (1 - args.val_percent))
        val_size = len(train_dataset) - train_size
        
        logging.info(f"Creating validation split: {train_size} training samples, {val_size} validation samples")
        
        # Create a temporary dataset with validation transforms
        try:
            temp_val_dataset = train_dataset.__class__(
                datapath=args.data_dir,
                aug_params={
                    'crop_size': (args.crop_height, args.crop_width),
                    'do_flip': False,  # No flipping for validation
                },
                test=True,
                mono=None,
                scale_factor=1.0
            )
            
            # Create random indices for splitting
            indices = list(range(len(train_dataset)))
            random.shuffle(indices)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            # Split datasets
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(temp_val_dataset, val_indices)
            
            logging.info(f"Dataset split complete")
        except Exception as e:
            logging.error(f"Error creating validation split: {str(e)}")
            raise

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Calculate steps per epoch for checkpoint saving
    args.steps_per_epoch = len(train_loader)
    
    # Initialize model with correct dtype
    model = FoundationStereo(args)
    
   
    model.cuda()
    
    # Load pretrained model if specified
    if args.pretrained_model is not None:
        logging.info(f"Loading pretrained model from {args.pretrained_model}")
        ckpt = torch.load(args.pretrained_model)
        model.load_state_dict(ckpt['model'], strict=False)
    
    # Freeze depth_anything part if requested
    if args.freeze_depth_anything:
        logging.info("Freezing depth_anything part of the model")
        for name, param in model.named_parameters():
            if 'depth_anything' in name:
                param.requires_grad = False
                logging.debug(f"Freezing parameter: {name}")
    
    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Set up optimizer (only for trainable parameters)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=args.lr, weight_decay=args.weight_decay)
    
    # LR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,
        cycle_momentum=False,
        anneal_strategy='cos'
    )
    
    # Initialize best metrics
    best_epe = float('inf')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f"Starting epoch {epoch}/{args.epochs}")
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, epoch, args)
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Validate
        val_epe, val_d1 = validate(model, val_loader, epoch, args)
        logging.info(f"Epoch {epoch} - Val EPE: {val_epe:.6f}, D1: {val_d1:.6f}")
        
        # Check if this is the best model
        is_best = val_epe < best_epe
        if is_best:
            best_epe = val_epe
            logging.info(f"New best model with EPE: {best_epe:.6f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0 or is_best:
            save_checkpoint(model, optimizer, epoch, val_epe, val_d1, is_best, args)
    
    # Clean up
    wandb.finish()
    logging.info("Training completed!")


if __name__ == "__main__":
    main() 