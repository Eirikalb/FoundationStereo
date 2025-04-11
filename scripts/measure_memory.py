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
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import logging
import time
from omegaconf import OmegaConf
import json
# Add parent directory to path for imports
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

# Import project modules
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder

def measure_inference_memory(model, crop_sizes, batch_sizes, args, num_iters=12):
    """Measure memory usage during inference with different crop sizes and batch sizes"""
    memory_allocated = np.zeros((len(crop_sizes), len(batch_sizes)))
    memory_reserved = np.zeros((len(crop_sizes), len(batch_sizes)))
    compute_times = np.zeros((len(crop_sizes), len(batch_sizes)))
    
    # Use None to indicate OOM or errors
    memory_allocated.fill(None)
    memory_reserved.fill(None)
    compute_times.fill(None)
    
    for i, crop_size in enumerate(tqdm(crop_sizes, desc="Measuring inference memory")):
        for j, batch_size in enumerate(batch_sizes):
            try:
                # Clear cache before each measurement
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create random inputs with the current crop size and batch size
                height, width = crop_size, crop_size
                left = torch.randn(batch_size, 3, height, width).cuda()
                right = torch.randn(batch_size, 3, height, width).cuda()
                
                # Run model in eval mode
                model.eval()
                
                # Warm-up run
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    _ = model(left, right, iters=num_iters)
                
                # Clear cache again
                torch.cuda.empty_cache()
                gc.collect()
                
                # Run model and measure memory and time
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    _ = model(left, right, iters=num_iters)
                
                torch.cuda.synchronize()
                end_time = time.time()
                compute_time = (end_time - start_time) * 1000  # Convert to ms
                    
                # Record memory usage
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                
                memory_allocated[i, j] = allocated / (1024 ** 2)  # Convert to MB
                memory_reserved[i, j] = reserved / (1024 ** 2)    # Convert to MB
                compute_times[i, j] = compute_time
                
                print(f"Crop Size: {crop_size}x{crop_size}, Batch Size: {batch_size}, Allocated: {allocated / (1024 ** 2):.2f} MB, Reserved: {reserved / (1024 ** 2):.2f} MB, Time: {compute_time:.2f} ms")
            
            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory for crop size {crop_size}x{crop_size}, batch size {batch_size} during inference. Skipping.")
            except Exception as e:
                print(f"Error for crop size {crop_size}x{crop_size}, batch size {batch_size} during inference: {str(e)}. Skipping.")
            finally:
                # Make sure to clear cache on error to recover memory
                torch.cuda.empty_cache()
                gc.collect()
    
    return memory_allocated, memory_reserved, compute_times

def measure_training_memory(model, crop_sizes, batch_sizes, args, num_iters=12):
    """Measure memory usage during training with different crop sizes and batch sizes"""
    memory_allocated = np.zeros((len(crop_sizes), len(batch_sizes)))
    memory_reserved = np.zeros((len(crop_sizes), len(batch_sizes)))
    compute_times = np.zeros((len(crop_sizes), len(batch_sizes)))
    
    # Use None to indicate OOM or errors
    memory_allocated.fill(None)
    memory_reserved.fill(None)
    compute_times.fill(None)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    
    # Set up scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    
    for i, crop_size in enumerate(tqdm(crop_sizes, desc="Measuring training memory")):
        for j, batch_size in enumerate(batch_sizes):
            try:
                # Clear cache before each measurement
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create random inputs with the current crop size and batch size
                height, width = crop_size, crop_size
                left = torch.randn(batch_size, 3, height, width).cuda()
                right = torch.randn(batch_size, 3, height, width).cuda()
                
                # Create fake ground truth
                disp_gt = torch.rand(batch_size, 1, height, width).cuda() * 100.0
                valid_mask = torch.ones_like(disp_gt, dtype=torch.bool).cuda()
                
                # Run model in train mode
                model.train()
                optimizer.zero_grad()
                
                # Warm-up run with mixed precision
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    init_disp, disp_preds = model(left, right, iters=num_iters)
                    loss = torch.nn.functional.smooth_l1_loss(disp_preds[-1][valid_mask], disp_gt[valid_mask])
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear cache again
                torch.cuda.empty_cache()
                gc.collect()
                
                # Run model and measure memory and time
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    init_disp, disp_preds = model(left, right, iters=num_iters)
                    loss = torch.nn.functional.smooth_l1_loss(disp_preds[-1][valid_mask], disp_gt[valid_mask])
                
                scaler.scale(loss).backward()
                
                torch.cuda.synchronize()
                end_time = time.time()
                compute_time = (end_time - start_time) * 1000  # Convert to ms
                
                # Record memory usage
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                
                memory_allocated[i, j] = allocated / (1024 ** 2)  # Convert to MB
                memory_reserved[i, j] = reserved / (1024 ** 2)    # Convert to MB
                compute_times[i, j] = compute_time
                
                print(f"Crop Size: {crop_size}x{crop_size}, Batch Size: {batch_size}, Allocated: {allocated / (1024 ** 2):.2f} MB, Reserved: {reserved / (1024 ** 2):.2f} MB, Time: {compute_time:.2f} ms")
                
                # Clear gradients
                optimizer.zero_grad()
            
            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory for crop size {crop_size}x{crop_size}, batch size {batch_size} during training. Skipping.")
            except Exception as e:
                print(f"Error for crop size {crop_size}x{crop_size}, batch size {batch_size} during training: {str(e)}. Skipping.")
            finally:
                # Make sure to clear cache and reset gradients on error to recover memory
                if 'optimizer' in locals():
                    optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
    
    return memory_allocated, memory_reserved, compute_times

def plot_memory_usage(crop_sizes, batch_sizes, inference_allocated, inference_reserved, training_allocated, training_reserved, 
                      inference_times, training_times, args, output_file=None):
    """Plot memory usage and compute times for different crop sizes and batch sizes"""
    # Create a figure with 6 subplots (2 rows, 3 columns)
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training Memory Allocated (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[0, 0].plot(crop_sizes, training_allocated[:, j], label=f'Batch {batch_size}',marker='o')
    ax[0, 0].set_xlabel('Crop Size')
    ax[0, 0].set_ylabel('Memory Allocated (MB)')
    ax[0, 0].set_title('Training Memory Allocated')
    ax[0, 0].set_ylim(0, 13000)
    ax[0, 0].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[0, 0].legend()
    ax[0, 0].grid(True)
    
    # Plot 2: Training Memory Reserved (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[0, 1].plot(crop_sizes, training_reserved[:, j], label=f'Batch {batch_size}',marker='o')
    ax[0, 1].set_xlabel('Crop Size')
    ax[0, 1].set_ylabel('Memory Reserved (MB)')
    ax[0, 1].set_title('Training Memory Reserved')
    ax[0, 1].set_ylim(0, 13000)
    ax[0, 1].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[0, 1].legend()
    ax[0, 1].grid(True)
    
    # Plot 3: Training Compute Time (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[0, 2].plot(crop_sizes, training_times[:, j], label=f'Batch {batch_size}',marker='o')
    ax[0, 2].set_xlabel('Crop Size')
    ax[0, 2].set_ylabel('Compute Time (ms)')
    ax[0, 2].set_title('Training Compute Time')
    ax[0, 2].set_ylim(0, 5000)  # 5 seconds in milliseconds
    ax[0, 2].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[0, 2].legend()
    ax[0, 2].grid(True)
    
    # Plot 4: Inference Memory Allocated (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[1, 0].plot(crop_sizes, inference_allocated[:, j], label=f'Batch {batch_size}',marker='o')
    ax[1, 0].set_xlabel('Crop Size')
    ax[1, 0].set_ylabel('Memory Allocated (MB)')
    ax[1, 0].set_title('Inference Memory Allocated')
    ax[1, 0].set_ylim(0, 13000)
    ax[1, 0].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[1, 0].legend()
    ax[1, 0].grid(True)
    
    # Plot 5: Inference Memory Reserved (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[1, 1].plot(crop_sizes, inference_reserved[:, j], label=f'Batch {batch_size}',marker='o')
    ax[1, 1].set_xlabel('Crop Size')
    ax[1, 1].set_ylabel('Memory Reserved (MB)')
    ax[1, 1].set_title('Inference Memory Reserved')
    ax[1, 1].set_ylim(0, 13000)
    ax[1, 1].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[1, 1].legend()
    ax[1, 1].grid(True)
    
    # Plot 6: Inference Compute Time (line plot)
    for j, batch_size in enumerate(batch_sizes):
        ax[1, 2].plot(crop_sizes, inference_times[:, j], label=f'Batch {batch_size}',marker='o')
    ax[1, 2].set_xlabel('Crop Size')
    ax[1, 2].set_ylabel('Compute Time (ms)')
    ax[1, 2].set_title('Inference Compute Time')
    ax[1, 2].set_ylim(0, 5000)  # 5 seconds in milliseconds
    ax[1, 2].set_xlim(min(crop_sizes)*0.9, max(crop_sizes)*1.1)
    ax[1, 2].legend()
    ax[1, 2].grid(True)
    
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Measure memory usage of FoundationStereo model')
    
    # Model parameters
    parser.add_argument('--pretrained_model', type=str, default='pretrained_models/23-51-11/model_best_bp2.pth', help='path to pretrained model')
    parser.add_argument('--max_disp', type=int, default=416, help='maximum disparity')
    
    # Measurement parameters
    parser.add_argument('--min_crop_size', type=int, default=128, help='minimum crop size to test')
    parser.add_argument('--max_crop_size', type=int, default=512*2, help='maximum crop size to test')
    parser.add_argument('--step_size', type=int, default=64, help='step size between crop sizes')
    parser.add_argument('--min_batch_size', type=int, default=1, help='minimum batch size to test')
    parser.add_argument('--max_batch_size', type=int, default=8, help='maximum batch size to test')
    parser.add_argument('--batch_step_size', type=int, default=1, help='step size between batch sizes')
    parser.add_argument('--num_iters', type=int, default=12, help='number of iterations')
    parser.add_argument('--output_file', type=str, default='memory_usage.png', help='output file for plot')
    
    # Architecture parameters
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128, 128, 128], help='hidden dimensions')
    parser.add_argument('--n_downsample', type=int, default=2, help='number of downsample layers')
    parser.add_argument('--n_gru_layers', type=int, default=3, help='number of GRU layers')
    parser.add_argument('--corr_levels', type=int, default=2, help='number of correlation levels')
    parser.add_argument('--corr_radius', type=int, default=4, help='correlation radius')
    
    # Mixed precision parameters
    parser.add_argument('--mixed_precision', action='store_true', default=True, 
                        help='use mixed precision training (required for FlashAttention)')
    parser.add_argument('--dtype', type=str, default='bf16', choices=['fp16', 'bf16'],
                        help='data type for mixed precision (fp16 or bf16)')
    parser.add_argument('--freeze_depth_anything', action='store_true', default=True,
                        help='Freeze depth anything part of model (for memory testing)')
    parser.add_argument('--output_dir', type=str, default='memory_results', help='output directory for results')
    parser.add_argument('--previous_results', type=str, default=None, help='path to previous results file')
    
    
    args = parser.parse_args()
    if args.previous_results is not None:
        with open(args.previous_results, 'r') as f:
            results = json.load(f)
            crop_sizes = results['crop_sizes']
            batch_sizes = results['batch_sizes']
            
    else:        
        # Add dtype mapping to args
        args.dtype_map = {
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        
        # Generate crop sizes and batch sizes
        crop_sizes = list(range(args.min_crop_size, args.max_crop_size + 1, args.step_size))
        batch_sizes = list(range(args.min_batch_size, args.max_batch_size + 1, args.batch_step_size))
        
        # Create model
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
        
        # Handle freeze_depth_anything - if not specified in config but specified in args

        model = FoundationStereo(args)
        
        # Load pretrained model if specified
        if args.pretrained_model is not None:
            print(f"Loading pretrained model from {args.pretrained_model}")
            ckpt = torch.load(args.pretrained_model)
            model.load_state_dict(ckpt['model'], strict=False)
        
        # Freeze depth_anything part if requested
        if args.freeze_depth_anything:
            print("Freezing depth_anything part of the model for testing")
            for name, param in model.named_parameters():
                if 'depth_anything' in name:
                    param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        model.cuda()
        
        print(f"Model created and moved to GPU")
        print(f"Mixed precision: {args.mixed_precision}, dtype: {args.dtype}")
        print(f"Testing crop sizes: {crop_sizes}")
        print(f"Testing batch sizes: {batch_sizes}")
        
        # Measure memory usage and compute time during inference
        inference_allocated, inference_reserved, inference_times = measure_inference_memory(
            model, crop_sizes, batch_sizes, args, num_iters=args.num_iters
        )
        
        # Measure memory usage and compute time during training
        training_allocated, training_reserved, training_times = measure_training_memory(
            model, crop_sizes, batch_sizes, args, num_iters=args.num_iters
        )
        # Save results to file
        results = {
            'args': {
                'mixed_precision': args.mixed_precision,
                'dtype': args.dtype,
                'num_iters': args.num_iters
            },
            'model_stats': {
                'total_params': total_params,
                'trainable_params': trainable_params
            },
            'crop_sizes': crop_sizes,
            'batch_sizes': batch_sizes,
            'inference': {
                'allocated': inference_allocated.tolist(),
                'reserved': inference_reserved.tolist(),
                'times': inference_times.tolist()
            },
            'training': {
                'allocated': training_allocated.tolist(),
                'reserved': training_reserved.tolist(), 
                'times': training_times.tolist()
            }
        }

        # Save results as JSON
        results_path = os.path.join(os.path.dirname(args.output_file), 'memory_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")
    
    inference_allocated = np.array(results['inference']['allocated'])
    inference_reserved = np.array(results['inference']['reserved'])
    training_allocated = np.array(results['training']['allocated'])
    training_reserved = np.array(results['training']['reserved'])
    inference_times = np.array(results['inference']['times'])
    training_times = np.array(results['training']['times'])
    
    # Plot results
    print("Plotting results...")
    plot_memory_usage(
        crop_sizes, batch_sizes,
        inference_allocated, inference_reserved,
        training_allocated, training_reserved,
        inference_times, training_times,
        args,
        output_file=args.output_file
    )
    
    # Print summary of results
    print("\nSummary of Results:")
    print(f"Mixed Precision Enabled: {args.mixed_precision}")
    print(f"Precision Type: {args.dtype}")
    
    # Find the largest successful crop size and batch size for inference
    max_inf_crop = None
    max_inf_batch = None
    max_inf_memory = 0
    max_inf_time = 0
    
    for i in range(len(crop_sizes)):
        for j in range(len(batch_sizes)):
            if inference_allocated[i, j] is not None:
                if max_inf_crop is None or (crop_sizes[i] > max_inf_crop and batch_sizes[j] >= max_inf_batch):
                    max_inf_crop = crop_sizes[i]
                    max_inf_batch = batch_sizes[j]
                    max_inf_memory = inference_allocated[i, j]
                    max_inf_time = inference_times[i, j]
    
    # Find the largest successful crop size and batch size for training
    max_train_crop = None
    max_train_batch = None
    max_train_memory = 0
    max_train_time = 0
    
    for i in range(len(crop_sizes)):
        for j in range(len(batch_sizes)):
            if training_allocated[i, j] is not None:
                if max_train_crop is None or (crop_sizes[i] > max_train_crop and batch_sizes[j] >= max_train_batch):
                    max_train_crop = crop_sizes[i]
                    max_train_batch = batch_sizes[j]
                    max_train_memory = training_allocated[i, j]
                    max_train_time = training_times[i, j]
    
    if max_inf_crop is not None:
        print(f"Largest Successful Inference: {max_inf_crop}x{max_inf_crop} with batch size {max_inf_batch}")
        print(f"Maximum Memory Usage - Inference: {max_inf_memory:.2f} MB")
        print(f"Maximum Compute Time - Inference: {max_inf_time:.2f} ms")
    else:
        print("No successful inference measurements")
    
    if max_train_crop is not None:
        print(f"Largest Successful Training: {max_train_crop}x{max_train_crop} with batch size {max_train_batch}")
        print(f"Maximum Memory Usage - Training: {max_train_memory:.2f} MB")
        print(f"Maximum Compute Time - Training: {max_train_time:.2f} ms")
    else:
        print("No successful training measurements")
    
    # Calculate memory and time efficiency
    if max_inf_crop is not None and max_train_crop is not None:
        print(f"\nEfficiency Analysis:")
        print(f"Memory Efficiency (Training/Inference): {max_train_memory / max_inf_memory:.2f}x")
        print(f"Time Efficiency (Training/Inference): {max_train_time / max_inf_time:.2f}x")
        
        # Calculate memory and time per pixel
        inf_pixels = max_inf_crop * max_inf_crop * max_inf_batch
        train_pixels = max_train_crop * max_train_crop * max_train_batch
        
        print(f"Memory per pixel (Inference): {max_inf_memory / inf_pixels:.6f} MB/pixel")
        print(f"Memory per pixel (Training): {max_train_memory / train_pixels:.6f} MB/pixel")
        print(f"Time per pixel (Inference): {max_inf_time / inf_pixels:.6f} ms/pixel")
        print(f"Time per pixel (Training): {max_train_time / train_pixels:.6f} ms/pixel")

if __name__ == "__main__":
    main() 