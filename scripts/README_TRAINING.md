# Training FoundationStereo

This guide will help you prepare your data and train the FoundationStereo model on your own stereo datasets.

## Model Architecture Overview

FoundationStereo is a state-of-the-art stereo matching model that leverages:

1. Feature extraction from both EdgeNeXt and DepthAnything models
2. Cost volume construction and processing with attention mechanisms
3. Iterative disparity refinement using ConvGRU updates
4. Multi-scale processing with hierarchical features
5. Upsampling mechanisms for full-resolution output

## Data Preparation

To train FoundationStereo, organize your dataset in the following structure:

```
data_dir/
├── train/
│   ├── left/          # Left stereo images (.png, .jpg)
│   ├── right/         # Right stereo images (.png, .jpg)
│   └── disparity/     # Ground truth disparity maps (.png)
└── test/              # (Optional) Same structure as train
    ├── left/
    ├── right/
    └── disparity/
```

### Data Requirements:
- Left and right images should be rectified and have the same names across folders
- Disparity maps should be single-channel images with pixel values proportional to disparity
- Common datasets like Scene Flow, KITTI, and ETH3D can be arranged in this format

### Data Format Notes:
- The training script handles disparity maps where values are either:
  - Directly in pixel units (values typically < 256)
  - Stored in 16-bit format with scaling (values > 256 are divided by 256)
- Adjust the scaling in `StereoDataset.__getitem__` if your data format differs

## Training the Model

### Prerequisites:
- PyTorch (1.8+)
- CUDA-enabled GPU (8GB+ VRAM recommended)
- Required Python packages (listed in `environment.yml`)

### Basic Training:

```bash
python scripts/train_foundation.py --data_dir /path/to/dataset --checkpoint_dir /path/to/save/checkpoints
```

### Fine-tuning from a Pretrained Model:

```bash
python scripts/train_foundation.py \
    --data_dir /path/to/dataset \
    --checkpoint_dir /path/to/save/checkpoints \
    --pretrained_model /path/to/model_best.pth \
    --epochs 10 \
    --lr 0.00005
```

### Important Parameters:

- **Data parameters:**
  - `--data_dir`: Path to your dataset directory
  - `--val_percent`: Percentage of training data to use for validation (default: 0.1)
  - `--max_disp`: Maximum disparity value (default: 192)

- **Training parameters:**
  - `--batch_size`: Batch size (default: 2, adjust based on GPU memory)
  - `--epochs`: Number of training epochs (default: 20)
  - `--lr`: Learning rate (default: 0.0001)
  - `--train_iters`: Number of GRU iterations during training (default: 12)
  - `--valid_iters`: Number of GRU iterations during validation (default: 32)

- **Architecture parameters:**
  - `--hidden_dims`: Hidden dimensions (default: [128, 128, 128])
  - `--n_downsample`: Number of downsample layers (default: 3)
  - `--n_gru_layers`: Number of GRU layers (default: 3)
  - `--corr_levels`: Number of correlation levels (default: 2)
  - `--corr_radius`: Correlation radius (default: 4)

## Monitoring Training

The training script tracks several metrics:

- **Loss**: Combination of smooth L1 losses at different scales
- **EPE**: End-Point-Error in validation (average absolute disparity error)
- **D1**: Percentage of points with error > 3px or >5% of ground truth

All metrics are logged to Weights & Biases (wandb). To use wandb:

1. Install wandb:
```bash
pip install wandb
```

2. Login to your wandb account:
```bash
wandb login
```

3. Run training with wandb configuration:
```bash
python scripts/train_foundation.py \
    --data_dir /path/to/dataset \
    --checkpoint_dir /path/to/save/checkpoints \
    --pretrained_model /path/to/model_best.pth \
    --wandb_project foundation-stereo \
    --wandb_entity your-username \
    --epochs 10 \
    --lr 0.00005
```

The metrics will be automatically logged to your wandb dashboard, where you can:
- Track training progress in real-time
- Compare different runs
- Visualize metrics and model architecture
- Share results with team members

## Using a Trained Model

After training, you can use your model for inference:

```bash
python scripts/run_demo.py \
    --left_file path/to/left.png \
    --right_file path/to/right.png \
    --ckpt_dir /path/to/checkpoints/model_best.pth
```

## Training Tips

1. **GPU Memory**: If you encounter out-of-memory issues, reduce the batch size or use mixed precision training (`--mixed_precision`)
2. **Learning Rate**: Start with lr=0.0001 for training from scratch, and lr=0.00005 or lower for fine-tuning
3. **Iteration Count**: More iterations (--train_iters) generally improves results but increases training time
4. **Dataset Size**: If your dataset is small, consider using data augmentation (random crop, color jitter, etc.)
5. **Pre-training**: For best results, start with the provided pretrained model and fine-tune on your data

## Performance Expectations

Training time and performance depend on several factors:

- A full training on Scene Flow (~35K images) takes ~2-3 days on a single NVIDIA RTX 3090
- Fine-tuning on a smaller dataset like KITTI (~200 images) can take a few hours
- The model typically achieves competitive EPE (<1px) on standard benchmarks when properly trained

## Troubleshooting

- **High Loss Values**: Check your disparity map format and ensure proper scaling
- **NaN in Loss**: Reduce learning rate or check for invalid (inf, NaN) disparity values
- **Poor Convergence**: Adjust learning rate, batch size, or use a pretrained model
- **OOM Errors**: Reduce batch size, image resolution, or max_disp value 