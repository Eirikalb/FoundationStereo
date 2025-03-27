#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance


class RandomCrop:
    """Random crop the image and disparity map."""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, left_img, right_img, disp, has_disp):
        assert left_img.size == right_img.size
        width, height = left_img.size
        crop_height, crop_width = self.size
        
        # Ensure the crop area is not larger than the image
        if width < crop_width or height < crop_height:
            # Resize to at least minimum dimensions
            scale = max(crop_width / width, crop_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            left_img = left_img.resize((new_width, new_height), Image.BILINEAR)
            right_img = right_img.resize((new_width, new_height), Image.BILINEAR)
            if has_disp:
                # Scale disparity map accordingly
                disp = np.array(disp, dtype=np.float32) * scale
            
            width, height = new_width, new_height
        
        # Get random crop coordinates
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        # Crop images
        left_img = left_img.crop((left, top, left + crop_width, top + crop_height))
        right_img = right_img.crop((left, top, left + crop_width, top + crop_height))
        
        # Crop disparity map if available
        if has_disp:
            if isinstance(disp, np.ndarray):
                disp = disp[top:top + crop_height, left:left + crop_width]
            else:
                disp = disp.crop((left, top, left + crop_width, top + crop_height))
        
        return left_img, right_img, disp, has_disp


class ColorJitter:
    """Apply color jitter to images."""
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, left_img, right_img, disp, has_disp):
        # Apply the same color transformations to both images
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        # Apply transforms in sequence
        left_img = ImageEnhance.Brightness(left_img).enhance(brightness_factor)
        right_img = ImageEnhance.Brightness(right_img).enhance(brightness_factor)
        
        left_img = ImageEnhance.Contrast(left_img).enhance(contrast_factor)
        right_img = ImageEnhance.Contrast(right_img).enhance(contrast_factor)
        
        left_img = ImageEnhance.Color(left_img).enhance(saturation_factor)
        right_img = ImageEnhance.Color(right_img).enhance(saturation_factor)
        
        # Convert to HSV for hue adjustment and back to RGB
        left_img = TF.adjust_hue(left_img, hue_factor)
        right_img = TF.adjust_hue(right_img, hue_factor)
        
        return left_img, right_img, disp, has_disp


class RandomGamma:
    """Apply random gamma correction to images."""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
    
    def __call__(self, left_img, right_img, disp, has_disp):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        
        left_img = TF.adjust_gamma(left_img, gamma)
        right_img = TF.adjust_gamma(right_img, gamma)
        
        return left_img, right_img, disp, has_disp


class ToTensor:
    """Convert images and disparity to tensors."""
    def __call__(self, left_img, right_img, disp, has_disp):
        # Convert images to tensors
        left_tensor = TF.to_tensor(left_img)
        right_tensor = TF.to_tensor(right_img)
        
        # Convert disparity to tensor if available
        if has_disp:
            if isinstance(disp, np.ndarray):
                disp_tensor = torch.from_numpy(disp).unsqueeze(0).float()
            else:
                disp_tensor = torch.from_numpy(np.array(disp, dtype=np.float32)).unsqueeze(0)
        else:
            # Create dummy disparity tensor
            disp_tensor = torch.zeros((1, left_tensor.shape[1], left_tensor.shape[2]))
        
        return left_tensor, right_tensor, disp_tensor, has_disp


class Normalize:
    """Normalize tensor images with mean and standard deviation."""
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, left_tensor, right_tensor, disp_tensor, has_disp):
        # Normalize images
        left_tensor = TF.normalize(left_tensor, self.mean, self.std)
        right_tensor = TF.normalize(right_tensor, self.mean, self.std)
        
        return left_tensor, right_tensor, disp_tensor, has_disp


class RandomHorizontalFlip:
    """Randomly flip images horizontally."""
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, left_img, right_img, disp, has_disp):
        if random.random() < self.prob:
            # Swap left and right images
            left_img, right_img = right_img, left_img
            
            # Update disparity map if available
            if has_disp:
                if isinstance(disp, np.ndarray):
                    disp = -disp
                else:
                    disp = np.array(disp, dtype=np.float32) * -1
        
        return left_img, right_img, disp, has_disp


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, left_img, right_img, disp, has_disp):
        for t in self.transforms:
            left_img, right_img, disp, has_disp = t(left_img, right_img, disp, has_disp)
        return left_img, right_img, disp, has_disp


def get_training_transforms(crop_size=(384, 512), do_augmentation=True):
    """Return standard transforms for stereo training."""
    transforms_list = []
    
    if do_augmentation:
        transforms_list.extend([
            RandomCrop(crop_size),
            ColorJitter(0.3, 0.3, 0.3, 0.3),
            RandomGamma((0.8, 1.2)),
        ])
    
    transforms_list.append(ToTensor())
    
    return Compose(transforms_list)


def get_validation_transforms():
    """Return standard transforms for stereo validation."""
    return Compose([
        ToTensor(),
    ]) 