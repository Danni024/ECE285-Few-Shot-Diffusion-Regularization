# ECE285-Few-Shot-Diffusion-Regularization
# Few-Shot Diffusion Generation via LoRA with Latent Regularization

This repository contains the official implementation for the course project. We focus on mitigating mode collapse in extreme few-shot scenarios.

## Core Features
- **LoRA Adaptation**: Parameter-efficient fine-tuning for Stable Diffusion v1.5.
- **Latent MixUp**: Manifold augmentation in the VAE latent space.
- **Diversity Loss**: Orthogonal regularization of noise predictions.

## Usage
1. Train the model: `python train.py`
2. Generate evaluation images: `python generate.py`
3. Calculate metrics: `python eval_metrics.py`
