"""
Generate a qualitative comparison grid across the four ablation groups.

Loads each group's LoRA checkpoint (trained via train.py), generates one image per
random seed, and tiles them into a single grid image (one row per group, one column
per seed) for visual comparison.

Usage:
    python generate_samples.py
"""

import os

import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"

# (label, path to LoRA checkpoint dir, or None for the un-fine-tuned base model)
MODEL_CONFIGS = [
    ("1_SD_Base", None),
    ("2_Baseline_LoRA", "exp_baseline_lora/checkpoints/epoch_9"),
    ("3_Baseline_MixUp", "exp_baseline_mixup/checkpoints/epoch_9"),
    ("4_Proposed_Full", "exp_proposed/checkpoints/epoch_9"),
]
SEEDS = [42, 123, 555, 777, 8888]
PROMPT = "a photo of a cat"


def generate_comparison_grid(model_configs, prompt, seeds, device="cuda", out_path="comparison_grid.png"):
    all_images = []

    for label, lora_path in model_configs:
        print(f"generating group: {label}...")
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, safety_checker=None
        ).to(device)

        if lora_path and os.path.exists(lora_path):
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)

        exp_images = []
        for seed in seeds:
            generator = torch.Generator(device).manual_seed(seed)
            image = pipe(prompt, num_inference_steps=50, generator=generator, guidance_scale=7.5).images[0]
            exp_images.append(image)
        all_images.append(exp_images)

        del pipe
        torch.cuda.empty_cache()

    rows, cols = len(model_configs), len(seeds)
    w, h = all_images[0][0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, row_imgs in enumerate(all_images):
        for j, img in enumerate(row_imgs):
            grid.paste(img, box=(j * w, i * h))

    grid.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    generate_comparison_grid(MODEL_CONFIGS, PROMPT, SEEDS)
