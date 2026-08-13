"""
Quantitative evaluation: FID, Precision, Recall for each ablation group.

Generates `num_images` samples per group, then computes:
  - FID (Frechet Inception Distance) via the `pytorch-fid` CLI
  - Precision / Recall via the `prdc` library (using Inception-v3 features)

Usage:
    python evaluate.py --num_images 100
"""

import argparse
import os
import subprocess

import numpy as np
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from peft import PeftModel
from PIL import Image
from prdc import compute_prdc
from torchvision.models import inception_v3
from tqdm import tqdm

MODEL_ID = "runwayml/stable-diffusion-v1-5"
REAL_DIR = "cat_selected"

MODEL_CONFIGS = [
    ("1_SD_Base", None),
    ("2_Baseline_LoRA", "exp_baseline_lora/checkpoints/epoch_9"),
    ("3_Baseline_MixUp", "exp_baseline_mixup/checkpoints/epoch_9"),
    ("4_Proposed_Full", "exp_proposed/checkpoints/epoch_9"),
]


def batch_generate_for_metrics(model_configs, prompt="a photo of a cat", num_images=100):
    for label, lora_path in model_configs:
        print(f"generating group: {label}...")
        save_dir = f"eval_images/{label}"
        os.makedirs(save_dir, exist_ok=True)

        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, variant="fp16", safety_checker=None
        ).to("cuda")
        pipe.enable_attention_slicing()

        if lora_path and os.path.exists(lora_path):
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet.to(device="cuda", dtype=torch.float16)

        for i in tqdm(range(num_images)):
            seed = 2000 + i
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt, num_inference_steps=25, generator=generator, guidance_scale=7.5).images[0]
            image.save(f"{save_dir}/seed_{seed}.png")

        del pipe
        torch.cuda.empty_cache()


def get_features(folder_path, model, device, num_samples=100):
    model.eval()
    features = []
    transform = T.Compose(
        [
            T.Resize((299, 299)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    files = [f for f in os.listdir(folder_path) if f.endswith((".png", ".jpg"))][:num_samples]
    for f in files:
        img = Image.open(os.path.join(folder_path, f)).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_t)
        features.append(feat.cpu().numpy())
    return np.concatenate(features, axis=0)


def compute_fid(real_dir: str, gen_dir: str) -> float:
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", real_dir, gen_dir, "--device", "cuda:0"],
        capture_output=True,
        text=True,
        check=True,
    )
    last_line = result.stdout.strip().splitlines()[-1]
    return float(last_line.split(":")[-1])


def evaluate(model_configs, real_dir=REAL_DIR):
    device = "cuda"
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()

    real_feats = get_features(real_dir, inception_model, device)

    print(f"{'Experiment Group':<20} | {'FID':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 60)
    for label, _ in model_configs:
        gen_dir = f"eval_images/{label}"
        fid_val = compute_fid(real_dir, gen_dir)

        gen_feats = get_features(gen_dir, inception_model, device)
        metrics = compute_prdc(real_features=real_feats, fake_features=gen_feats, nearest_k=5)

        print(f"{label:<20} | {fid_val:<10.2f} | {metrics['precision']:<10.4f} | {metrics['recall']:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation, only compute metrics")
    args = parser.parse_args()

    if not args.skip_generation:
        batch_generate_for_metrics(MODEL_CONFIGS, num_images=args.num_images)

    evaluate(MODEL_CONFIGS)
