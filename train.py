"""
Train Stable Diffusion v1.5 with LoRA on a 15-image few-shot support set.

Supports four ablation configurations by toggling Latent MixUp and the Diversity Loss:
  1. baseline_sd    - vanilla Stable Diffusion, no fine-tuning
  2. baseline_lora   - LoRA fine-tuning only
  3. baseline_mixup  - LoRA + Latent MixUp
  4. proposed        - LoRA + Latent MixUp + Diversity Loss (full method)

Usage:
    python train.py --exp_type proposed --epochs 10 --save_every 2
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

MODEL_ID = "runwayml/stable-diffusion-v1-5"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_diversity_loss(noise_pred: torch.Tensor) -> torch.Tensor:
    """Section 3.3: penalize cosine similarity between batch-wise noise predictions."""
    batch_size = noise_pred.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=noise_pred.device)
    flat_noise = noise_pred.view(batch_size, -1)
    norm_noise = F.normalize(flat_noise, p=2, dim=1)
    sim_matrix = torch.mm(norm_noise, norm_noise.t())
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=noise_pred.device)
    return sim_matrix[mask].mean()


def apply_latent_mixup(latents: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Section 3.4: linear interpolation between latents of two support samples."""
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(latents.shape[0]).to(latents.device)
    return lam * latents + (1 - lam) * latents[index, :]


class SelectedCatDataset(Dataset):
    """Loads the 15-image few-shot support set, repeated to fill out an epoch."""

    def __init__(self, img_dir: str, tokenizer, size: int = 512, epoch_len: int = 60):
        self.img_dir = img_dir
        self.img_names = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith(".")
        ]
        self.tokenizer = tokenizer
        self.epoch_len = epoch_len
        self.transform = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.img_dir, self.img_names[idx % len(self.img_names)])
        img = Image.open(img_path).convert("RGB")
        input_ids = self.tokenizer(
            "a photo of a cat",
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        ).input_ids[0]
        return {"pixel_values": self.transform(img), "input_ids": input_ids}


def run_experiment(
    exp_type: str = "proposed",
    epochs: int = 30,
    save_every: int = 5,
    data_dir: str = "cat_selected",
) -> None:
    set_seed(42)  # reset before each experiment so ablations are directly comparable
    print(f"\nstart experiment: {exp_type.upper()}")

    output_dir = f"exp_{exp_type}"
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_ID, subfolder="text_encoder").cuda()
    vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").cuda()
    unet = UNet2DConditionModel.from_pretrained(MODEL_ID, subfolder="unet").cuda()
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    if exp_type == "baseline_sd":
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
        for i in range(5):
            img = pipe("a photo of a cat", num_inference_steps=30).images[0]
            img.save(f"{output_dir}/samples/sd_orig_{i}.png")
        return

    lora_config = LoraConfig(r=16, lora_alpha=16, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet = get_peft_model(unet, lora_config)
    unet.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-5)
    dataset = SelectedCatDataset(img_dir=data_dir, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        unet.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                latents = vae.encode(batch["pixel_values"].cuda()).latent_dist.sample() * 0.18215

                if exp_type in ("baseline_mixup", "proposed"):
                    latents = apply_latent_mixup(latents)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cuda").long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"].cuda())[0]
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                l_simple = F.mse_loss(noise_pred.float(), noise.float())

                if exp_type == "proposed":
                    l_div = calculate_diversity_loss(noise_pred)
                    loss = l_simple + 0.03 * l_div
                else:
                    l_div = torch.tensor(0.0, device="cuda")
                    loss = l_simple

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_postfix({"loss": loss.item(), "div": l_div.item()})

        if epoch % save_every == 0 or epoch == epochs - 1:
            unet.save_pretrained(f"{output_dir}/checkpoints/epoch_{epoch}")

            unet.eval()
            with torch.no_grad():
                unet.half()
                generator = torch.Generator("cuda").manual_seed(42)
                unet_for_sampling = unet.module if hasattr(unet, "module") else unet
                pipe = StableDiffusionPipeline.from_pretrained(
                    MODEL_ID, unet=unet_for_sampling, torch_dtype=torch.float16, safety_checker=None
                ).to("cuda")
                img = pipe("a photo of a cat", num_inference_steps=30, generator=generator).images[0]
                img.save(f"{output_dir}/samples/epoch_{epoch}.png")
                unet.float()
                del pipe
                torch.cuda.empty_cache()

    print(f"{exp_type} experiment finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_type",
        choices=["baseline_sd", "baseline_lora", "baseline_mixup", "proposed"],
        default="proposed",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="cat_selected")
    args = parser.parse_args()

    run_experiment(args.exp_type, args.epochs, args.save_every, args.data_dir)
