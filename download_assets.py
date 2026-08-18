"""
download_assets.py - runs DURING docker build, not at container runtime.

Downloads the model and both datasets into local paths baked into the image,
so the Salad container never needs to hit the network for these at runtime
(only for pushing checkpoints back out, which is much smaller traffic).

HF_TOKEN is read from environment - passed in via Docker BuildKit secret,
never persisted into an image layer.
"""

import os
from huggingface_hub import login, snapshot_download
from datasets import load_dataset

login(token=os.environ["HF_TOKEN"])

# CORRECTED: Unsloth publishes this as the plain repo, quantized on-the-fly
# via load_in_4bit=True at load time - there is NOT a separate pre-quantized
# "-bnb-4bit" repo for Gemma 4 the way there was for Gemma 3. This means the
# download is the full ~24GB model, not a small pre-quantized one - size your
# GitHub Actions runner disk and Salad node accordingly.
MODEL_REPO = "unsloth/gemma-4-12b-it"
CURATED_REPO_ID = "YOUR_HF_USERNAME/luau-curated-308"  # same as before

print(f"Downloading model: {MODEL_REPO} ...")
snapshot_download(repo_id=MODEL_REPO, local_dir="/app/model")

print("Downloading TorpedoSoftware/Roblox-Luau-Reasoning-v1.0 ...")
corpus_ds = load_dataset("TorpedoSoftware/Roblox-Luau-Reasoning-v1.0", split="train")
corpus_ds.save_to_disk("/app/data/corpus")

print(f"Downloading curated dataset: {CURATED_REPO_ID} ...")
curated_ds = load_dataset(CURATED_REPO_ID, split="train")
curated_ds.save_to_disk("/app/data/curated")

print("All assets downloaded and cached in image.")
