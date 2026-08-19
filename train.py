"""
train.py - runs on the Salad GPU node.
Corrected to load baked-in assets from local disk instead of re-downloading them.
"""

import os
import re
from datasets import load_from_disk, concatenate_datasets # Updated to load_from_disk
from huggingface_hub import login
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

login(token=os.environ["HF_TOKEN"])

SYSTEM_PROMPT = "You are an expert Roblox Luau scripting assistant. You write correct, idiomatic, well-typed Luau code following current Roblox best practices, and explain your reasoning concisely."

OVERSAMPLE_FACTOR = 4

# --- Legacy-pattern filtering ---
BAD_PATTERNS = [
    r"\bwait\(",
    r":connect\(",
    r"\btick\(\)",
    r"BrickColor\.new\(",
    r"Enum\.FontSize\b",
    r"\.CoordinateFrame\b",
]
GOOD_PATTERNS = [
    r"--!strict",
    r"task\.wait\(",
    r":Connect\(",
    r"\bos\.clock\(\)",
    r": *(number|string|boolean|Instance|Player|BasePart)\b",
]

def modernity_score(text: str) -> int:
    score = 0
    for pat in GOOD_PATTERNS:
        score += len(re.findall(pat, text))
    for pat in BAD_PATTERNS:
        score -= len(re.findall(pat, text))
    return score

def filter_corpus_dataset(ds, min_score: int = -2):
    def keep(example):
        combined_text = example["code"] + "\n" + example.get("chain_of_thought", "")
        return modernity_score(combined_text) >= min_score

    before = len(ds)
    filtered = ds.filter(keep)
    after = len(filtered)
    print(f"Legacy-pattern filter: kept {after}/{before} rows ({after/before:.1%})")
    return filtered

def format_corpus_example(example):
    assistant_content = (
        f"```lua\n{example['code']}\n```\n\n{example['explanation']}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": assistant_content},
        ]
    }

def format_curated_example(example):
    messages = example["messages"]
    normalized = []
    for m in messages:
        role = "assistant" if m["role"] == "model" else m["role"]
        normalized.append({"role": role, "content": m["content"]})
    return {"messages": normalized}

def main():
    # FIXED: Load the datasets from local image disk instead of downloading them at runtime
    print("Loading pre-baked TorpedoSoftware/Roblox-Luau-Reasoning-v1.0 from disk...")
    corpus_ds = load_from_disk("/app/data/corpus")

    print("Filtering legacy-pattern-heavy rows ...")
    corpus_ds = filter_corpus_dataset(corpus_ds, min_score=-2)

    print("Formatting corpus dataset ...")
    corpus_ds = corpus_ds.map(format_corpus_example, remove_columns=corpus_ds.column_names)

    # FIXED: Load your curated dataset from local image disk
    print("Loading pre-baked curated dataset from disk...")
    curated_ds = load_from_disk("/app/data/curated")
    curated_ds = curated_ds.map(format_curated_example, remove_columns=[c for c in curated_ds.column_names if c != "messages"])

    print(f"Oversampling curated set {OVERSAMPLE_FACTOR}x ({len(curated_ds)} -> {len(curated_ds) * OVERSAMPLE_FACTOR}) ...")
    curated_oversampled = concatenate_datasets([curated_ds] * OVERSAMPLE_FACTOR)

    print("Merging and shuffling ...")
    combined = concatenate_datasets([corpus_ds, curated_oversampled])
    combined = combined.shuffle(seed=42)
    print(f"Final combined dataset size: {len(combined)}")

    # FIXED: Load the baked-in model from local path (/app/model) instead of the Hugging Face Hub
    print("Loading baked-in Qwen model from local disk...")
    model, tokenizer = FastModel.from_pretrained(
        model_name="/app/model",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )

    # FIXED: Changed from gemma-4 to qwen-2.5 to match the base model architecture
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        return {"text": text}

    combined = combined.map(apply_template)

    # --- Resume support ---
    from huggingface_hub import HfApi, snapshot_download

    HUB_MODEL_ID = "YOUR_HF_USERNAME/qwen3.8-luau-lora"  # Recommend changing this to reflect your Qwen model
    resume_path = None

    try:
        api = HfApi()
        files = api.list_repo_files(HUB_MODEL_ID)
        checkpoint_dirs = sorted(
            {f.split("/")[0] for f in files if f.startswith("checkpoint-")},
            key=lambda x: int(x.split("-")[1]),
        )
        if checkpoint_dirs:
            latest = checkpoint_dirs[-1]
            print(f"Found existing checkpoint on Hub: {latest} - downloading to resume ...")
            local_dir = snapshot_download(
                repo_id=HUB_MODEL_ID,
                allow_patterns=f"{latest}/*",
            )
            resume_path = f"{local_dir}/{latest}"
            print(f"Will resume from: {resume_path}")
        else:
            print("No prior checkpoint found on Hub - starting fresh.")
    except Exception as e:
        print(f"Could not check for prior checkpoint: {e}")

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=50,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs",
            push_to_hub=True,
            hub_model_id=HUB_MODEL_ID,
            hub_strategy="every_save",
            report_to="none",
        ),
    )

    # FIXED: Replaced Gemma markers with correct Qwen ChatML markers
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("Starting training ...")
    trainer.train(resume_from_checkpoint=resume_path)

    print("Saving final adapter ...")
    model.save_pretrained("final_adapter")
    tokenizer.save_pretrained("final_adapter")

if __name__ == "__main__":
    main()
