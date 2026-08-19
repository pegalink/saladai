"""
train.py - runs on the Salad GPU node.

Loads TorpedoSoftware/Roblox-Luau-Reasoning-v1.0 (~15.3K rows) + our curated
308-example dataset from the Hub, filters out the most legacy-pattern-heavy
rows from the corpus-derived set, oversamples our curated set so it isn't
statistically drowned out, formats everything into chat messages, and trains
Gemma 4 12B with Unsloth QLoRA. Pushes checkpoints to HF Hub as it goes.
"""

import os
import re
import random
from datasets import load_dataset, concatenate_datasets
from huggingface_hub import login
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

login(token=os.environ["HF_TOKEN"])

SYSTEM_PROMPT = "You are an expert Roblox Luau scripting assistant. You write correct, idiomatic, well-typed Luau code following current Roblox best practices, and explain your reasoning concisely."

# CHANGE THIS to match what you used in push_dataset_to_hub.py
CURATED_REPO_ID = "DiamantPetko/luau-curated-308"

OVERSAMPLE_FACTOR = 4  # repeat our clean 277 curated examples this many times

# --- Legacy-pattern filtering for the corpus-derived dataset ---
BAD_PATTERNS = [
    r"\bwait\(",              # bare wait() instead of task.wait()
    r":connect\(",             # lowercase legacy event API
    r"\btick\(\)",              # legacy timing, prefer os.clock()/os.time()
    r"BrickColor\.new\(",       # legacy color API, prefer Color3
    r"Enum\.FontSize\b",        # deprecated
    r"\.CoordinateFrame\b",     # legacy camera property, now Camera.CFrame
]
GOOD_PATTERNS = [
    r"--!strict",
    r"task\.wait\(",
    r":Connect\(",
    r"\bos\.clock\(\)",
    r": *(number|string|boolean|Instance|Player|BasePart)\b",  # rough type-annotation signal
]

def modernity_score(text: str) -> int:
    score = 0
    for pat in GOOD_PATTERNS:
        score += len(re.findall(pat, text))
    for pat in BAD_PATTERNS:
        score -= len(re.findall(pat, text))
    return score

def filter_corpus_dataset(ds, min_score: int = -2):
    """Drop rows whose code+CoT skew heavily toward deprecated patterns.
    min_score is intentionally lenient - this is a rough bias correction,
    not a hard modern-only filter, or we'd lose most of the corpus."""
    def keep(example):
        combined_text = example["code"] + "\n" + example.get("chain_of_thought", "")
        return modernity_score(combined_text) >= min_score

    before = len(ds)
    filtered = ds.filter(keep)
    after = len(filtered)
    print(f"Legacy-pattern filter: kept {after}/{before} rows ({after/before:.1%})")
    return filtered

def format_corpus_example(example):
    """TorpedoSoftware format -> chat messages.
    Per Unsloth's own Gemma 4 guidance: 'For most production assistants, the
    simplest setup is to fine-tune on the final visible answer only.' So we
    deliberately drop chain_of_thought here rather than trying to preserve it -
    simpler, matches their recommended default, and avoids the ambiguity of
    hand-guessing thinking-mode formatting."""
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
    """Our dataset already has messages with role 'model' - normalize to 'assistant'
    to match the corpus-derived formatting, since your tokenizer's chat template
    expects a consistent role name."""
    messages = example["messages"]
    normalized = []
    for m in messages:
        role = "assistant" if m["role"] == "model" else m["role"]
        normalized.append({"role": role, "content": m["content"]})
    return {"messages": normalized}

def main():
    print("Loading TorpedoSoftware/Roblox-Luau-Reasoning-v1.0 ...")
    corpus_ds = load_dataset("TorpedoSoftware/Roblox-Luau-Reasoning-v1.0", split="train")

    print("Filtering legacy-pattern-heavy rows ...")
    corpus_ds = filter_corpus_dataset(corpus_ds, min_score=-2)

    print("Formatting corpus dataset ...")
    corpus_ds = corpus_ds.map(format_corpus_example, remove_columns=corpus_ds.column_names)

    print(f"Loading curated dataset from {CURATED_REPO_ID} ...")
    curated_ds = load_dataset(CURATED_REPO_ID, split="train")
    curated_ds = curated_ds.map(format_curated_example, remove_columns=[c for c in curated_ds.column_names if c != "messages"])

    print(f"Oversampling curated set {OVERSAMPLE_FACTOR}x ({len(curated_ds)} -> {len(curated_ds) * OVERSAMPLE_FACTOR}) ...")
    curated_oversampled = concatenate_datasets([curated_ds] * OVERSAMPLE_FACTOR)

    print("Merging and shuffling ...")
    combined = concatenate_datasets([corpus_ds, curated_oversampled])
    combined = combined.shuffle(seed=42)
    print(f"Final combined dataset size: {len(combined)}")

    # --- Model + tokenizer ---
    print("Loading Gemma 4 26B (downloads at container start, not build time) ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name="unsloth/gemma-4-26B-A4B-it",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,      # text-only for this use case
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=42,
    )

    # Non-thinking template - matches the "final answer only" formatting above.
    # Use "gemma-4-thinking" instead only if you reformat examples to include
    # visible chain-of-thought consistently (Unsloth explicitly warns against
    # mixing thinking/non-thinking formats in the same dataset).
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    def apply_template(example):
        # Unsloth's own recipe strips the leading <bos> here since the
        # tokenizer adds exactly one automatically - leaving it in causes a
        # duplicate <bos> token during training.
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        return {"text": text}

    combined = combined.map(apply_template)

    # --- Resume support: check the Hub for a checkpoint from a prior run,
    # since Salad nodes on lower priority tiers can be reallocated mid-training
    # and local disk does NOT survive that - only what's already been pushed
    # to the Hub is safe.
    from huggingface_hub import HfApi, snapshot_download

    HUB_MODEL_ID = "YOUR_HF_USERNAME/gemma4-luau-lora"  # must match hub_model_id below
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
        print(f"Could not check for prior checkpoint (probably first run): {e}")

    # --- Train ---
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=combined,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=2,  # keep low - corpus-derived rows can dominate with more epochs
            learning_rate=2e-4,
            logging_steps=10,
            save_steps=50,  # more frequent saves - minimize lost progress on reallocation
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

    # Only compute loss on the assistant's actual response, not the user turn -
    # standard practice, and specifically documented for Gemma 4 by Unsloth.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    print("Starting training ...")
    trainer.train(resume_from_checkpoint=resume_path)

    print("Saving final adapter ...")
    model.save_pretrained("final_adapter")
    tokenizer.save_pretrained("final_adapter")

if __name__ == "__main__":
    main()
