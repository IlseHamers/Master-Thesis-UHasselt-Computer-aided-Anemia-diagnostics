"""
This script fine-tunes a large language model (LLM) using QLoRA (Quantized
Low-Rank Adaptation).

Note that this file is an adaptation of an existing MUMC+ script.

use:
    python finetune.py --config config.yaml (eerst naar map navigeren)
"""

import os
import time
from datetime import datetime

import yaml
import torch
import pandas as pd
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


# UTILITY FUNCTIONS
def read_yaml(path: str) -> dict:
    """
    Read and parse a YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    """
    Create a directory if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)


def pick_compute_dtype() -> torch.dtype:
    """
    Select the optimal compute data type for the current GPU.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def normalize_target_modules(x):
    """
    Normalize the target_modules parameter for LoRA configuration.
    """
    if isinstance(x, str):
        s = x.strip()
        # Handle comma-separated lists in YAML
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
        return s
    return x


# DATA FORMATTING

META_COLS = {"UniekLabnummer", "Geslacht", "Leeftijd", "Beschrijving"}

SYSTEM_PROMPT = """Je bent een deskundige klinisch chemicus die beknopte rapportages schrijft voor huisartsen over anemie.
Beoordeel op basis van patiëntgegevens en laboratoriumwaarden of sprake is van anemie. Formuleer een medische conclusie van maximaal 100 woorden.

Richtlijnen:
- Begin altijd met "Anemie-protocol bij [leeftijd]-jarige [man/vrouw]."
- Primaire conclusie: stel vast of er wel of geen sprake is van anemie op basis van de Hemoglobine-waarde, rekening houdend met geslacht.
  * Referentie man: geen anemie bij hemoglobine van 8.2 mmol/L of hoger
  * Referentie vrouw: geen anemie bij hemoglobine van 7.3 mmol/L of hoger
- Indien geen anemie: schrijf EXACT alleen: "Anemie-protocol bij [leeftijd]-jarige [man/vrouw]. Geen anemie." Dit is de volledige conclusie.
- Indien anemie: specificeer het type (bijv. absolute/reactieve ijzeranemie, renaal bepaald, vitamine B12/foliumzuur deficiëntie, etc.) en of het microcytair, normocytair of macrocytair is op basis van MCV.
- Stijl: professionele medische terminologie begrijpelijk voor een huisarts. Wees feitelijk.

Controleer vóór het geven van de output:
Indien "Geen anemie." voorkomt → output mag NIETS na deze zin bevatten.
Indien er tekst na "Geen anemie." staat → verwijder deze.
"""

# Column display order
COL_ORDER = [
    "Hemoglobine",
    "Hematocriet",
    "MCV",
    "Reticulocyten absoluut",
    "Reticulocyten relatief",
    "Trombocyten",
    "Leucocyten",
    "Erytrocyten",
    "Bezinking",
    "Kreatinine",
    "eGFR CKD-EPI",
    "ALAT",
    "LD",
    "NT-proBNP",
    "Transferrine",
    "Transf.sat.",
    "Ferritine",
    "IJzer",
    "CRP",
    "Haptoglobine",
    "TSH",
    "Vitamine B12",
    "Foliumzuur",
]

# Reference ranges per parameter and sex.
REF: dict[str, dict[str, tuple]] = {
    "Hemoglobine": {"M": (8.2, 11.0), "V": (7.3, 9.7)},
    "Hematocriet": {"M": (0.41, 0.52), "V": (0.36, 0.48)},
    "MCV": {"M": (80, 100), "V": (80, 100)},
    "Reticulocyten absoluut": {"M": (20, 90), "V": (20, 90)},
    "Reticulocyten relatief": {"M": (2, 20), "V": (2, 20)},
    "Trombocyten": {"M": (130, 350), "V": (130, 350)},
    "Leucocyten": {"M": (3.5, 11.0), "V": (3.5, 11.0)},
    "Erytrocyten": {"M": (4.20, 5.60), "V": (3.70, 5.00)},
    "Bezinking": {"M": (0, 14), "V": (0, 19)},
    "Kreatinine": {"M": (60, 115), "V": (50, 100)},
    "eGFR CKD-EPI": {"M": (90, None), "V": (90, None)},
    "ALAT": {"M": (None, 44), "V": (None, 33)},
    "LD": {"M": (None, 247), "V": (None, 246)},
    "NT-proBNP": {"M": (None, 14.9), "V": (None, 14.9)},
    "Transferrine": {"M": (1.50, 3.50), "V": (1.50, 3.50)},
    "Transf.sat.": {"M": (20.0, 45.0), "V": (20.0, 45.0)},
    "Ferritine": {"M": (30, 400), "V": (15, 200)},
    "IJzer": {"M": (14.0, 27.0), "V": (11.0, 25.0)},
    "CRP": {"M": (None, 9.9), "V": (None, 9.9)},
    "Haptoglobine": {"M": (0.25, 1.90), "V": (0.25, 1.90)},
    "TSH": {"M": (0.65, 6.7), "V": (0.65, 6.7)},
    "Vitamine B12": {"M": (145, 569), "V": (145, 569)},
    "Foliumzuur": {"M": (8.0, 60.8), "V": (8.0, 60.8)},
}


def _ref_str(col: str, geslacht: str) -> str:
    """Return a compact reference range string, e.g. '8.2–11.0', '≥90', or '—'."""
    # Strip units in parentheses to match REF keys
    base = col.split(" (")[0].strip()
    if base not in REF:
        return "—"
    key = "M" if str(geslacht).strip().upper() == "M" else "V"
    low, high = REF[base][key]
    if low is None and high is None:
        return "—"
    if low is None:
        return f"≤{high}"
    if high is None:
        return f"≥{low}"
    return f"{low}–{high}"


def row_to_prompt(row: pd.Series) -> str:
    """
    Convert a patient row to a Markdown table with reference ranges,
    in the clinical column order defined by COL_ORDER.
    """
    geslacht = str(row.get("Geslacht", "")).strip()
    present = [c for c in row.index if c not in META_COLS]

    def base(col):
        return col.split(" (")[0].strip()

    ordered = [c for o in COL_ORDER for c in present if base(c) == o]
    extras = [c for c in present if base(c) not in COL_ORDER]
    lab_cols = ordered + extras

    lines = [
        f"Patiënt: {row.get('Leeftijd', 'onbekend')}j, {geslacht}",
        "",
        "|Naam |Resultaat |Normaalwaarde |",
        "|-----------|--------|------------|",
    ]
    for col in lab_cols:
        val = row[col]
        val_str = "NA" if pd.isna(val) else str(val)
        ref = _ref_str(col, geslacht)
        lines.append(f"| {col} | {val_str} | {ref} |")

    return "\n".join(lines)


def build_dataset(csv_path: str, text_field: str, output_col: str, tokenizer) -> Dataset:
    """
    Build a HuggingFace Dataset from the CSV/Excel.
    Each example = system prompt + user message (lab table) + assistant conclusion.
    """
    p = csv_path
    df = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p, quotechar='"', quoting=0)

    missing = {"Geslacht", "Leeftijd", "Beschrijving"} - set(df.columns)
    if missing:
        raise ValueError(f"Data file missing required columns: {missing}")

    print(f"Loaded {len(df)} training examples from {csv_path}")

    texts = []
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row_to_prompt(row)},
            {"role": "assistant", "content": str(row[output_col]).strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)

    return Dataset.from_dict({text_field: texts})


# MODEL LOADING
def load_base_model(model_cfg: dict, quant_cfg: dict):
    """
    Load the base model with 4-bit quantization (27b).
    """
    model_id = model_cfg["base_model_id"]
    compute_dtype = pick_compute_dtype()

    # Model loading arguments
    kwargs = dict(
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        low_cpu_mem_usage=bool(model_cfg.get("low_cpu_mem_usage", True)),
        attn_implementation="eager",
        torch_dtype=compute_dtype,
    )

    if quant_cfg.get("use_8bit", False):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif quant_cfg.get("use_4bit", True):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=bool(quant_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(quant_cfg.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_storage=compute_dtype,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=kwargs["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    return model, tokenizer, compute_dtype


# MAIN TRAINING FUNCTION
def train(cfg_path: str):
    """
    Main training function
    """
    cfg = read_yaml(cfg_path)

    if not os.getenv("HF_HOME"):
        raise RuntimeError(
            "HF_HOME is not set. Set it to a writable cache directory.\n" "Example: export HF_HOME=/path/to/cache"
        )

    # -------------------------------------------------------------------------
    # STEP 1: Setup paths and output directory
    # -------------------------------------------------------------------------
    data_dir = cfg["paths"]["data_dir"]
    csv_path = os.path.join(data_dir, cfg["paths"]["csv_name"])

    # Create timestamped output directory for this training run
    base_out = os.getenv("OUT_BASE_DIR", cfg["paths"]["out_base_dir"])
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(base_out, f"qlora-medgemma-{run_id}")
    ensure_dir(out_dir)

    print(f"Output directory: {out_dir}")

    # -------------------------------------------------------------------------
    # STEP 3: Load the base model with quantization
    # -------------------------------------------------------------------------
    print(f"Loading base model: {cfg['model']['base_model_id']}")
    t0 = time.time()
    model, tokenizer, compute_dtype = load_base_model(cfg["model"], cfg.get("quantization", {}))
    print(f"Model loaded in {time.time() - t0:.1f}s (dtype={compute_dtype})")

    # Configure tokenizer for training
    # Right padding is standard for training (left padding for inference)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # -------------------------------------------------------------------------
    # STEP 2: Load and prepare the dataset
    # -------------------------------------------------------------------------
    text_field = cfg["prompting"]["dataset_text_field"]
    output_col = cfg["prompting"]["output_column"]

    dataset = build_dataset(csv_path, text_field, output_col, tokenizer)
    print(f"Loaded dataset: {len(dataset)} training examples")

    # -------------------------------------------------------------------------
    # STEP 4: Configure LoRA adapters
    # -------------------------------------------------------------------------
    # LoRA (Low-Rank Adaptation) adds small trainable matrices to the model
    # Instead of training all 27B parameters, we train only ~1% of them

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),  # Rank: higher = more capacity, more memory
        lora_alpha=int(lora_cfg["lora_alpha"]),  # Scaling factor (usually same as r)
        lora_dropout=float(lora_cfg["lora_dropout"]),  # Regularization
        bias=str(lora_cfg["bias"]),  # "none" = don't train bias terms
        task_type=str(lora_cfg["task_type"]),  # "CAUSAL_LM" for text generation
        target_modules=normalize_target_modules(lora_cfg["target_modules"]),
    )

    # -------------------------------------------------------------------------
    # STEP 5: Configure the trainer
    # -------------------------------------------------------------------------
    # SFTTrainer (Supervised Fine-Tuning Trainer) from TRL library
    # handles the training loop, logging, and checkpointing

    tr_cfg = cfg["training"]
    sft_args = SFTConfig(
        output_dir=out_dir,
        # Training duration
        num_train_epochs=int(tr_cfg["epochs"]),  # How many times to go through the data
        # Batch size and gradient accumulation
        # Effective batch size = batch_size * grad_accum
        # Larger effective batch = more stable training but more memory
        per_device_train_batch_size=int(tr_cfg["batch_size"]),
        gradient_accumulation_steps=int(tr_cfg["grad_accum"]),
        # Learning rate schedule
        learning_rate=float(tr_cfg["learning_rate"]),  # How fast to learn (2e-4 is typical for LoRA)
        lr_scheduler_type=str(tr_cfg["lr_scheduler_type"]),  # "cosine" = gradually decrease LR
        warmup_ratio=float(tr_cfg["warmup_ratio"]),  # Gradually increase LR at start
        # Logging and saving
        logging_steps=int(tr_cfg["logging_steps"]),  # Log every N steps
        save_strategy=str(tr_cfg["save_strategy"]),  # "epoch" = save after each epoch
        # Optimizer
        optim=str(tr_cfg["optim"]),  # "paged_adamw_8bit" = memory-efficient optimizer
        # Misc
        report_to="none",  # Disable wandb/tensorboard logging
        seed=int(tr_cfg["seed"]),  # For reproducibility
        # Dataset configuration
        dataset_text_field=text_field,
        max_length=int(tr_cfg["max_length"]),  # Max tokens per example
        packing=bool(tr_cfg["packing"]),  # Pack multiple examples into one sequence
        # Precision
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
    )

    # Create the trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        peft_config=peft_config,  # This tells the trainer to use LoRA
        processing_class=tokenizer,
    )

    # -------------------------------------------------------------------------
    # STEP 6: Train!
    # -------------------------------------------------------------------------
    print("\nStarting training...")
    trainer.train()

    # -------------------------------------------------------------------------
    # STEP 7: Save the trained adapters
    # -------------------------------------------------------------------------
    # We only save the LoRA adapters, not the full model
    # The adapters are small (~50-200MB) and can be loaded on top of the base model

    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\nTraining complete!")
    print(f"Saved QLoRA adapter + tokenizer to: {out_dir}")
    print(f"\nTo run inference, use:")
    print(f"  python inference.py --config config.yaml --adapter_dir {out_dir}")


# ENTRY POINT
if __name__ == "__main__":
    import argparse

    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Fine-tune MedGemma using QLoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune.py --config config.yaml
  CONFIG_PATH=/app/config.yaml python finetune.py
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.getenv("CONFIG_PATH", "config.yaml"),
        help="Path to YAML configuration file (default: config.yaml or CONFIG_PATH env var)",
    )

    args = parser.parse_args()
    train(args.config)
