"""
This script loads a fine-tuned model and generates responses to new inputs.

use:
------
    # Use the latest trained adapter
    python inference.py --config config.yaml

    # Use a specific adapter directory
    python inference.py --config config.yaml --adapter_dir /path/to/adapter

    # Run base model without adapter (for comparison)
    python inference.py --config config.yaml --no_adapter

Note that this script is an edit of an existing MUMC+ script
"""

import os
import re
from datetime import datetime
from typing import Optional

import yaml
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

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


# UTILITY FUNCTIONS
def read_yaml(path: str) -> dict:
    """
    Read and parse a YAML configuration file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_compute_dtype() -> torch.dtype:
    """
    Select the optimal compute data type for the current GPU.
    """
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def extract_id(filename: str) -> str:
    """Extract numeric patient ID from filename, fallback to stem."""
    match = re.search(r"(\d+)", filename)
    return match.group(1) if match else os.path.splitext(filename)[0]


# Regex to match adapter directory names
# Format: qlora-medgemma-YYYYMMDD-HHMMSS
RUN_RE = re.compile(r"^qlora-medgemma-(\d{8}-\d{6})$")


def find_latest_adapter_dir(base_out: str) -> str:
    """
    Find the most recent adapter directory in the output folder.
    """
    if not os.path.isdir(base_out):
        raise FileNotFoundError(f"Output base dir not found: {base_out}")

    candidates = []
    for name in os.listdir(base_out):
        m = RUN_RE.match(name)
        if not m:
            continue

        ts = m.group(1)
        try:
            dt = datetime.strptime(ts, "%Y%m%d-%H%M%S")
        except ValueError:
            dt = datetime.min

        path = os.path.join(base_out, name)

        has_adapter = any(
            os.path.exists(os.path.join(path, f))
            for f in ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        )

        candidates.append((has_adapter, dt, os.path.getmtime(path), path))

    if not candidates:
        raise FileNotFoundError(
            f"No adapter run dirs found in {base_out}.\n"
            f"Expected directories named: qlora-medgemma-YYYYMMDD-HHMMSS\n"
            f"Run finetune.py first to create an adapter."
        )

    # Sort by: has_adapter (True first), timestamp (newest first), mtime (newest first)
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return candidates[0][3]


# MODEL LOADING
def load_base_model(model_cfg: dict, quant_cfg: dict):
    """
    Load the base model with quantization.
    """
    model_id = model_cfg["base_model_id"]
    compute_dtype = pick_compute_dtype()

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


# INFERENCE
def generate(
    model,
    tokenizer,
    user_input: str,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
) -> str:
    """Run greedy inference for a single pre-formatted patient input."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    if getattr(tokenizer, "chat_template", None):
        enc = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    else:
        flat = f"### Instructie:\n{SYSTEM_PROMPT}\n\n" f"### Labwaarden:\n{user_input}\n### Antwoord:\n"
        enc = tokenizer(flat, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(model.device)

    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = out_ids[0][prompt_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    return raw


# MAIN
def run(cfg_path: str, adapter_dir: Optional[str], no_adapter: bool, max_new_tokens: int, repetition_penalty: float):

    cfg = read_yaml(cfg_path)
    base_out = os.getenv("OUT_BASE_DIR", cfg["paths"]["out_base_dir"])

    input_dir = cfg["paths"]["input_dir"]
    output_dir = cfg["paths"]["output_dir"]
    suffix = cfg["paths"].get("output_suffix", "qlora")
    os.makedirs(output_dir, exist_ok=True)

    if no_adapter:
        adapter_dir = None
    elif adapter_dir is None:
        adapter_dir = find_latest_adapter_dir(base_out)
        print(f"Auto-detected adapter: {adapter_dir}")

    model, tokenizer, _ = load_base_model(cfg["model"], cfg.get("quantization", {}))

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    print(f"Found {len(txt_files)} .txt files in {input_dir}")

    for file_path in txt_files:
        print(f"Processing: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            user_input = f.read().strip()

        patient_id = extract_id(os.path.basename(file_path))
        conclusion = generate(model, tokenizer, user_input, max_new_tokens, repetition_penalty)

        output_file = os.path.join(output_dir, f"{patient_id}_{suffix}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(conclusion)

        print(f"Saved: {output_file}")

    print("Done.")


# ENTRY POINT
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inference — base or fine-tuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=os.getenv("CONFIG_PATH", "config_anemia.yaml"))
    parser.add_argument(
        "--adapter_dir", type=str, default=None, help="Path to adapter folder. Auto-detects latest if omitted."
    )
    parser.add_argument("--no_adapter", action="store_true", help="Run base model without adapter (for comparison).")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    args = parser.parse_args()
    run(
        cfg_path=args.config,
        adapter_dir=args.adapter_dir,
        no_adapter=args.no_adapter,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )
