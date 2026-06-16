"""This script is used for prompt engineering with MedGemma 4B, Phi-4 1.5 14B, and MedGemma 27B.

It generates anemia conclusions for the 10 test cases using greedy decoding. Models are loaded in either 4-bit, 8-bit, or full precision.
Directories are placeholder names.

The system prompt here is the one for zero-shot prompting, the same script was used for few-shot prompting with a different system
prompt.

Note that this script is an edit of an existing MUMC+ script
"""

from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import os
import glob
import time
import re

# CACHE CONFIGURATION
print("=" * 60)
print("CACHE CONFIGURATION:")
print("=" * 60)
print(f"HF_HOME: {os.getenv('HF_HOME', 'Not set')}")
print(f"HF_HUB_CACHE: {os.getenv('HF_HUB_CACHE', 'Not set')}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', 'Not set')}")

# MODEL CONFIGURATION
BASE_MODEL_ID = "google/medgemma-27b-text-it"  # or google/medgemma-4b-1.5-it or microsoft/phi-4
USE_4BIT = True

model_kwargs = {
    "attn_implementation": "eager",
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16
model_kwargs["torch_dtype"] = torch_dtype

if USE_4BIT:
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch.uint8,
    )

# CACHE VERIFICATION
hf_cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
print(f"\nCache directory: {hf_cache_dir}")
print(f"Directory exists: {os.path.exists(hf_cache_dir)}")
if os.path.exists(hf_cache_dir):
    print(f"Directory is writable: {os.access(hf_cache_dir, os.W_OK)}")
    hub_dir = os.path.join(hf_cache_dir, "hub")
    if os.path.exists(hub_dir):
        cached_models = [
            d for d in os.listdir(hub_dir) if os.path.isdir(os.path.join(hub_dir, d)) and d.startswith("models--")
        ]
        print(f"\nCached models found: {len(cached_models)}")
        for model in cached_models:
            print(f"  - {model}")
        # Controleer of het specifieke model gecached is
        model_folder = BASE_MODEL_ID.replace("/", "--")
        model_cache = os.path.join(hub_dir, f"models--{model_folder}")
        if os.path.exists(model_cache):
            print(f"\nModel gevonden in cache: {model_cache}")
        else:
            print(f"\nWARNING: Model '{BASE_MODEL_ID}' NIET gevonden in cache.")
            print(f"  Verwachte locatie: {model_cache}")
            print(f"  Het model wordt gedownload van HuggingFace Hub.")
    else:
        print(f"No hub directory at {hub_dir}")
else:
    print(f"WARNING: Cache directory does not exist!")

# LOAD MODEL
print("\nLoading tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
print(f"Tokenizer loaded. (took {time.time() - start_time:.2f}s)")

print("Loading model...")
start_time = time.time()
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
print(f"Model loaded. (took {time.time() - start_time:.2f}s)")

print("Creating pipeline...")
start_time = time.time()
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True)
print(f"Pipeline created. (took {time.time() - start_time:.2f}s)")

# PROMPTING
INPUT_DIR = os.getenv("ANEMIA_INPUT_DIR", "input/10testcases")
OUTPUT_DIR = os.getenv("ANEMIA_OUTPUT_DIR", "output/zeroshot")  # edit here when running oneshot or threeshot

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_id(filename: str) -> str:
    match = re.search(r"(\d+)", filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0]  # fallback


def strip_thinking(text: str) -> str:  # remove thinking pattern
    return re.sub(r"<unused\d+>.*?<unused\d+>", "", text, flags=re.DOTALL).strip()


# LOOP OVER ALLE TXT BESTANDEN
txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

print(f"Found {len(txt_files)} .txt files")

for file_path in txt_files:
    print(f"\nProcessing: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        user_input = f.read().strip()

    patient_id = extract_id(os.path.basename(file_path))

    messages = [
        {
            "role": "system",
            "content": """Je bent een deskundige klinisch chemicus die beknopte rapportages schrijft voor huisartsen over anemie.
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
""",
        },
        {"role": "user", "content": user_input},
    ]

    # GENERATE
    out = pipe(messages, max_new_tokens=1000, do_sample=False)

    raw = out[0]["generated_text"][-1]["content"]
    conclusion = strip_thinking(raw)

    # SAVE OUTPUT
    output_file = os.path.join(OUTPUT_DIR, f"{patient_id}_zeroshot_medgemma27b.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(conclusion)

    print(f"Saved: {output_file}")
