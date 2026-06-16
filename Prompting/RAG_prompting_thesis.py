"""
Script for RAG prompting.

Generates anemia conclusions for the 10 test cases with greedy decoding. For each case, KNN retrieval is done to select 3 most similar
reference cases from the reference database (gold standard or historic) and puts them into the prompt.
"""

from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import glob
import time
import re
import pandas as pd
import numpy as np

from RAG_retriever import RAGRetriever, build_rag_prompt

# CACHE CONFIGURATION
print("=" * 60)
print("CACHE CONFIGURATION:")
print("=" * 60)
print(f"HF_HOME: {os.getenv('HF_HOME', 'Not set')}")
print(f"HF_HUB_CACHE: {os.getenv('HF_HUB_CACHE', 'Not set')}")
print(f"TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', 'Not set')}")

if not os.getenv("HF_HOME"):
    print("ERROR: HF_HOME environment variable is not set. Please set it to a valid directory.")
    exit(1)
print("=" * 60)

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

# RAG CONFIGURATION
RAG_DIR = os.getenv("ANEMIA_RAG_DIR", "input/RAG")
RAG_REF_NORM_PATH = os.path.join(RAG_DIR, "data_goud_norm_RAG.csv")
RAG_REF_RAW_PATH = os.path.join(RAG_DIR, "Gouden_standaard.csv")
TESTCASES_NORM_PATH = os.path.join(RAG_DIR, "10_testcases_norm_RAG.csv")

print("\nLoading RAG retriever...")
retriever = RAGRetriever(
    ref_norm_path=RAG_REF_NORM_PATH,
    ref_raw_path=RAG_REF_RAW_PATH,
    top_k=3,
)
print("RAG retriever loaded.")

testcases_norm_df = pd.read_csv(TESTCASES_NORM_PATH)
testcases_norm_df["UniekLabnummer"] = testcases_norm_df["UniekLabnummer"].astype(str)

# PROMPTING
INPUT_DIR = os.getenv("ANEMIA_INPUT_DIR", "input/10testcases")
OUTPUT_DIR = os.getenv("ANEMIA_OUTPUT_DIR", "output/RAG_goud")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def extract_id(filename: str) -> str:
    """Extract ID from filename"""
    match = re.search(r"gen_data_(.+?)(?:\.\w+)?$", filename)
    if match:
        return match.group(1)
    match = re.search(r"(\d+)", filename)
    if match:
        return match.group(1)
    return os.path.splitext(filename)[0]


def strip_thinking(text: str) -> str:
    """Remove thinking pattern from output."""
    match = re.search(r"\*\*Conclusion:\*\*\s*", text)
    if match:
        text = text[match.end() :]
    return text.strip()


# LOOP OVER ALL TXT FILES
txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
print(f"\nFound {len(txt_files)} .txt files")

results_summary = []
for file_path in txt_files:
    print(f"\nProcessing: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        user_input = f.read().strip()

    patient_id = extract_id(os.path.basename(file_path))
    print(f"  UniekLabnummer: {patient_id}")

    norm_rows = testcases_norm_df[testcases_norm_df["UniekLabnummer"] == patient_id]
    current_matches = []

    if norm_rows.empty:
        print(f"  WARNING: UniekLabnummer '{patient_id}' niet gevonden in {TESTCASES_NORM_PATH}. Geen RAG-context.")
        rag_context = ""
    else:
        testcase_norm_row = norm_rows.iloc[0]
        rag_context = retriever.get_context(testcase_norm_row)
        query = np.array([float(testcase_norm_row.get(c, np.nan)) for c in retriever.feature_cols], dtype=float)
        distances = retriever._manhattan_with_penalty(query)
        matched_ids = retriever.retrieve(testcase_norm_row)
        print(f"  RAG: {len(matched_ids)} referentiecasus(sen) gevonden.")
        for uid in matched_ids:
            idx = retriever._ref_ids.index(uid)
            dist_value = distances[idx]
            match_str = f"ID: {uid} | Afstand: {dist_value:.2f}"
            print(f"    -> {match_str}")
            current_matches.append(match_str)

    results_summary.append({"casus": patient_id, "matches": list(current_matches)})

    messages = build_rag_prompt(
        user_input=user_input,
        rag_context=rag_context,
        system_prompt=SYSTEM_PROMPT,
    )

    # Debugging only!
    print("\n" + "=" * 80)
    print("VOLLEDIGE PROMPT VOOR PATIËNT: " + patient_id)
    print("=" * 80)
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n--- {role} ---\n{content}")
    print("\n" + "=" * 80 + "\n")

    out = pipe(messages, max_new_tokens=500, do_sample=False, return_full_text=False)
    raw = out[0]["generated_text"]

    if isinstance(raw, list):
        raw = raw[-1]["content"]
    elif isinstance(raw, dict):
        raw = raw.get("content", str(raw))

    conclusion = strip_thinking(raw)

    output_file = os.path.join(OUTPUT_DIR, f"{patient_id}_medgemma27b_rag_goud.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(conclusion)

    print(f"  Saved: {output_file}")

# EINDRAPPORTAGE
print("\n" + "=" * 60)
print("OVERZICHT RAG-RESULTATEN")
print("=" * 60)

if not results_summary:
    print("Geen resultaten om weer te geven.")
else:
    for res in results_summary:
        print(f"\nCasus [{res['casus']}]:")
        if res["matches"]:
            for m in res["matches"]:
                print(f"  - Referentie: {m}")
        else:
            print("  - Geen referentiecasussen gevonden (ID niet in database of afstand > max_distance).")

print("\n" + "=" * 60)
print("Script succesvol voltooid.")
