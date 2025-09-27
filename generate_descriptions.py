#!/usr/bin/env python3
"""
generate_descriptions.py

Reads CSV headers (first non-empty row) and generates short,
descriptive text for each header using a small open-source model
that runs locally (default: google/flan-t5-small). Outputs to console
and writes results to output.txt.

Includes a fallback dictionary for common headers to guarantee meaningful output.
"""

import argparse
import csv
import os
import sys
from typing import List

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        AutoModelForCausalLM,
    )
except Exception:
    print("Missing dependencies. Run: pip install transformers torch sentencepiece")
    raise

# Fallback dictionary for common headers
FALLBACK_DICT = {
    "Invoice_ID": "likely a unique identifier for each invoice",
    "Vendor_Name": "the name of the supplier or vendor",
    "Amount": "the payment amount in currency",
    "Payment_Date": "the date the payment was made"
}

def read_headers_from_csv(path: str) -> List[str]:
    """Return the first non-empty row as a list of headers."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and any(cell.strip() for cell in row):
                return [cell.strip() for cell in row if cell.strip()]
    return []

def load_model_and_tokenizer(preferred_model: str = "google/flan-t5-small", device: str = "cpu"):
    """Load seq2seq model or fallback to GPT-2."""
    try:
        print(f"Loading seq2seq model {preferred_model} ... (may download ~100MB)")
        tokenizer = AutoTokenizer.from_pretrained(preferred_model, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(preferred_model)
        model.to(device)
        return tokenizer, model, "seq2seq"
    except Exception as e:
        print(f"Could not load {preferred_model}: {e}")
        print("Falling back to gpt2 (causal).")
        fallback = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(fallback, use_fast=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(fallback)
        model.to(device)
        return tokenizer, model, "causal"

def generate_description(header: str, tokenizer, model, model_type: str, device: str = "cpu") -> str:
    """Generate a short, clear description for a CSV header."""
    header = header.strip()
    if not header:
        return ""

    # Use fallback if available
    if header in FALLBACK_DICT:
        return FALLBACK_DICT[header]

    if model_type == "seq2seq":
        prompt = f"Column name: '{header}'. Write a clear, concise description of what this column likely contains in one short sentence."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            num_return_sequences=1
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    else:
        prompt = f"Column name: '{header}'. Describe clearly what this column likely represents in one sentence:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            num_return_sequences=1
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):].strip()

    # Postprocess: first line, clean punctuation, limit length
    text = text.splitlines()[0].strip(' "\'–-')
    if len(text) > 140:
        text = text[:137].rsplit(" ", 1)[0] + "..."

    # Fallback for repetitive/garbage output
    if len(set(text.replace(" ", ""))) < 5 or text.lower().startswith(header.lower()):
        text = "(could not generate a proper description)"

    return text

def format_results(headers: List[str], descriptions: List[str]) -> List[str]:
    return [f"{h} → {d}" for h, d in zip(headers, descriptions)]

def save_output(lines: List[str], out_path: str = "output.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
    print(f"Wrote {len(lines)} items to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate short descriptions for CSV headers.")
    parser.add_argument("csvfile", nargs="?", help="Path to CSV file (first row used as headers).")
    parser.add_argument("--out", default="output.txt", help="Output text file (default: output.txt)")
    parser.add_argument("--model", default="google/flan-t5-small", help="Hugging Face model name.")
    args = parser.parse_args()

    if not args.csvfile:
        print("No CSV provided — using example headers.")
        headers = ["Invoice_ID", "Vendor_Name", "Amount", "Payment_Date"]
    else:
        headers = read_headers_from_csv(args.csvfile)
        if not headers:
            print("No headers found in CSV. Exiting.")
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer, model, model_type = load_model_and_tokenizer(preferred_model=args.model, device=device)

    results = []
    for h in headers:
        try:
            desc = generate_description(h, tokenizer, model, model_type, device=device)
        except Exception as e:
            print(f"Error generating for header '{h}': {e}")
            desc = "(generation failed)"
        results.append(desc)

    lines = format_results(headers, results)

    print("\nGenerated descriptions:")
    for line in lines:
        print(line)

    save_output(lines, args.out)

if __name__ == "__main__":
    main()
