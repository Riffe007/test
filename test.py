import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(42)

CONFIG_PATH = Path("export/configs/llm/config_gemma_3_1b_pt_optimum.json")
RESULTS_DIR = Path("evaluation/gemma_3_1b_pt/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with CONFIG_PATH.open() as f:
        cfg = json.load(f)

    model_id = cfg["model_id"]
    prompt = "The reason the sky appears blue is"

    print(f"Loading tokenizer from: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading model from: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    ).eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    print("Running PyTorch baseline generation...")
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    end = time.perf_counter()

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    latency_sec = end - start

    result = {
        "model": "gemma_3_1b_pt",
        "runtime": "pytorch_fp32",
        "prompt": prompt,
        "generated_text": text,
        "latency_seconds": latency_sec,
        "max_new_tokens": 200,
        "dtype": "float32",
        "use_cache": False,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
        "seed": 42,
    }

    out_file = RESULTS_DIR / "pytorch_baseline_results.json"
    out_file.write_text(json.dumps(result, indent=2))

    print(text)
    print(f"Latency seconds: {latency_sec:.4f}")
    print(f"Wrote: {out_file}")


if __name__ == "__main__":
    main()
