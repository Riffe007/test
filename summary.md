# Gemma 3-1B-PT ExecuTorch Project Summary + Runbook + Known Issues

---

# Executive Summary

This project successfully exported the Hugging Face local snapshot of **Gemma 3-1B-PT** into an **ExecuTorch `.pte` artifact** for portable on-device inference preparation.

The model was validated locally using PyTorch, exported using `torch.export`, lowered into an ExecuTorch Edge program, and emitted as a working `.pte` file.

This establishes the initial deployment baseline for future optimization work including:

* XNNPACK delegation
* INT8 / INT4 quantization
* ETDump profiling
* Runtime performance benchmarking
* Mobile deployment validation

Current state:

**Portable CPU FP32 export complete**

This is the required foundation before advanced optimization work.

---

# Project Root

```bash
cd /home/timothy_riffe/Documents/projects/llm
```

---

# Model Information

## Model

* Model: Gemma 3-1B-PT
* Type: Base pretrained causal language model
* Source: Local Hugging Face snapshot
* Instruction tuned: No (base PT model, not instruction-tuned)

## Local Model Path

```text
/home/timothy_riffe/Documents/projects/llm/models/gemma-3-1b-pt
```

---

# Deliverables Completed

## Completed Successfully

* Local model validation
* Tokenizer verification
* PyTorch inference baseline
* ExecuTorch export pipeline
* Portable FP32 `.pte` artifact
* Artifact smoke test
* Artifact metadata report
* Export summary report
* Full project runbook
* Known issues documentation

## Artifact

### Output Directory

```text
output/models/gemma_3_1b_pt_optimum/
```

### Current Artifact

```text
gemma_3_1b_pt_portable_fp32.pte
```

### Artifact Size

```text
3814.83 MB
```

### Runtime Target

```text
Portable CPU FP32
```

---

# Python Environment

## Activate Environment

```bash
cd /home/timothy_riffe/Documents/projects/llm
source .venv/bin/activate
python --version
```

Expected:

```text
Python 3.11.x
```

---

# JSON Configuration

## Config File

```text
export/configs/llm/config_gemma_3_1b_pt_optimum.json
```

## Validate Before Running

```bash
python -m json.tool export/configs/llm/config_gemma_3_1b_pt_optimum.json
```

Purpose:

Avoid runtime failures caused by malformed JSON.

---

# Local Model Load Test

## File

```text
export/llm/test_gemma_load.py
```

## Run

```bash
python export/llm/test_gemma_load.py
```

## Purpose

Confirms:

* tokenizer loads
* model loads
* local inference works

This validates the local Hugging Face snapshot before export.
