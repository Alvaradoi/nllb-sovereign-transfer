# nllb-sovereign-transfer
A privacy-first, offline Neural Machine Translation (NMT) system for low-resource languages using NLLB-200 600M. Implements a local Transfer Learning pipeline designed for Data Sovereignty and Cultural Preservation.

# Sovereign NLLB Transfer Learning (PoC)

## Mission
This project establishes a **Data Sovereign** Neural Machine Translation (NMT) pipeline for the **Kalispel** language (Salish family). It fine-tunes Meta's **NLLB-200 600M** model using Low-Rank Adaptation (LoRA) or full transfer learning, designed specifically to run on consumer hardware (NVIDIA 3070 Ti) without sending a single byte of training data to the cloud.

## Architecture
The system is architected for **Offline-First** execution to ensure cultural heritage data remains in the custody of the tribe.

* **Base Model:** `facebook/nllb-200-distilled-600M`
* **Technique:** Transfer Learning on a custom parallel corpus.
* **Infrastructure:** Local Training Loop (Hugging Face `Trainer` + `accelerate`) optimized for 8GB VRAM.
* **Data Pipeline:** "Agentic" cleaning workflow using local LLMs (Llama 3 / Qwen) for OCR correction.

## Data Sovereignty & Ethics
**Crucial Note:** This repository contains **code only**.
The linguistic datasets (dictionaries, recordings, and parallel texts) used to train this model are the sovereign property of the Kalispel Tribe and are **strictly excluded** from this public repository via `.gitignore` policies.

To reproduce this work, you must provide your own parallel dataset in `JSONL` format.

## Usage (Bring Your Own Data)

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/nllb-sovereign-transfer.git](https://github.com/YOUR_USERNAME/nllb-sovereign-transfer.git)
pip install -r requirements.txt
