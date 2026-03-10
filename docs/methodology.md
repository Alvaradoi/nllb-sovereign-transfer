# Methodology: Sovereignty-First Neural Machine Translation for Kalispel Salish

**Project:** nllb-sovereign-transfer
**Author:** Isaac Alvarado, Arizona State University — Master of Computer Science
**Affiliation:** Kalispel Tribal Member; Kalispel Tribe Language Program
**Advisor:** [Advisor Name]
**Status:** Prototype — active development

---

## 1. Research Problem

The Kalispel (Pend d'Oreilles) language is a critically endangered Interior Salish
language spoken by members of the Kalispel Tribe of Indians (Usk, Washington) and
the Confederated Salish and Kootenai Tribes. Fewer than a handful of fully fluent
first-language speakers remain. The Kalispel Tribe Language Program, led by Language
Director JR Bluff, is actively pursuing revitalization through documentation, curriculum
development, and technology-assisted learning.

Neural Machine Translation (NMT) offers potential value in this context — both as a
productivity tool for language workers and as a demonstration that modern NLP can
serve Indigenous communities. However, existing NMT systems present two critical
problems for endangered language contexts:

1. **Data scarcity.** State-of-the-art NMT requires millions of parallel sentence
   pairs. Kalispel has fewer than a few hundred documented parallel utterances in
   the public record.

2. **Data sovereignty.** Standard NLP workflows rely on cloud APIs, remote training
   infrastructure, and public model hubs — all of which require transmitting
   linguistic data to third parties. For a tribal language community operating under
   Indigenous data governance principles (OCAP, CARE), this is unacceptable. The
   tribe must retain full control over its linguistic heritage.

This project addresses both problems through: (a) transfer learning from a
multilingual pretrained model to reduce the data requirement, and (b) a fully
offline, locally-executed pipeline that never transmits linguistic data beyond
the tribe's own computing infrastructure.

---

## 2. Research Questions

1. Can parameter-efficient fine-tuning (LoRA) of a large multilingual NMT model
   produce useful Kalispel Salish ↔ English translations from fewer than 300
   parallel utterances?

2. What data pipeline architecture is required to extract, clean, and structure
   interlinear glossed text from digitized historical linguistic fieldwork into
   usable NMT training pairs?

3. How can data sovereignty constraints (OCAP/CARE) be enforced at the
   infrastructure level — not merely as policy — in an NLP research pipeline?

---

## 3. Source Data

The primary training data source is:

> Camp, K. (2007). *An Interlinear Analysis of Seven Kalispel Texts From Hans Vogt.*
> Master's thesis, University of Montana.

Camp (2007) presents seven Kalispel narrative texts originally recorded by Hans Vogt
in the 1930s–40s, with full five-tier interlinear glossing:

- **Tier 1:** Kalispel surface form (original orthography)
- **Tier 2:** Normalized Kalispel (segmented phonological representation)
- **Tier 3:** Morpheme segmentation (morpheme boundaries marked with `-` and `=`)
- **Tier 4:** Morpheme-by-morpheme gloss (Leipzig Glossing Rules, abbreviated labels)
- **Tier 5:** Free English translation

This format — Interlinear Glossed Text (IGT) — is standard in linguistic fieldwork
but presents significant extraction challenges when sourced from PDF. See
`docs/data_pipeline.md` for the full extraction methodology.

The texts cover approximately 311 utterances across seven narrative texts (labeled
I through VIII), spanning themes of travel, encounter, and daily life. All texts
were originally elicited from Kalispel-speaking informants by Vogt.

**Data governance note:** Camp (2007) is a published academic thesis. The Kalispel
Tribe Language Program is the custodian of all derived artifacts (trained models,
processed JSONL pairs, character repair maps). No derived data is stored in any
public repository. See `docs/data_governance.md`.

---

## 4. Approach: Transfer Learning from NLLB-200

### 4.1 Base Model

We use Meta AI's **NLLB-200-distilled-600M** (No Language Left Behind) as the
base model. NLLB-200 is a multilingual sequence-to-sequence transformer trained
on 200 languages with approximately 40 billion tokens of parallel data. The
distilled 600M variant is selected for:

- Compatibility with consumer GPU hardware (8GB VRAM, NVIDIA RTX 3070 Ti)
- Strong multilingual representations, particularly for morphologically complex languages
- Permissive FLORES license for research use

Kalispel Salish is not among NLLB-200's 200 languages. However, the model has
been trained on numerous Native American languages (Cherokee, Navajo, Quechua)
and morphologically rich languages (Finnish, Turkish, Georgian) whose structural
properties partially overlap with Interior Salish languages. The hypothesis is that
cross-lingual transfer from this rich multilingual space will outperform training
from scratch on the available data.

### 4.2 Parameter-Efficient Fine-Tuning (LoRA)

Full fine-tuning of a 600M parameter model on 246 training pairs would cause
severe overfitting and require more VRAM than the target hardware provides.

We apply **Low-Rank Adaptation (LoRA)** (Hu et al., 2021), which freezes the
base model's weights and inserts trainable low-rank decomposition matrices into
selected attention and feed-forward layers. This reduces the trainable parameter
count from 600M to approximately 8.65M (1.39% of total), enabling:

- Training on 8GB VRAM without quantization
- Faster iteration (10 epochs in ~43 seconds on RTX 3070 Ti)
- Catastrophic forgetting mitigation — the base multilingual representations are preserved
- Adapter portability — the adapter file (~33MB) is separable from the base weights

**LoRA configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 16 | Balance between expressiveness and parameter count |
| Alpha | 32 | Scaling factor (alpha/r = 2.0 — standard) |
| Target modules | q_proj, v_proj, k_proj, out_proj, fc1, fc2 | Full attention + FFN coverage |
| Dropout | 0.05 | Light regularization for low-data regime |
| Bias | none | Standard for seq2seq |

### 4.3 Language Code Handling

NLLB-200's tokenizer uses BCP-47 language codes (e.g., `eng_Latn`, `fra_Latn`).
Kalispel Salish (`kal_Latn`) is not in the vocabulary. For the Salish→English
direction, we set the forced BOS (beginning-of-sequence) token to `eng_Latn`
(token id 256047), which correctly steers generation into English. The LoRA
adapters learn to map Kalispel surface forms into the model's latent space
regardless of the source-side language tag. This is a known technique in
zero-shot cross-lingual transfer (Artetxe & Schwartz, 2019).

---

## 5. Evaluation

**Metrics:**
- **sacreBLEU** — standard MT evaluation metric; scores 0–100, higher is better.
  Low absolute values are expected for extremely low-resource pairs.
- **ChrF** — character n-gram F-score; more informative than BLEU for
  morphologically complex target languages and short sentences.

**Dev/test split:** 80/10/10 deterministic split (seed=42) of 306 valid pairs.
30 dev pairs used for epoch-level evaluation; 30 test pairs reserved for
final held-out evaluation after hyperparameter decisions are frozen.

**Baseline:** Untrained NLLB-200-600M (zero-shot) on dev set. To be documented
in a future log entry.

---

## 6. Infrastructure and Sovereignty Enforcement

All computation runs locally on:

- **CPU:** AMD Ryzen (host machine)
- **GPU:** NVIDIA RTX 3070 Ti (8GB GDDR6X)
- **OS:** Ubuntu Linux
- **Python:** 3.12, venv isolated environment
- **Key libraries:** PyTorch 2.10 (CUDA 12.8), HuggingFace Transformers 5.3,
  PEFT 0.18, MLflow 3.10 (local tracking)

**Sovereignty enforcement is structural, not merely policy:**

- No cloud SDK imports (`boto3`, `openai`, `google-cloud-*`, etc.) in any source file
- `local_files_only=True` on all model load calls — runtime cannot reach Hugging Face Hub
- `report_to=[]` in training arguments — disables all external experiment trackers
- `data/`, `models/`, `outputs/`, `mlruns/` are all `.gitignore`d — cannot be staged
- Pre-commit sovereignty audit script verifies no restricted data or prohibited packages are staged before any commit

---

## 7. Limitations and Future Work

1. **Training data volume.** 246 pairs is insufficient for production-quality MT.
   Future work includes annotating additional Kalispel text sources (Giorda 1877,
   Vogt 1940, tribal curriculum materials) and applying data augmentation.

2. **Evaluation validity.** sacreBLEU and ChrF are designed for high-resource
   languages. Results should be interpreted qualitatively and validated by the
   Kalispel Tribe Language Program rather than treated as absolute performance measures.

3. **Orthography.** Camp (2007) uses a non-standard orthography derived from Vogt.
   A mapping to the modern tribal Unicode standard would improve tokenizer alignment
   and allow integration with existing curriculum materials.

4. **Bidirectional training.** The current prototype focuses on Salish→English.
   English→Salish is architecturally supported but requires additional expert
   validation before any output is used in teaching contexts.

5. **Morphology-aware tokenization.** NLLB's SentencePiece tokenizer was not trained
   on Kalispel. A custom BPE/morpheme-aware tokenizer trained on the tribal lexicon
   could significantly improve subword segmentation quality.

---

## 8. References

- Camp, K. (2007). *An Interlinear Analysis of Seven Kalispel Texts From Hans Vogt.*
  University of Montana.
- Costa-jussa, M. R., et al. (2022). No Language Left Behind: Scaling Human-Centered
  Machine Translation. *arXiv:2207.04672.*
- Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  *arXiv:2106.09685.*
- First Nations Information Governance Centre. (2014). *Ownership, Control, Access
  and Possession (OCAP): The Path to First Nations Information Governance.* FNIGC.
- Carroll, S. R., et al. (2020). The CARE Principles for Indigenous Data Governance.
  *Data Science Journal, 19*(1), 43.
- Artetxe, M., & Schwartz, H. (2019). Massively Multilingual Sentence Embeddings
  for Zero-Shot Cross-Lingual Transfer. *TACL.*
- Vogt, H. (1940). *The Kalispel Language.* Oslo: Det Norske Videnskaps-Akademi.
- Giorda, J. (1877). *A Dictionary of the Kalispel or Flat-head Indian Language.*
  St. Ignatius Mission Print.
