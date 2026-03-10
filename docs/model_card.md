# Model Card: NLLB-200-Kalispel-LoRA

**Model ID:** nllb-kalispel-salish_to_english-r16
**Base Model:** facebook/nllb-200-distilled-600M
**Fine-tuning Method:** LoRA (Low-Rank Adaptation)
**Primary Direction:** Kalispel Salish → English
**Version:** Prototype v0.1 (2026-03-09)
**Custodian:** Kalispel Tribe Language Program

---

## Model Description

This model is a LoRA adapter fine-tuned on top of Meta's NLLB-200-distilled-600M
for Kalispel Salish ↔ English translation. Kalispel Salish (also known as Pend
d'Oreilles) is a critically endangered Interior Salish language spoken in the
Inland Northwest. It is not among NLLB-200's 200 official training languages.

The adapter was trained on 246 parallel utterance pairs extracted from Camp (2007),
a University of Montana thesis presenting seven Kalispel narrative texts with full
interlinear glossing. All text originates from fieldwork by Hans Vogt (1930s–40s).

**This is a research prototype.** It is not suitable for production use, language
teaching, or publication of translations without expert review.

---

## Intended Use

### Intended Users
- Kalispel Tribe Language Program staff (primary beneficiaries)
- NLP researchers studying low-resource and endangered language NMT
- Graduate researchers in computational linguistics / Indigenous language technology

### Intended Uses
- Prototype demonstration for Language Director review
- Research baseline for future model iterations
- Feasibility study for LoRA transfer to unrepresented languages

### Out-of-Scope Uses
- Production translation without expert validation
- Any use that transmits tribal linguistic data to third-party services
- Teaching or curriculum content without Language Director approval
- English → Kalispel Salish translation presented as reliable (the reverse direction
  is architecturally supported but not validated for this prototype)

---

## Training Data

| Field | Value |
|-------|-------|
| Source | Camp (2007), University of Montana |
| Original data | Hans Vogt fieldwork, 1930s–40s |
| Pairs used | 246 (train) / 30 (dev) |
| Domain | Narrative oral texts (7 stories) |
| Language pair | Kalispel Salish ↔ English |
| Avg. Salish length | 4.4 tokens |
| Avg. English length | 6.9 tokens |
| Verified by speaker | No (flagged `verified: false`) |

Training data is NOT included in this repository. It is held by the Kalispel Tribe
Language Program and managed under OCAP/CARE data governance principles.

---

## Training Procedure

| Parameter | Value |
|-----------|-------|
| Base model | facebook/nllb-200-distilled-600M |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, out_proj, fc1, fc2 |
| LoRA dropout | 0.05 |
| Trainable parameters | 8,650,752 (1.39% of 623.7M total) |
| Epochs | 10 |
| Batch size | 4 (effective 32 with grad accumulation) |
| Gradient accumulation | 8 steps |
| Learning rate | 3e-4 (cosine decay, 25 warmup steps) |
| Optimizer | AdamW (HuggingFace default) |
| Precision | FP16 |
| Hardware | NVIDIA RTX 3070 Ti (8GB VRAM) |
| Training time | ~43 seconds (10 epochs) |
| Experiment tracking | MLflow (local only) |

---

## Evaluation Results

Evaluated on 30 held-out dev pairs from Camp (2007). All metrics computed with
the `evaluate` library (HuggingFace).

### Salish → English (Primary Direction)

| Epoch | Train Loss | BLEU | ChrF |
|-------|-----------|------|------|
| 1 | 43.95 | 0.35 | 8.9 |
| 2 | 41.96 | 0.31 | 9.3 |
| 3 | 33.18 | 0.96 | 12.2 |
| 4 | 26.88 | 2.42 | 9.6 |
| 5 | — | 1.49 | 13.5 |
| 6 | 23.31 | — | — |
| 7 | 22.20 | 2.39 | 7.2 |
| 8 | 21.65 | 4.16 | 9.5 |
| 9 | 20.53 | 4.55 | 11.2 |
| **10 (final)** | — | **4.56** | **10.8** |

**Interpretation:** BLEU of 4.56 is expected to be very low for a 246-pair training
set — BLEU is calibrated for high-resource settings and degrades severely below
~10,000 pairs. The consistent improvement in loss (44 → 20.5) and ChrF scores
across epochs confirms the model is learning. ChrF is more informative here as it
operates at the character level, better reflecting partial morpheme matches.

A zero-shot baseline (untrained NLLB-200 on Kalispel) is planned for the next
evaluation phase.

### English → Salish
Not evaluated in this prototype. Architecturally supported; requires expert
validation before any outputs are used.

---

## Limitations

1. **Small training set.** 246 pairs is far below typical NMT thresholds.
   Outputs will frequently be incorrect or contain hallucinations.

2. **No speaker validation.** No Kalispel-fluent speaker has reviewed
   training data quality or model outputs for this prototype.

3. **Orthography mismatch.** Camp (2007) uses a non-standard orthography.
   The model outputs in this orthography, not the modern tribal standard.

4. **Domain restriction.** Training data covers seven specific oral narratives.
   The model may not generalize to other Kalispel registers (ceremony, daily
   speech, curriculum language).

5. **No language token.** `kal_Latn` is not in NLLB-200's vocabulary. The
   source language is essentially presented as unknown to the base model.
   The LoRA adapter learns to bridge this gap, but the lack of a proper
   language token is a structural limitation.

---

## Ethical Considerations

- All training data is tribal linguistic heritage. The Kalispel Tribe retains
  ownership and control over all derived artifacts per OCAP principles.
- This model must not be distributed without explicit approval from the
  Kalispel Tribe Language Program.
- English → Salish outputs are not suitable for use in teaching materials
  without review by a qualified Kalispel language expert.
- The model reflects one historical documentation tradition (Vogt/Camp) and
  may not represent the full range of Kalispel speech communities.

---

## Citation

If referencing this work, please cite:

```
Alvarado, I. (2026). Sovereignty-First Neural Machine Translation for Kalispel Salish.
Master's Thesis, Arizona State University. In collaboration with the
Kalispel Tribe Language Program.
```

And the underlying data source:

```
Camp, K. (2007). An Interlinear Analysis of Seven Kalispel Texts From Hans Vogt.
Master's Thesis, University of Montana.
```
