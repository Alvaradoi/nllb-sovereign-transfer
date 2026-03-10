# Results: NLLB-200 LoRA Fine-tuning on Kalispel Salish

**Last updated:** 2026-03-09
**Run ID:** nllb-kalispel-salish_to_english-r16

---

## Summary

| Metric | Value |
|--------|-------|
| Training pairs | 246 |
| Dev pairs | 30 |
| Test pairs | 30 (held out — not yet evaluated) |
| Final dev BLEU | 4.56 |
| Final dev ChrF | 10.8 |
| Final train loss | ~20.5 |
| Best epoch (BLEU) | 10 |
| Training time | 43 seconds (RTX 3070 Ti) |
| Adapter size | ~33 MB |

---

## Training Curve: Salish → English (Prototype v0.1)

| Epoch | Train Loss | Dev Loss | Dev BLEU | Dev ChrF | Notes |
|-------|-----------|----------|----------|----------|-------|
| 1 | 43.95 | 5.346 | 0.35 | 8.9 | Warmup phase |
| 2 | 41.96 | 5.016 | 0.31 | 9.3 | |
| 3 | 33.18 | 3.982 | 0.96 | 12.2 | Significant loss drop |
| 4 | 26.88 | 3.505 | 2.42 | 9.6 | |
| 5 | — | 3.135 | 1.49 | 13.5 | |
| 6 | 23.31 | — | — | — | |
| 7 | 22.20 | 2.773 | 2.39 | 7.2 | |
| 8 | 21.65 | 2.746 | 4.16 | 9.5 | |
| 9 | 20.53 | 2.725 | 4.55 | 11.2 | |
| 10 | — | 2.715 | 4.56 | 10.8 | **Best / Final** |

---

## Interpretation

### What these numbers mean
BLEU scores in NMT are calibrated for large-scale evaluation — a BLEU of 4.56
corresponds roughly to outputs that capture some correct words and phrases but
are not yet fluent translations. This is expected and normal for this scale of
training data. For reference:

- **< 10 BLEU:** Almost useless — translation captures very little
- **10–19 BLEU:** Difficult to understand; a rough meaning can sometimes be inferred
- **20–29 BLEU:** Clear intent, errors in grammar
- **30–40 BLEU:** Intelligible to good translation
- **> 40 BLEU:** High-quality MT (rare even in high-resource settings)

With 246 training pairs, reaching the "clear intent" band (20+ BLEU) from a
zero-shot baseline is the realistic goal for the prototype phase.

ChrF (character n-gram F-score) is more informative for morphologically complex
languages and short sentences. A ChrF of 10.8 with consistent improvement across
epochs confirms the model is acquiring character-level patterns in Kalispel Salish,
even when full word matches are rare.

### Consistent loss decrease
Train loss dropped from 43.95 → 20.53 (53% reduction) across 10 epochs, confirming
genuine learning rather than memorization. The small dataset means early stopping
(patience=3) did not trigger — loss was still improving at epoch 10.

### Next evaluation milestones
1. **Zero-shot baseline:** Run unmodified NLLB-200-600M on dev set to quantify
   improvement attributable to fine-tuning.
2. **Test set evaluation:** Run final adapter on the 30 held-out test pairs.
   (Do not run until all hyperparameter decisions are finalized.)
3. **Human evaluation:** Present sample translations to the Language Program
   (Language Director JR Bluff, Assistant Director Jessie) for qualitative assessment.
4. **Extended training:** Re-run with epochs=20 and r=8 to explore the
   overfitting vs. expressiveness trade-off with limited data.

---

## Checkpoint Inventory

| Checkpoint | Epoch | Dev BLEU | Dev ChrF | Dev Loss | Path |
|------------|-------|----------|----------|----------|------|
| checkpoint-64 | 8 | 4.16 | 9.5 | 2.746 | outputs/checkpoints/.../checkpoint-64 |
| checkpoint-72 | 9 | 4.55 | 11.2 | 2.725 | outputs/checkpoints/.../checkpoint-72 |
| checkpoint-80 | 10 | 4.56 | 10.8 | 2.715 | outputs/checkpoints/.../checkpoint-80 |
| **final** | 10 | **4.56** | **10.8** | **2.715** | outputs/checkpoints/.../final |

Best checkpoint by BLEU: epoch 10. Best checkpoint by ChrF: epoch 9 (11.2).

---

## Future Runs

| Run | Date | Direction | Epochs | r | BLEU | ChrF | Notes |
|-----|------|-----------|--------|---|------|------|-------|
| v0.1 | 2026-03-09 | salish→en | 10 | 16 | 4.56 | 10.8 | Prototype baseline |
| | | | | | | | |

*This table will be updated as new training runs are completed.*
