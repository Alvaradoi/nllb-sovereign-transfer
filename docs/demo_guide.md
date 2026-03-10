# Demo Guide: Kalispel Salish Translation Prototype

**For:** Kalispel Tribe Language Program
**Prepared by:** Isaac Alvarado
**System:** Fully offline — no internet required

---

## What This Is

This is a prototype computer translation system for Kalispel Salish. It was trained
on Kalispel stories and taught to translate between Salish and English. It is an
early experiment — not a finished product — and all translations should be checked
by a fluent speaker before use in any teaching materials.

---

## Starting the Demo

1. Make sure the computer is on and connected to the external GPU (if used).
2. Open a terminal and navigate to the project folder.
3. Run:

```bash
streamlit run src/app.py
```

4. A browser window will open automatically showing the translation interface.
   (If it does not, open a browser and go to `http://localhost:8501`)

---

## Using the Translation Interface

### Salish → English
1. Type or paste a Kalispel Salish phrase into the text box on the left.
2. Click "Translate."
3. The English translation appears on the right.

### English → Salish
1. Select "English → Salish" from the direction selector.
2. Type or paste an English phrase.
3. Click "Translate."
4. **Important:** A yellow "Needs expert review" flag will appear with the output.
   English → Salish is the harder direction and requires checking by a Kalispel
   language expert before any use.

---

## What to Expect

The system was trained on a small set of Kalispel stories. It works best on:
- Short phrases similar to those in the training stories
- Simple declarative sentences
- Words and patterns from the Vogt texts

It will struggle with:
- Modern vocabulary not in the training texts
- Longer or complex sentences
- Names and proper nouns

This is normal for an early prototype. Each round of improvements adds more training
data and refines the system.

---

## Important Disclaimer

> **English → Salish output requires expert review before use in teaching.**
>
> All translations produced by this system are computer-generated and may contain
> errors. No output should be shared, printed, or used in curriculum materials
> without review and approval by a qualified Kalispel language expert.

---

## Feedback

Please note:
- Any translations that look correct or helpful
- Any translations that are clearly wrong or strange
- Words or phrases you would like the system to learn

Your feedback directly improves the next version of the system.

---

## Technical Notes (for Isaac)

- Model: NLLB-200-distilled-600M + LoRA adapter (r=16)
- Adapter location: `outputs/checkpoints/nllb-kalispel-salish_to_english-r16/final/`
- Base model: `models/nllb-200-600M/`
- All inference is local — no data leaves this machine
