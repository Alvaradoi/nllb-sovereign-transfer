# Data Pipeline: PDF Extraction to JSONL Training Pairs

**Source:** Camp (2007) — University of Montana Master's Thesis (PDF)
**Output:** `data/processed/camp2007_{train,dev,test}.jsonl`
**Status:** Complete — 306 valid pairs extracted

---

## Overview

The pipeline transforms a scanned/digitally-typeset PDF of interlinear glossed
linguistic fieldwork into structured JSONL training pairs suitable for NMT fine-tuning.
It consists of two major stages:

```
Camp2007.pdf
    |
    v
[Stage 1: PDF Extraction]  src/extract_pdf.py
    |   - Font CMap patch (SIL custom glyph names)
    |   - Control character repair
    |   - Superscript/footnote stripping
    |   - Page-level text assembly
    v
camp_raw.txt  (raw extracted text, ~8,000 lines)
    |
    v
[Stage 2: Interlinear Parsing]  src/data_cleaner.py
    |   - Page range filter (pp. 15-111)
    |   - Line classification (8-way)
    |   - Translation-anchored block assembly
    |   - Text ID recovery
    |   - JSONL serialization + train/dev/test split
    v
camp2007_train.jsonl (246 pairs)
camp2007_dev.jsonl   (30 pairs)
camp2007_test.jsonl  (30 pairs)
rejection_log.jsonl  (5 rejected)
```

---

## Stage 1: PDF Extraction (`src/extract_pdf.py`)

### 1.1 Font Encoding Problem

Camp (2007) uses a custom TrueType font (`DXCOJF+TTFF4A94D8t00`, hereafter R40)
for Kalispel Salish text. This font was produced with SIL (Summer Institute of
Linguistics) font tools and encodes Salish-specific characters (ejectives, glottal
stops, labialization, combining diacritics) using Private Use Area (PUA) codepoints
and custom glyph names.

When PyMuPDF (the extraction library) encounters the font's `ToUnicode` CMap, it
finds that only one character is mapped. For the remaining glyphs, PyMuPDF attempts
lookup in the Adobe Glyph List (AGL), which does not recognize SIL custom names.
The result: nearly all Salish characters are extracted as U+FFFD (replacement
character) or control characters.

**Glyph name to Unicode mapping (discovered via PDF xref analysis):**

| CID | Glyph Name | Unicode | Character |
|-----|-----------|---------|-----------|
| 4 | apostrophesupnosp | U+02BC | ʼ (ejective apostrophe) |
| 9 | commasuprightnosp | U+02BC | ʼ (ejective, variant glyph) |
| 16 | something | U+02B7 | ʷ (labialization) |
| 21 | wsuper | U+02B7 | ʷ (labialization) |
| 24 | haceknosp | U+030C | ̌ (combining caron, builds x̌ ǩ) |

### 1.2 Font CMap Patch (`patch_pdf_fonts()`)

Rather than post-processing extracted text, we patch the PDF's font encoding in
memory before extraction:

1. Iterate all font objects in the PDF via `doc.xref_object()`
2. Parse the `Encoding /Differences` array to find custom glyph names
3. Map glyph names to Unicode via `CUSTOM_GLYPH_UNICODE` dictionary
4. Generate a new `ToUnicode` CMap stream mapping CIDs to correct Unicode codepoints
5. Inject the CMap via `doc.update_stream(tou_xref, cmap_bytes)`

This approach is applied once before any text extraction, so all downstream code
works with correctly decoded Unicode text.

**Extraction results after patch:**

| Character | Count | Significance |
|-----------|-------|--------------|
| ʷ (U+02B7) | 3,390 | Labialization — phonemically contrastive |
| ʼ (U+02BC) | 4,792 | Ejective — phonemically contrastive |
| x̌-caron (U+030C) | 1,132 | Combines to form x̌ ǩ (retroflex fricatives) |
| ɬ (U+026C) | 2,084 | Lateral fricative |
| ƛ (U+019B) | 518 | Lateral affricate |
| ʔ (U+0294) | 3,362 | Glottal stop |
| U+FFFD remaining | 88 | Back-matter only (pp. 112-146, not in training data) |

### 1.3 Character Repair Layers

After the CMap patch, a five-layer repair pipeline handles remaining artifacts:

- **Layer 1:** Control character repair — maps specific ASCII control codes that
  appear mid-word due to encoding issues (e.g., `\x15` NAK → ʷ in some font variants)
- **Layer 2:** PUA repair map — maps any remaining Unicode Private Use Area codepoints
  (U+E000–U+F8FF) to their correct Unicode equivalents
- **Layer 3:** CID artifact repair — handles `(cid:N)` literal strings that survive
  some PDF extraction paths
- **Layer 4:** NFC normalization — ensures composed Unicode forms throughout
- **Layer 5:** Suspicious control audit — flags any unrecognized control characters
  for manual review

### 1.4 Superscript and Footnote Stripping

Camp (2007) uses footnote numbers as superscripts throughout the text. These appear
at 6.5–7pt font size, compared to 9–9.5pt for body text and small-caps gloss
abbreviations. A size ratio threshold (`SUPERSCRIPT_SIZE_RATIO = 0.72`) correctly
strips footnote numerals without removing small-caps abbreviations (e.g., PASS, OBL).

Footnote text blocks are identified via `FOOTNOTE_ANCHOR_RE` pattern matching and
excluded from the main extraction stream.

### 1.5 Y-Tolerance Line Merging

NLLB text in PDFs is sometimes split across multiple spans at slightly different
y-coordinates due to superscript placement. A `Y_TOLERANCE = 6.0pt` threshold
merges spans within the same visual line.

---

## Stage 2: Interlinear Parsing (`src/data_cleaner.py`)

### 2.1 Interlinear Glossed Text Structure

Camp (2007) presents text in five-tier Interlinear Glossed Text (IGT) format,
following the Leipzig Glossing Rules:

```
Tier 1  číʔcəntəm tčiná sqaltəmíxʷ          [Salish surface form]
Tier 2  číʔcəntəm t činá t sqaltəmíxʷ        [Normalized Salish]
Tier 3  čiʔc -nt -m t č=naqs t s+qlt+mixʷ    [Morpheme segmentation]
Tier 4  arrive.PL-TR-PASS OBL one.person OBL  [Morpheme gloss]
Tier 5  A man called on us.                    [Free English translation]
```

Morpheme boundaries use: `-` for suffixes, `=` for clitics, `+` for compounding.

### 2.2 Line Classifier (`classify()`)

Each extracted line is assigned one of eight labels:

| Label | Criteria | Examples |
|-------|----------|---------|
| `EMPTY` | Blank or whitespace only | |
| `PAGE_MARKER` | Divider line or `PAGE N` header | `────────`, `PAGE 42` |
| `SKIP` | Standalone number (page reference) | `12`, `7` |
| `TEXT_ID` | Roman numeral + dot + number pattern | `II.005`, `VIII`, `.007` |
| `SALISH` | Contains Salish phonological characters | Lines with ʷ ʔ ɬ č š etc. |
| `MORPHEME` | Salish chars + morpheme boundaries (`-` `=` `+`) | `čiʔc -nt -m` |
| `GLOSS` | Uppercase Leipzig abbreviations (PASS, OBL, TR, etc.) | `arrive.PL-TR-PASS OBL` |
| `TRANSLATION` | English sentence (ends in `.!?:`, no gloss markers) | `A man called on us.` |
| `FOOTNOTE` | Citation patterns, linguistic terminology, >280 chars | Long prose lines |

Key disambiguation rules:
- Typographic curly apostrophes (U+2018/U+2019) are excluded from Salish character
  detection — prevents English prose with smart quotes from being misclassified.
- Lines containing Salish characters plus a dictionary reference pattern `(NNN)` are
  classified as `FOOTNOTE` — Salish words quoted within footnote prose.
- Lowercase-starting lines with no gloss markers are accepted as `TRANSLATION` —
  Camp (2007) uses lowercase continuation translations for split utterances.
- Complete text IDs (matching full `ROMAN.NNN` pattern) bypass fragment accumulation
  to prevent doubling (e.g., `VV.001`).

### 2.3 Text ID Assembly

Text IDs appear as `II.005` — Roman numeral text number, dot, utterance number.
Across page breaks, the ID components may be split across pages. The `try_assemble_id()`
function assembles up to two fragments in either order:

- `"II"` + `".005"` → `"II.005"`
- `".005"` + `"II"` → `"II.005"` (reversed — common at page breaks)

### 2.4 Translation-Anchored Backward Window

The key algorithmic insight: rather than forward-streaming (which merges utterances
when translations are missing), we anchor on each `TRANSLATION` line and scan
backward through the classified line list to collect the preceding context.

The backward scan collects:
- `SALISH` and `MORPHEME` lines → accumulated as candidate pairs
- `GLOSS` lines → stored as morpheme gloss
- `TEXT_ID` → recorded for the utterance
- Stops at the previous `TRANSLATION` line (exclusive)

This makes each translation claim its own backward context, eliminating inter-utterance
bleeding that affected earlier forward-pass approaches.

**Known limitation:** When a translation line is entirely missing from an utterance
(transcription gap), the backward window expands past that utterance's content into
the preceding utterance. All output pairs are flagged `"verified": false` pending
review by a Kalispel language expert.

### 2.5 Record Validation

Each assembled pair is validated before output:

- `salish` must be non-empty
- `salish` must contain at least one non-ASCII character
- `salish` length must not exceed 1,000 characters
- Both `salish` and `english` must be present

Rejected pairs are written to `rejection_log.jsonl` with reason codes — no silent drops.

### 2.6 Output Schema

```json
{
  "salish": "ɬuʔ wiʔstés",
  "english": "when he had finished it",
  "source": "camp2007",
  "text_id": "V.009",
  "morpheme_gloss": "ɬuʔ wy -st -es",
  "verified": false
}
```

- `salish` — tier 2 (normalized) preferred; falls back to tier 1 (surface) if absent
- `morpheme_gloss` — tier 3 + tier 4 concatenated; null if not present
- `verified` — false for all automatic pairs; requires Language Director sign-off
- Unicode normalization: NFC applied to all Salish text

### 2.7 Pipeline Statistics

| Metric | Value |
|--------|-------|
| Total utterance blocks parsed | 311 |
| Valid pairs produced | 306 |
| Rejected (no Salish surface) | 5 |
| Pairs with recovered text IDs | 67 |
| Text ID range | II.005 – VIII.011 |
| Train / Dev / Test split | 246 / 30 / 30 |
| Average Salish tokens per pair | 4.4 |
| Average English tokens per pair | 6.9 |
| Pages processed | 15 – 111 |
| Pages excluded (back-matter) | 112 – 146 |

---

## Running the Pipeline

```bash
# Stage 1: PDF extraction
python src/extract_pdf.py \
    --input data/raw/Camp2007.pdf \
    --output data/raw/camp_raw.txt \
    --start-page 15 \
    --audit

# Stage 2: Interlinear parsing → JSONL
python src/data_cleaner.py \
    --input data/raw/ \
    --output data/processed/

# Audit PUA characters on a specific page
python src/extract_pdf.py \
    --input data/raw/Camp2007.pdf \
    --output /dev/null \
    --pua-audit --debug-page 20
```
