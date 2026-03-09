#!/usr/bin/env python3
"""
data_cleaner.py -- Parse Camp (2007) interlinear Kalispel Salish texts
                   into clean JSONL sentence pairs for NMT training.

Interlinear structure per utterance (5 tiers):
  Tier 1: Salish surface form (original orthography from Vogt 1940)
  Tier 2: Normalized Salish (word-by-word, clean spaces)
  Tier 3: Morpheme segmentation (Salish chars + '+' / '=' boundaries)
  Tier 4: Morpheme gloss (English abbreviations + morpheme meanings)
  Tier 5: Free English translation

Output JSONL schema (one object per line):
  {
    "salish":         str,         # NFC-normalized tier-2 surface
    "english":        str,         # Free translation (tier 5)
    "source":         "camp2007",
    "text_id":        str | null,  # e.g. "II.005", null if unrecovered
    "morpheme_gloss": str,         # Tier-4 gloss, empty string if unavailable
    "verified":       false
  }

Usage:
  python src/data_cleaner.py --input data/raw/ --output data/processed/
  python src/data_cleaner.py --input data/raw/camp_raw.txt --output data/processed/
"""

import argparse
import json
import logging
import random
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Character / pattern constants
# ---------------------------------------------------------------------------

# Salish-specific IPA characters reliable for language identification.
SALISH_MARKERS: frozenset[str] = frozenset("ʔɬƛʼəčšŋʷɣχʕɴ")

# Typographic characters that appear in English prose but NOT in Salish text.
# These must be excluded from the "has non-ASCII → must be Salish" heuristic.
ENGLISH_UNICODE: frozenset[str] = frozenset(
    "\u2018\u2019"   # curly single quotes  ' '
    "\u201C\u201D"   # curly double quotes  " "
    "\u2013\u2014"   # en-dash, em-dash
    "\u2026"         # ellipsis
    "\u00A0"         # non-breaking space
)

# Dictionary / lexicon reference in footnotes: "(156)" or "(1940a:82)"
DICT_REF_RE = re.compile(r"\(\d{2,4}\)")

# Uppercase gloss abbreviations used in Kalispel/Interior Salish interlinears.
_ABBREVS = (
    "PASS|ART|TR|TRANS|SG|PL|INTR|FUT|POSS|NEG|OBJ|CONT|ACT|PART|OBL|"
    "LOC|DEM|INCH|COLL|STAT|CAUS|PERF|INTS|RECIP|REFL|CISLOC|CUST|OPT|"
    "PRON|REDUP|EMPH|FOC|TOP|IRR|IPFV|PFV|QUOT|EVAL|INTS|MED|DIST"
)
GLOSS_ABBREV_RE = re.compile(rf"\b({_ABBREVS})\b")

# Footnote / scholarly prose signals
CITATION_RE = re.compile(r"\(\d{4}[a-z]?[:.]\d")   # (1940a:80)
LINGUISTIC_RE = re.compile(
    r"prefix|suffix|infix|morpheme|typesett|segment|gloss|orthograph|"
    r"cognate|lexical|analyz|analogi|proclitic|enclitic|reduplicat|"
    r"\bplural\b|\bin the translation\b|\bKalispel\b|\bVogt\b|\bCamp\b",
    re.IGNORECASE,
)

# Page structure markers produced by extract_pdf.py
PAGE_DIVIDER_RE = re.compile(r"^─{10,}$")
PAGE_HEADER_RE  = re.compile(r"^PAGE \d+$")
PAGE_NUMBER_RE  = re.compile(r"^PAGE (\d+)$")   # captures the page number

# Standalone integer on its own line = page number or footnote reference
STANDALONE_NUM_RE = re.compile(r"^\d{1,3}$")

# Text IDs in two forms:
#   Complete:  II.005   III.002
#   Fragment:  II       .005
COMPLETE_ID_RE = re.compile(r"^([IVX]{1,5})\.(\d{1,3})$")
ROMAN_ONLY_RE  = re.compile(r"^([IVX]{1,5})$")
DOTNUM_ONLY_RE = re.compile(r"^\.(\d{1,3})$")

# Source-file stem → source label for JSONL
SOURCE_MAP: dict[str, str] = {
    "camp_raw": "camp2007",
    "vogt_raw": "vogt1940",   # placeholder for future sources
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def has_salish(line: str) -> bool:
    """
    True if line contains Salish IPA characters or non-ASCII that is
    NOT a typographic English character (curly quotes, dashes, etc.).
    """
    if any(c in SALISH_MARKERS for c in line):
        return True
    return any(ord(c) > 127 and c not in ENGLISH_UNICODE for c in line)


def count_gloss_abbrevs(line: str) -> int:
    return len(GLOSS_ABBREV_RE.findall(line))


def has_dot_in_word(line: str) -> bool:
    """Detect gloss-style compound glosses: 'arrive.PL', '1SG.POSS'."""
    return bool(re.search(r"[A-Za-z0-9]\.[A-Z]", line))


def has_person_number(line: str) -> bool:
    """Detect glosses like '1SG', '3TRANS', '2PL'."""
    return bool(re.search(r"\b[123](SG|PL|TRANS)\b", line))


# ---------------------------------------------------------------------------
# Line classifier
# ---------------------------------------------------------------------------

LineKind = str  # one of the string constants below

EMPTY       = "empty"
PAGE_MARKER = "page_marker"
SKIP        = "skip"           # standalone page/footnote numbers
TEXT_ID     = "text_id"
SALISH      = "salish"         # tier 1 or 2 (surface; no morpheme '+')
MORPHEME    = "morpheme"       # tier 3 (has Salish chars AND '+')
GLOSS       = "gloss"          # tier 4 (ASCII, morpheme-gloss style)
TRANSLATION = "translation"    # tier 5 (free English sentence)
FOOTNOTE    = "footnote"       # scholarly prose — discard
UNKNOWN     = "unknown"


def classify(line: str) -> LineKind:
    s = line.strip()

    if not s:
        return EMPTY

    if PAGE_DIVIDER_RE.match(s) or PAGE_HEADER_RE.match(s):
        return PAGE_MARKER

    if STANDALONE_NUM_RE.match(s):
        return SKIP

    # Text ID patterns (complete or fragment; keep short to avoid false positives)
    if len(s) <= 9 and (COMPLETE_ID_RE.match(s) or ROMAN_ONLY_RE.match(s) or DOTNUM_ONLY_RE.match(s)):
        return TEXT_ID

    salish   = has_salish(s)
    has_plus = "+" in s

    # Tier 3: Salish chars AND morpheme '+' boundary marker
    if salish and has_plus:
        # But: footnote lines can have Salish words quoted + dictionary refs
        if DICT_REF_RE.search(s):
            return FOOTNOTE
        return MORPHEME

    # Tier 1 / 2: Salish chars without '+'
    if salish:
        # Footnote: quoted Salish word followed by dictionary reference number
        if DICT_REF_RE.search(s):
            return FOOTNOTE
        return SALISH

    # --- All-ASCII from here ---
    n_abbrev   = count_gloss_abbrevs(s)
    dot_word   = has_dot_in_word(s)
    person_num = has_person_number(s)

    # Tier 4: gloss line
    if n_abbrev >= 2 or dot_word or (n_abbrev == 1 and person_num):
        return GLOSS

    # Tier 5 candidate or footnote.
    # Some translations start with lowercase (continuation sentences like
    # "we drifted along." or "you might do some foolish things...").
    # Gloss lines almost never end with sentence-terminal punctuation,
    # so the s[-1] in ".!?:" check plus n_abbrev==0 is sufficient.
    if s[-1] in ".!?:" and " " in s and len(s) >= 8:
        if CITATION_RE.search(s) or LINGUISTIC_RE.search(s) or len(s) > 280:
            return FOOTNOTE
        # Require no gloss markers for lowercase-starting lines
        if n_abbrev == 0 and not dot_word and not person_num:
            return TRANSLATION
        # Uppercase start is still allowed even with a single stray abbreviation
        if s[0].isupper() and n_abbrev < 2 and not dot_word:
            return TRANSLATION

    # Single abbreviation or short partial gloss line
    if n_abbrev == 1 or person_num:
        return GLOSS

    return UNKNOWN


# ---------------------------------------------------------------------------
# Text-ID fragment assembler
# ---------------------------------------------------------------------------

def try_assemble_id(parts: list[str]) -> Optional[str]:
    """
    Given 1-2 raw text-ID fragments, try to produce a complete 'ROMAN.NUM'
    ID string.  Handles the fragmented case where the page break splits
    '.005' from 'II' (stored in raw text as '.005' then 'II').
    """
    if not parts:
        return None

    s = "".join(parts)
    if COMPLETE_ID_RE.match(s):
        return s

    if len(parts) == 2:
        # Try reversed assembly (number-part appears before roman-part in stream)
        reversed_s = parts[1] + parts[0]
        if COMPLETE_ID_RE.match(reversed_s):
            return reversed_s

    return None


# ---------------------------------------------------------------------------
# Page-range filter
# ---------------------------------------------------------------------------

def filter_page_range(
    raw_lines: list[str],
    min_page: int,
    max_page: int,
) -> list[str]:
    """
    Return only the lines that fall within [min_page, max_page] (inclusive).
    Page markers themselves are retained so classifiers can still see them.
    Lines before the first PAGE marker are excluded.
    """
    out: list[str] = []
    current_page: Optional[int] = None
    in_range = False

    for line in raw_lines:
        m = PAGE_NUMBER_RE.match(line.strip())
        if m:
            current_page = int(m.group(1))
            in_range = min_page <= current_page <= max_page
        if current_page is not None and in_range:
            out.append(line)

    return out


# ---------------------------------------------------------------------------
# Utterance data model
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    text_id:        Optional[str] = None
    salish_lines:   list[str]     = field(default_factory=list)
    morpheme_lines: list[str]     = field(default_factory=list)
    gloss_lines:    list[str]     = field(default_factory=list)
    translation:    Optional[str] = None


# ---------------------------------------------------------------------------
# Two-pass translation-anchored parser
# ---------------------------------------------------------------------------
# Pass 1: classify every line, building a flat list of (kind, stripped) tuples
#         with assembled text IDs.
# Pass 2: for each TRANSLATION entry, scan backward through the flat list
#         (stopping at the previous TRANSLATION) to collect the salish,
#         morpheme, and gloss lines for that utterance.
#
# This avoids the block-merging problem: a missing translation no longer
# causes subsequent Salish lines to pile into the same block.

def _classify_all(page_filtered_lines: list[str]) -> list[tuple[str, str]]:
    """
    Classify every line, assembling fragmented text IDs on the fly.
    Returns a list of (kind, stripped_text) pairs including TEXT_ID entries
    that represent fully assembled IDs.
    """
    out: list[tuple[str, str]] = []
    id_parts: list[str] = []

    for line in page_filtered_lines:
        kind = classify(line)
        s    = line.strip()

        if kind == TEXT_ID:
            if COMPLETE_ID_RE.match(s):
                # Already a full ID — emit directly, discard any stale fragments
                id_parts.clear()
                out.append((TEXT_ID, s))
            else:
                # Partial fragment (Roman-only or dot-number-only)
                id_parts.append(s)
                assembled = try_assemble_id(id_parts)
                if assembled:
                    out.append((TEXT_ID, assembled))
                    id_parts.clear()
                elif len(id_parts) > 2:
                    # Stale — keep only most recent fragment
                    id_parts = id_parts[-1:]
            continue

        # Non-ID line resets id_parts accumulation (prevents cross-utterance merging)
        if id_parts and kind not in (EMPTY, PAGE_MARKER, SKIP):
            id_parts.clear()

        out.append((kind, s))

    return out


def parse_blocks(
    raw_lines: list[str],
    min_page: int = 15,
    max_page: int = 111,
) -> list[Utterance]:
    """
    Parse interlinear lines into Utterance objects using a
    translation-anchored backward window.

    For each TRANSLATION line, scan backward through classified lines
    (stopping at the previous TRANSLATION) to collect:
      - salish lines  (tier 1 / 2)
      - morpheme lines (tier 3)
      - gloss lines    (tier 4)
      - nearest text ID

    This approach is robust to missing translations: each translation
    only claims the lines in its own backward window.
    """
    page_lines  = filter_page_range(raw_lines, min_page, max_page)
    classified  = _classify_all(page_lines)

    # Find indices of all TRANSLATION lines
    trans_indices = [i for i, (k, _) in enumerate(classified) if k == TRANSLATION]

    utterances: list[Utterance] = []

    for ti in trans_indices:
        _, trans_text = classified[ti]

        salish_lines:   list[str] = []
        morpheme_lines: list[str] = []
        gloss_lines:    list[str] = []
        text_id: Optional[str]    = None

        # Scan backward from ti-1; stop at a previous TRANSLATION
        for i in range(ti - 1, -1, -1):
            kind, s = classified[i]

            if kind == TRANSLATION:
                break   # hit boundary of previous utterance

            if kind in (EMPTY, PAGE_MARKER, SKIP, FOOTNOTE, UNKNOWN):
                continue

            if kind == TEXT_ID:
                if text_id is None:
                    text_id = s   # take the nearest (most recent) text ID
                continue

            if kind == SALISH:
                salish_lines.insert(0, nfc(s))
            elif kind == MORPHEME:
                morpheme_lines.insert(0, nfc(s))
            elif kind == GLOSS:
                gloss_lines.insert(0, s)

        utterances.append(Utterance(
            text_id        = text_id,
            salish_lines   = salish_lines,
            morpheme_lines = morpheme_lines,
            gloss_lines    = gloss_lines,
            translation    = trans_text,
        ))

    return utterances


# ---------------------------------------------------------------------------
# Record construction and validation
# ---------------------------------------------------------------------------

def utterance_to_record(u: Utterance, source: str) -> Optional[dict]:
    """Convert an Utterance to a JSONL dict.  Returns None if unusable."""
    if not u.translation:
        return None

    # Prefer tier-2 (normalized, second salish line); fall back to tier-1
    if len(u.salish_lines) >= 2:
        salish = u.salish_lines[1]
    elif len(u.salish_lines) == 1:
        salish = u.salish_lines[0]
    else:
        return None  # no Salish surface recovered

    gloss = " ".join(u.gloss_lines)

    return {
        "salish":         nfc(salish),
        "english":        u.translation,
        "source":         source,
        "text_id":        u.text_id,
        "morpheme_gloss": gloss,
        "verified":       False,
    }


def validate_record(record: dict) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    rejection_reason is an empty string when valid.
    """
    salish  = record.get("salish", "")
    english = record.get("english", "")

    if not salish:
        return False, "empty_salish"

    if not english:
        return False, "empty_english"

    # Salish must contain at least one non-ASCII character
    if all(ord(c) < 128 for c in salish):
        return False, "no_salish_chars"

    # Basic length guard (512 NLLB subword tokens ~ 350 English words;
    # Salish utterances in this corpus are much shorter than any limit)
    if len(salish) > 1000 or len(english) > 2000:
        return False, "exceeds_length_limit"

    return True, ""


# ---------------------------------------------------------------------------
# Train / dev / test split
# ---------------------------------------------------------------------------

def split_records(
    records: list[dict],
    dev_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = max(1, int(n * test_ratio))
    n_dev  = max(1, int(n * dev_ratio))

    test   = shuffled[:n_test]
    dev    = shuffled[n_test : n_test + n_dev]
    train  = shuffled[n_test + n_dev :]

    return train, dev, test


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logging.info("Wrote %d records to %s", len(records), path)


def source_label(stem: str) -> str:
    """Map a file stem like 'camp_raw' to a source label like 'camp2007'."""
    return SOURCE_MAP.get(stem, stem)


def output_prefix(stem: str) -> str:
    """Map a file stem to the JSONL output prefix."""
    label = source_label(stem)
    # camp2007_raw -> camp2007, vogt1940_raw -> vogt1940
    return label.replace("_raw", "")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_file(
    raw_file: Path,
    output_dir: Path,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    min_page: int = 15,
    max_page: int = 111,
) -> tuple[int, int]:
    """
    Process one extracted text file.  Returns (n_valid, n_rejected).
    """
    stem   = raw_file.stem                 # e.g. 'camp_raw'
    src    = source_label(stem)
    prefix = output_prefix(stem)

    logging.info("Processing: %s  (source=%s)", raw_file.name, src)

    with open(raw_file, encoding="utf-8") as fh:
        raw_lines = fh.readlines()

    utterances = parse_blocks(raw_lines, min_page=min_page, max_page=max_page)
    logging.info("  Utterance blocks parsed: %d (pages %d-%d)", len(utterances), min_page, max_page)

    records:   list[dict] = []
    rejected:  list[dict] = []

    for u in utterances:
        rec = utterance_to_record(u, src)
        if rec is None:
            rejected.append({
                "reason":  "no_salish_or_translation",
                "text_id": u.text_id,
                "salish_lines":  u.salish_lines,
                "translation":   u.translation,
            })
            continue

        valid, reason = validate_record(rec)
        if valid:
            records.append(rec)
        else:
            rejected.append({"reason": reason, **rec})

    train, dev, test = split_records(records, dev_ratio, test_ratio, seed)

    write_jsonl(train, output_dir / f"{prefix}_train.jsonl")
    write_jsonl(dev,   output_dir / f"{prefix}_dev.jsonl")
    write_jsonl(test,  output_dir / f"{prefix}_test.jsonl")
    write_jsonl(rejected, output_dir / "rejection_log.jsonl")

    return len(records), len(rejected)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Parse Camp (2007) interlinear Kalispel Salish texts into JSONL training pairs.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a *_raw.txt file or a directory containing *_raw.txt files.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory for output JSONL files.",
    )
    parser.add_argument(
        "--dev-ratio",  type=float, default=0.1,
        help="Fraction of pairs to place in the dev set   (default: 0.1).",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1,
        help="Fraction of pairs to place in the test set  (default: 0.1).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/dev/test split        (default: 42).",
    )
    parser.add_argument(
        "--min-page", type=int, default=15,
        help="First PDF page to include (default: 15, start of interlinear text).",
    )
    parser.add_argument(
        "--max-page", type=int, default=111,
        help="Last PDF page to include  (default: 111, end of interlinear text).",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_dir():
        raw_files = sorted(input_path.glob("*_raw.txt"))
        if not raw_files:
            parser.error(f"No *_raw.txt files found in {input_path}")
    else:
        raw_files = [input_path]

    total_valid    = 0
    total_rejected = 0

    for raw_file in raw_files:
        n_valid, n_rejected = process_file(
            raw_file, output_path,
            args.dev_ratio, args.test_ratio, args.seed,
            min_page=args.min_page, max_page=args.max_page,
        )
        total_valid    += n_valid
        total_rejected += n_rejected

    print()
    print(f"  Files processed : {len(raw_files)}")
    print(f"  Valid pairs     : {total_valid}")
    print(f"  Rejected        : {total_rejected}")
    if total_valid + total_rejected > 0:
        pct = 100 * total_valid / (total_valid + total_rejected)
        print(f"  Acceptance rate : {pct:.1f}%")


if __name__ == "__main__":
    main()
