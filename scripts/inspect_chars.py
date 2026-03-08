"""
inspect_chars.py — Identify mystery characters in extracted PDF text

Prints the exact Unicode codepoint, category, and name of every
non-ASCII character on a given page. This tells us exactly what
the 'box' character (ʷ) actually is so we can remap it.

Usage:
    python scripts/inspect_chars.py \
        --input "data/raw/Camp2007.pdf" \
        --page 15
"""

import argparse
import unicodedata
from pathlib import Path
from collections import Counter

try:
    import fitz
except ImportError:
    import sys; sys.exit("pip install pymupdf")


def inspect(pdf_path: Path, page_num: int):
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]

    # Get raw text — no processing, no repair
    raw_text = page.get_text("text")
    doc.close()

    # ── Find every unique non-ASCII character ────────────────────────────────
    char_counts = Counter(c for c in raw_text if ord(c) > 127)

    print(f"\n{'='*75}")
    print(f"CHARACTER INVENTORY: Page {page_num}")
    print(f"{'='*75}")
    print(f"{'CHAR':<6} {'CODEPOINT':<12} {'COUNT':<7} {'CATEGORY':<5} NAME")
    print(f"{'─'*75}")

    # Group into: known-good Salish chars, suspicious chars, whitespace
    known_salish = set("čšžƛəɬʔʼʷáéíóúàèìòùâêîôûāēīōūąęłńśźżčšžɾɹɲñŋýϕ"
                       "ʕɣɦɸβθðʃʒɾɹʎʟɱɳɴŋɰʔʼʷʲɥɪʏʊɔæɑɒœɛɜɞʌɐɘɵɤɶɷ")

    suspicious = []
    good = []
    for char, count in sorted(char_counts.items(), key=lambda x: ord(x[0])):
        cp = f"U+{ord(char):04X}"
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "<no name>"
        cat = unicodedata.category(char)

        # Flag anything that looks wrong
        is_combining = cat.startswith('M')
        is_control   = cat.startswith('C')
        is_private   = 0xE000 <= ord(char) <= 0xF8FF
        is_replace   = char == '\ufffd'

        row = (char, cp, count, cat, name,
               is_combining, is_control, is_private, is_replace)

        if is_private or is_replace or is_control:
            suspicious.append(row)
        else:
            good.append(row)

    # Print suspicious first (these are the problem characters)
    if suspicious:
        print("\nSUSPICIOUS CHARACTERS (likely remapping needed):")
        for char, cp, count, cat, name, *_ in suspicious:
            print(f"  {repr(char):<6} {cp:<12} ×{count:<6} {cat:<5} {name}")
    else:
        print("\nNo PUA/control/replacement characters found")

    # Print all non-ASCII chars
    print(f"\n📋 ALL NON-ASCII CHARACTERS:")
    for char, cp, count, cat, name, *_ in good:
        marker = "  "
        # Flag chars that look like boxes (no Unicode name = suspicious)
        if name == "<no name>":
            marker = "❓"
        elif cat.startswith('M'):
            marker = "◌ "   # combining mark — normal for Salish diacritics
        print(f"{marker}{repr(char):<6} {cp:<12} ×{count:<6} {cat:<5} {name}")

    # ── Show context for any character missing a name ─────────────────────
    unnamed = [c for c in char_counts if unicodedata.category(c) == 'Cn']
    if unnamed:
        print(f"\nUNASSIGNED CODEPOINTS (no Unicode name) — showing context:")
        for target in unnamed[:5]:
            cp = f"U+{ord(target):04X}"
            idx = raw_text.find(target)
            if idx >= 0:
                ctx = raw_text[max(0,idx-5):idx+6].replace('\n','↵')
                print(f"   {cp}  context: '{ctx}'")

    # ── Show the exact bytes of a sample Salish word ──────────────────────
    print(f"\n🔬 BYTE-LEVEL INSPECTION of first line containing 'sqaltəmíx':")
    for line in raw_text.split('\n'):
        if 'sqaltəm' in line or 'sqalt' in line.lower():
            print(f"   Text:  {line.strip()!r}")
            print(f"   Bytes: ", end="")
            for c in line.strip():
                if ord(c) > 127:
                    print(f"[U+{ord(c):04X}]", end="")
                else:
                    print(c, end="")
            print()
            break

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--page",  type=int, default=15)
    args = p.parse_args()
    inspect(Path(args.input), args.page)