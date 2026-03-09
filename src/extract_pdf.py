"""
extract_pdf.py — Sovereign PDF Extraction Pipeline
Kalispel Salish NMT Project

Fixes in this version:
  1. Y_TOLERANCE raised to 6.0pt   → merges gloss small-caps into one line
  2. PUA character audit/remap      → identifies Unicode Private Use Area chars
     (the 'box' characters like ʷ that PyMuPDF resolves to wrong codepoints)
  3. Superscript detection tightened → only true footnote numbers stripped
  4. Footnote block removal          → strips bottom-of-page footnote paragraphs

OCAP compliant: 100% local, no data leaves machine.

Usage:
    # First: audit PUA characters on a sample page
    python src/extract_pdf.py --input "data/raw/Camp2007.pdf" \
        --output data/raw/camp_raw.txt --pua-audit --debug-page 15

    # Then: full extraction once PUA map is built
    python src/extract_pdf.py --input "data/raw/Camp2007.pdf" \
        --output data/raw/camp_raw.txt --start-page 15

    # Debug single page
    python src/extract_pdf.py ... --debug-page 15
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("Missing: pip install pymupdf")

# ── Tuning Constants ──────────────────────────────────────────────────────────

Y_TOLERANCE = 6.0   # raised from 4.5 → merges 2pt small-caps offset in gloss lines

# Font size ratio below which a digits-only span is a footnote superscript.
# 9.5pt small-caps glosses / 11.0pt median = 0.86 → set threshold BELOW 0.86
# so glosses are NOT stripped. True footnote numbers are 6.5-7.0pt → ratio ~0.6
SUPERSCRIPT_SIZE_RATIO = 0.72   # strips spans < 72% of median (catches 6.5/7pt, not 9.5pt)

FOOTNOTE_ANCHOR_RE = re.compile(r'^\d+[A-Za-z\("]')

# ── Unicode Private Use Area (PUA) Remap ─────────────────────────────────────
#
# Camp's thesis uses SIL fonts (Charis SIL / Doulos SIL) that map some
# Salish characters to PUA codepoints instead of standard Unicode.
# PyMuPDF resolves CID→PUA but not PUA→Unicode.
#
# Run with --pua-audit to discover which PUA codepoints appear in your PDF.
# Then add them here. Common SIL IPA font mappings:
#
# The 'box' character (ʷ) is likely one of these PUA points:
PUA_REPAIR_MAP = {
    "\uF02A": "ʷ",   # SIL IPA93: labialization superscript w (most likely culprit)
    "\uF02B": "ʷ",   # alternate SIL mapping for ʷ
    "\uF048": "ʼ",   # ejective apostrophe
    "\uF04A": "ʔ",   # glottal stop
    "\uF02C": "ɬ",   # voiceless lateral fricative
    "\uF041": "č",   # postalveolar affricate
    "\uF044": "š",   # postalveolar fricative
    "\uF078": "x̌",   # uvular fricative
    "\uF04C": "ƛ",   # lateral affricate
    "\uF064": "ə",   # schwa
    "\uF06E": "ŋ",   # velar nasal
    "\uF071": "q",   # uvular stop (if remapped)
    "\uF077": "w",   # plain w (if remapped)
    # Add more after running --pua-audit
}

# CID artifacts that PyMuPDF still passes through
CID_REPAIR_MAP = {
    "(cid:9)":  "ʷ",
    "(cid:21)": "ʷ",
    "(cid:4)":  "",
    "(cid:10)": "ʼ",
    "(cid:11)": "ʔ",
    "(cid:12)": "ɬ",
    "(cid:13)": "č",
    "(cid:14)": "š",
    "(cid:15)": "x̌",
    "(cid:16)": "ƛ",
    "(cid:17)": "ə",
}

# Control character remap
# \x15 (NAK, 0x15) is how Camp's font encodes ʷ (labialization marker).
# PyMuPDF passes raw font bytes through for unmapped glyphs.
# This is the most critical repair — ʷ appears ~100+ times in the thesis.
CONTROL_REPAIR_MAP = {
    "\x15": "ʷ",   # NAK → MODIFIER LETTER SMALL W (U+02B7)
    "\x0e": "ƛ",   # SO  → LATIN SMALL LETTER TURNED Y (lateral affricate)
    "\x0f": "ɬ",   # SI  → LATIN SMALL LETTER L WITH BELT (if needed)
    "\x14": "ʼ",   # DC4 → MODIFIER LETTER APOSTROPHE (ejective)
    "\x16": "ʕ",   # SYN → LATIN LETTER PHARYNGEAL VOICED FRICATIVE
}

CORRUPTION_SIGNALS = ["(cid:", "\x00", "\ufffd"]

# ── Font CMap Patch ───────────────────────────────────────────────────────────
#
# Camp (2007) uses an embedded TrueType font (R40 / DXCOJF+TTFF4A94D8t00) whose
# ToUnicode CMap maps only ONE character ('e').  For all other glyphs, PyMuPDF
# tries to look up the glyph name in the Adobe Glyph List (AGL) — but the font
# uses custom SIL names (wsuper, commasuprightnosp, haceknosp, etc.) that the
# AGL doesn't know, so they all collapse to U+FFFD.
#
# Fix: read the font's Encoding Differences array (which has the glyph names),
# map each name to the correct Unicode codepoint via the table below, then inject
# a complete ToUnicode CMap into the in-memory document before any text extraction.
#
# Confirmed by inspection of xref 126 (Encoding) and xref 128 (ToUnicode) in
# Camp2007.pdf.  Font R9 (xref 125) uses only standard AGL names and is fine.

CUSTOM_GLYPH_UNICODE: dict[str, str] = {
    # SIL/non-AGL glyph names found in Camp (2007) font encoding
    # → correct Unicode for Kalispel Salish orthography
    "wsuper":              "\u02B7",   # ʷ  MODIFIER LETTER SMALL W (labialization)
    "apostrophesupnosp":   "\u02BC",   # ʼ  MODIFIER LETTER APOSTROPHE (ejective)
    "commasuprightnosp":   "\u02BC",   # ʼ  same ejective marker, different glyph variant
    "haceknosp":           "\u030C",   # ̌   COMBINING CARON (used to form x̌, etc.)
    "macronnosp":          "\u0304",   # ̄   COMBINING MACRON
    "dotnosp":             "\u0307",   # ̇   COMBINING DOT ABOVE
    "dotsubnosp":          "\u0323",   # ̣   COMBINING DOT BELOW
    "halflength":          "\u02D1",   # ˑ   MODIFIER LETTER HALF TRIANGULAR COLON
    # Standard IPA names that the AGL may or may not carry:
    "glottalstop":         "\u0294",   # ʔ  LATIN LETTER GLOTTAL STOP
    "lbelt":               "\u026C",   # ɬ  LATIN SMALL LETTER L WITH BELT
    "lambdabar":           "\u019B",   # ƛ  LATIN SMALL LETTER LAMBDA WITH STROKE
    "schwa":               "\u0259",   # ə  LATIN SMALL LETTER SCHWA
    "scaron":              "\u0161",   # š  LATIN SMALL LETTER S WITH CARON
    "ccaron":              "\u010D",   # č  LATIN SMALL LETTER C WITH CARON
}


def _parse_differences(obj_text: str) -> dict[int, str]:
    """
    Parse a PDF Encoding Differences array into {char_code: glyph_name}.
    Input is the raw xref object string from doc.xref_object().
    """
    # Extract content inside brackets
    m = re.search(r'\[\s*(.*?)\s*\]', obj_text, re.DOTALL)
    if not m:
        return {}
    tokens = re.findall(r'/\w+|-?\d+', m.group(1))
    result: dict[int, str] = {}
    code = 0
    for tok in tokens:
        if tok.lstrip('-').isdigit():
            code = int(tok)
        elif tok.startswith('/'):
            result[code] = tok[1:]
            code += 1
    return result


def _make_tounicode_cmap(entries: dict[int, str]) -> bytes:
    """
    Generate a ToUnicode CMap stream from {char_code: unicode_string} mapping.
    unicode_string may be a multi-char sequence (e.g. a base + combining char).

    The codespacerange is set to exactly the codes we map so that codes outside
    this range (e.g. standard ASCII letters not in the font's Differences array)
    fall back to PyMuPDF's glyph-name → AGL resolution rather than FFFD.
    """
    if not entries:
        return b""
    min_code = min(entries)
    max_code = max(entries)
    lines = [
        "/CIDInit /ProcSet findresource begin",
        "12 dict begin",
        "begincmap",
        "/CMapType 2 def",
        "1 begincodespacerange",
        f"<{min_code:02X}><{max_code:02X}>",
        "endcodespacerange",
        f"{len(entries)} beginbfchar",
    ]
    for code in sorted(entries):
        uni = entries[code]
        hex_code = f"<{code:02X}>"
        hex_uni = "".join(f"{ord(c):04X}" for c in uni)
        lines.append(f"{hex_code} <{hex_uni}>")
    lines += ["endbfchar", "endcmap",
              "CMapName currentdict /CMap defineresource pop",
              "end end"]
    return "\n".join(lines).encode("latin-1")


def patch_pdf_fonts(doc) -> int:
    """
    Scan all fonts in the document for Encoding Differences with unresolvable
    glyph names.  Inject a complete ToUnicode CMap so PyMuPDF resolves them
    instead of outputting U+FFFD.

    Returns the number of fonts patched.
    """
    # Collect xrefs of all Font objects that have an Encoding
    patched = 0
    seen_enc_xrefs: set[int] = set()

    for pnum in range(len(doc)):
        for f in doc.get_page_fonts(pnum):
            font_xref = f[0]
            font_dict_text = doc.xref_object(font_xref)

            # Find the Encoding xref
            enc_m = re.search(r'/Encoding\s+(\d+)\s+\d+\s+R', font_dict_text)
            if not enc_m:
                continue
            enc_xref = int(enc_m.group(1))
            if enc_xref in seen_enc_xrefs:
                continue
            seen_enc_xrefs.add(enc_xref)

            enc_text = doc.xref_object(enc_xref)
            diff_map = _parse_differences(enc_text)
            if not diff_map:
                continue

            # Build char_code → Unicode for every glyph we can resolve
            # Start from WinAnsiEncoding base for codes not in Differences
            unicode_map: dict[int, str] = {}
            for code, gname in diff_map.items():
                uni = CUSTOM_GLYPH_UNICODE.get(gname)
                if uni:
                    unicode_map[code] = uni
                    continue
                # Try simple single-char names
                if len(gname) == 1:
                    unicode_map[code] = gname
                    continue
                # Known multi-char glyph name conventions (standard AGL subset)
                AGL_SIMPLE = {
                    "space": " ", "period": ".", "comma": ",", "hyphen": "-",
                    "plus": "+", "equal": "=", "colon": ":", "semicolon": ";",
                    "question": "?", "exclam": "!", "slash": "/",
                    "parenleft": "(", "parenright": ")",
                    "bracketleft": "[", "bracketright": "]",
                    "asterisk": "*", "quotesingle": "'",
                    "quoteright": "\u2019", "quoteleft": "\u2018",
                    "quotedblleft": "\u201C", "quotedblright": "\u201D",
                    "endash": "\u2013", "emdash": "\u2014",
                    "radical": "\u221A", "underscore": "_",
                    "zero": "0", "one": "1", "two": "2", "three": "3",
                    "four": "4", "five": "5", "six": "6", "seven": "7",
                    "eight": "8", "nine": "9",
                    "oacute": "\u00F3", "eacute": "\u00E9", "aacute": "\u00E1",
                    "uacute": "\u00FA", "iacute": "\u00ED",
                }
                if gname in AGL_SIMPLE:
                    unicode_map[code] = AGL_SIMPLE[gname]
                # Unknown names are left out — PyMuPDF already resolves them or
                # they'll surface in the SUSPICIOUS_CONTROLS audit

            if not unicode_map:
                continue

            cmap_bytes = _make_tounicode_cmap(unicode_map)

            # Find or create the ToUnicode xref for this font
            tou_m = re.search(r'/ToUnicode\s+(\d+)\s+\d+\s+R', font_dict_text)
            if tou_m:
                tou_xref = int(tou_m.group(1))
                doc.update_stream(tou_xref, cmap_bytes)
            else:
                # Create a new stream object and link it
                tou_xref = doc.get_new_xref()
                doc.update_object(tou_xref, "<<>>")
                doc.update_stream(tou_xref, cmap_bytes)
                new_font_dict = font_dict_text.rstrip(">").rstrip() + \
                    f"\n  /ToUnicode {tou_xref} 0 R\n>>"
                doc.update_object(font_xref, new_font_dict)

            patched += 1

    return patched


# Control chars that are NOT valid in clean Salish text
# (anything below 0x20 except newline/CR is suspicious — tab included because
# tabs in Salish word spans are always encoding artifacts, not layout whitespace)
SUSPICIOUS_CONTROLS = set(chr(i) for i in range(0x01, 0x20)
                          if i not in (0x0a, 0x0d))


# ── PUA Detection ─────────────────────────────────────────────────────────────

def is_pua(char: str) -> bool:
    """Return True if char is in Unicode Private Use Area (E000-F8FF or F0000+)."""
    cp = ord(char)
    return (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0xFFFFF)


def find_pua_chars(text: str) -> list[tuple[str, str]]:
    """Return list of (char, hex_codepoint) for each PUA character in text."""
    return [(c, f"U+{ord(c):04X}") for c in text if is_pua(c)]


def repair_text(text: str) -> tuple[str, list[str], list[str], list[str]]:
    """
    Apply control char, PUA, and CID repair maps in order.
    Returns (repaired_text, unknown_pua_chars, unknown_cid_codes, unknown_controls)
    """
    # ── Layer 1: Control character repair (catches \x15 → ʷ etc.) ──────────
    for bad, good in CONTROL_REPAIR_MAP.items():
        text = text.replace(bad, good)

    # ── Layer 2: PUA repair ──────────────────────────────────────────────────
    unknown_pua = []
    for char in list(text):
        if is_pua(char):
            if char in PUA_REPAIR_MAP:
                text = text.replace(char, PUA_REPAIR_MAP[char])
            else:
                unknown_pua.append(f"U+{ord(char):04X} ('{char}')")

    # ── Layer 3: CID repair ──────────────────────────────────────────────────
    unknown_cid = [c for c in set(re.findall(r'\(cid:\d+\)', text))
                   if c not in CID_REPAIR_MAP]
    for bad, good in CID_REPAIR_MAP.items():
        text = text.replace(bad, good)

    # ── Layer 4: Final Unicode normalization ─────────────────────────────────
    text = unicodedata.normalize('NFC', text)

    # ── Layer 5: Audit remaining suspicious control characters ───────────────
    # SUSPICIOUS_CONTROLS chars that survive all layers are unresolved encoding
    # artifacts. Collect them for audit output — do not silently drop them.
    unknown_controls = [f"\\x{ord(c):02X}" for c in text if c in SUSPICIOUS_CONTROLS]

    return text, unknown_pua, unknown_cid, unknown_controls


# ── Superscript Detection ─────────────────────────────────────────────────────

def get_page_median_fontsize(blocks: list) -> float:
    sizes = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("text", "").strip():
                    sizes.append(span.get("size", 12))
    if not sizes:
        return 12.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def is_superscript(span: dict, median_size: float, line_y: float) -> bool:
    """
    True only for genuine footnote superscript numbers.
    Criteria:
      - digits only
      - font size < SUPERSCRIPT_SIZE_RATIO * median
    Note: small-caps gloss abbreviations (9.5pt on 11pt page) are NOT stripped.
    """
    text = span.get("text", "").strip()
    if not text.isdigit():
        return False
    size = span.get("size", 12)
    return size < median_size * SUPERSCRIPT_SIZE_RATIO


# ── Core Line Reconstruction ──────────────────────────────────────────────────

def extract_page_lines(page) -> list[str]:
    """
    Reconstruct logical lines from PDF spans using bounding-box Y grouping.
    """
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    median_size = get_page_median_fontsize(blocks)

    raw_spans = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            first_y = line["spans"][0]["bbox"][1] if line["spans"] else 0
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                if is_superscript(span, median_size, first_y):
                    continue
                bbox = span["bbox"]
                y0, x0, x1 = bbox[1], bbox[0], bbox[2]
                bucket = round(y0 / Y_TOLERANCE) * Y_TOLERANCE
                raw_spans.append((bucket, x0, x1, text.strip()))

    if not raw_spans:
        return []

    y_groups: dict[float, list] = defaultdict(list)
    for bucket, x0, x1, text in raw_spans:
        y_groups[bucket].append((x0, x1, text))

    lines = []
    for bucket in sorted(y_groups.keys()):
        row = sorted(y_groups[bucket], key=lambda s: s[0])
        line_text = ""
        prev_x1 = None
        for x0, x1, text in row:
            if prev_x1 is not None and x0 - prev_x1 > 4:
                line_text += " "
            line_text += text
            prev_x1 = x1
        if line_text.strip():
            lines.append(line_text.strip())

    # Strip footnote blocks from page bottom
    clean = []
    in_footnotes = False
    for line in lines:
        if FOOTNOTE_ANCHOR_RE.match(line):
            in_footnotes = True
        if not in_footnotes:
            clean.append(line)

    return clean


# ── PUA Audit Mode ────────────────────────────────────────────────────────────

def pua_audit(pdf_path: Path, page_num: int):
    """
    Print every PUA character found on a page with its hex codepoint.
    Use this to build out PUA_REPAIR_MAP.
    """
    doc = fitz.open(str(pdf_path))
    n_patched = patch_pdf_fonts(doc)
    if n_patched:
        print(f"[font patch] {n_patched} font(s) patched with complete ToUnicode CMap")
    page = doc[page_num - 1]
    full_text = page.get_text("text")
    doc.close()

    pua_found: dict[str, list[str]] = {}
    for i, char in enumerate(full_text):
        if is_pua(char):
            cp = f"U+{ord(char):04X}"
            context = full_text[max(0, i-3):i+4].replace('\n', '↵')
            if cp not in pua_found:
                pua_found[cp] = []
            if len(pua_found[cp]) < 3:   # show up to 3 examples
                pua_found[cp].append(f"'...{context}...'")

    print(f"\n{'='*70}")
    print(f"PUA AUDIT: Page {page_num}")
    print(f"{'='*70}")

    if not pua_found:
        print("✅ No PUA characters found on this page.")
    else:
        print(f"Found {len(pua_found)} distinct PUA codepoints:\n")
        print("Add these to PUA_REPAIR_MAP in extract_pdf.py:\n")
        for cp, examples in sorted(pua_found.items()):
            char = chr(int(cp[2:], 16))
            already = "✅ mapped" if char in PUA_REPAIR_MAP else "UNKNOWN"
            print(f"  {cp}  {already}")
            for ex in examples:
                print(f"    context: {ex}")
        print()
        print("Cross-reference with Salish character inventory:")
        print("  ʷ (U+02B7)  ʔ (U+0294)  ɬ (U+026C)  ə (U+0259)")
        print("  č (U+010D)  š (U+0161)  ƛ (U+019B)  ʼ (U+02BC)")
    print()


# ── Debug Mode ────────────────────────────────────────────────────────────────

def debug_page(pdf_path: Path, page_num: int):
    doc = fitz.open(str(pdf_path))
    n_patched = patch_pdf_fonts(doc)
    if n_patched:
        print(f"[font patch] {n_patched} font(s) patched with complete ToUnicode CMap")
    page = doc[page_num - 1]
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    median = get_page_median_fontsize(blocks)

    print(f"\n{'='*70}")
    print(f"DEBUG: Page {page_num}  |  Median: {median:.1f}pt  |  Y_TOL: {Y_TOLERANCE}pt")
    print(f"{'='*70}")

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            first_y = line["spans"][0]["bbox"][1] if line["spans"] else 0
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                x0, y0, x1, _ = span["bbox"]
                size = span.get("size", 0)
                sup = "⬆ SUPER" if is_superscript(span, median, first_y) else ""
                # Show hex for PUA chars and suspicious control chars
                pua_info = " ".join(f"[{cp}]" for _, cp in find_pua_chars(text))
                ctrl_info = " ".join(
                    f"[\\x{ord(c):02X}]" for c in text if c in SUSPICIOUS_CONTROLS
                )
                annot = " ".join(filter(None, [pua_info, f"CTRL:{ctrl_info}" if ctrl_info else ""]))
                print(f"  y={y0:6.1f}  x={x0:5.1f}→{x1:5.1f}  sz={size:4.1f}"
                      f"  '{text}'  {sup} {annot}")

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTED LINES:")
    print(f"{'='*70}")
    raw_lines = extract_page_lines(page)
    for i, line in enumerate(raw_lines, 1):
        repaired, upua, ucid, uctrl = repair_text(line)
        flags = ""
        if upua:
            flags += f"  PUA: {upua}"
        if ucid:
            flags += f"  CID: {ucid}"
        if uctrl:
            flags += f"  CTRL: {sorted(set(uctrl))}"
        print(f"  {i:3d}: {repaired}{flags}")

    doc.close()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def extract_pipeline(pdf_path: Path, output_path: Path,
                     start_page: int = 1, end_page: int = None,
                     audit: bool = False):
    doc = fitz.open(str(pdf_path))
    n_patched = patch_pdf_fonts(doc)
    total = len(doc)
    end_page = end_page or total

    print(f"{pdf_path.name}  ({total} pages)")
    if n_patched:
        print(f"[font patch] {n_patched} font(s) patched with complete ToUnicode CMap")
    print(f"   Pages {start_page}–{end_page}  |  Y_TOL={Y_TOLERANCE}pt\n")

    all_pua: dict[str, int] = {}
    all_cid: dict[str, int] = {}
    all_ctrl: dict[str, int] = {}
    output_lines = []

    for pnum in range(start_page, min(end_page + 1, total + 1)):
        page = doc[pnum - 1]
        raw_lines = extract_page_lines(page)
        output_lines += [f"\n{'─'*60}", f"PAGE {pnum}", f"{'─'*60}\n"]

        for line in raw_lines:
            repaired, upua, ucid, uctrl = repair_text(line)
            for u in upua:
                all_pua[u] = all_pua.get(u, 0) + 1
            for u in ucid:
                all_cid[u] = all_cid.get(u, 0) + 1
            for u in uctrl:
                all_ctrl[u] = all_ctrl.get(u, 0) + 1
            output_lines.append(repaired)

    doc.close()

    if audit:
        if all_pua:
            print("Unknown PUA characters (add to PUA_REPAIR_MAP):")
            for cp, n in sorted(all_pua.items(), key=lambda x: -x[1]):
                print(f"   {cp}  x{n}")
        else:
            print("No unknown PUA characters.")
        if all_cid:
            print("Unknown CID codes (add to CID_REPAIR_MAP):")
            for c, n in sorted(all_cid.items(), key=lambda x: -x[1]):
                print(f"   \"{c}\": \"?\",   # x{n}")
        else:
            print("No unknown CID codes.")
        if all_ctrl:
            print("Unknown control characters (add to CONTROL_REPAIR_MAP):")
            for c, n in sorted(all_ctrl.items(), key=lambda x: -x[1]):
                print(f"   \"{c}\": \"?\",   # x{n}")
        else:
            print("No unknown control characters.")
        print()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Done -> {output_path}")
    total_issues = len(all_pua) + len(all_cid) + len(all_ctrl)
    if total_issues:
        print(f"{total_issues} unknown character type(s) remain — run --audit or --debug-page N")
    else:
        print("Zero character artifacts in output")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",        required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--start-page",   type=int, default=1)
    p.add_argument("--end-page",     type=int, default=None)
    p.add_argument("--audit",        action="store_true")
    p.add_argument("--debug-page",   type=int, default=None)
    p.add_argument("--pua-audit",    action="store_true",
                   help="Print all PUA codepoints found (use with --debug-page)")
    p.add_argument("--y-tolerance",  type=float, default=Y_TOLERANCE)
    args = p.parse_args()

    Y_TOLERANCE = args.y_tolerance
    pdf = Path(args.input)
    out = Path(args.output)

    if args.pua_audit and args.debug_page:
        pua_audit(pdf, args.debug_page)
        sys.exit(0)

    if args.debug_page:
        debug_page(pdf, args.debug_page)
        sys.exit(0)

    extract_pipeline(pdf, out,
                     start_page=args.start_page,
                     end_page=args.end_page,
                     audit=args.audit)