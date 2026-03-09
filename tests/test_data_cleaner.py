"""
tests/test_data_cleaner.py

Unit tests for data_cleaner.py line classifier and block parser.
Covers the main classification rules and known edge cases encountered
in Camp (2007) extraction output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from data_cleaner import (
    classify,
    try_assemble_id,
    filter_page_range,
    parse_blocks,
    nfc,
    TRANSLATION, SALISH, MORPHEME, GLOSS, FOOTNOTE, TEXT_ID,
    EMPTY, PAGE_MARKER, SKIP, UNKNOWN,
)


# ---------------------------------------------------------------------------
# Line classifier
# ---------------------------------------------------------------------------

class TestClassifyEmpty:
    def test_blank(self):
        assert classify("") == EMPTY

    def test_whitespace_only(self):
        assert classify("   \t  ") == EMPTY


class TestClassifyPageMarkers:
    def test_divider(self):
        assert classify("────────────────────────────────") == PAGE_MARKER

    def test_page_header(self):
        assert classify("PAGE 15") == PAGE_MARKER

    def test_page_header_with_spaces(self):
        assert classify("  PAGE 42  ") == PAGE_MARKER


class TestClassifySkip:
    def test_standalone_number(self):
        assert classify("12") == SKIP

    def test_standalone_number_page_ref(self):
        assert classify("7") == SKIP


class TestClassifyTextID:
    def test_complete_id(self):
        assert classify("II.005") == TEXT_ID

    def test_complete_id_three_digit(self):
        assert classify("VIII.010") == TEXT_ID

    def test_roman_only(self):
        assert classify("III") == TEXT_ID

    def test_dotnum_only(self):
        assert classify(".007") == TEXT_ID

    def test_not_text_id_too_long(self):
        assert classify("II.00512345") != TEXT_ID  # too long


class TestClassifySalish:
    def test_salish_surface_ejective(self):
        assert classify("číʔcəntəm tčiná sqaltəmíxʷ") == SALISH

    def test_salish_surface_labialized(self):
        assert classify("cúntəm ɬuʔ inpʼox̌ʷút") == SALISH

    def test_salish_surface_ipa_chars(self):
        assert classify("tʼkʼʷəncút u čin tʼkʼʷəncút") == SALISH

    def test_curly_quote_not_salish(self):
        # Line with only curly apostrophe (Vogt\u2019s) should NOT be Salish
        assert classify("Vogt\u2019s dictionary has this entry.") != SALISH


class TestClassifyMorpheme:
    def test_morpheme_plus_boundary(self):
        assert classify("čiʔc -nt -m t č=naqs t s+qlt+mixʷ") == MORPHEME

    def test_morpheme_with_equals(self):
        assert classify("ye l sewɬ=kʷ čɬ+niʔkʷ=étkʷ") == MORPHEME

    def test_morpheme_explicit_plus(self):
        assert classify("hoy qeʔ tkʷʔut qeʔ epɬ+šil+mín") == MORPHEME


class TestClassifyGloss:
    def test_gloss_with_abbreviations(self):
        assert classify("arrive.PL-TR-PASS OBL one.personOBL man") == GLOSS

    def test_gloss_uppercase_abbrevs(self):
        assert classify("say -TR-PASS ART 1SG.POSS- parent") == GLOSS

    def test_gloss_person_number(self):
        assert classify("NEG 1SG.INTR FUT- go") == GLOSS

    def test_gloss_dot_in_word(self):
        assert classify("then 1PL.INTRwalk 1PL.INTR have.axe") == GLOSS


class TestClassifyTranslation:
    def test_simple_translation(self):
        assert classify("A man called on us.") == TRANSLATION

    def test_translation_with_colon(self):
        assert classify("He said to my parents:") == TRANSLATION

    def test_translation_question(self):
        assert classify("What are you going to do with him?") == TRANSLATION

    def test_lowercase_translation(self):
        # Camp (2007) has lowercase-starting translations for continuations
        assert classify("we drifted along.") == TRANSLATION

    def test_lowercase_continuation(self):
        assert classify("you might do some foolish things in front of all the people.") == TRANSLATION

    def test_not_translation_gloss(self):
        # Short gloss line shouldn't be a translation
        assert classify("arrive.PL-TR-PASS OBL man") != TRANSLATION


class TestClassifyFootnote:
    def test_citation(self):
        line = "Question marks in parentheses indicate Vogt\u2019s doubts (1940a:81)."
        assert classify(line) == FOOTNOTE

    def test_linguistic_term(self):
        assert classify("The prefix l- in this form is unexplained.") == FOOTNOTE

    def test_salish_with_dict_ref(self):
        # Salish word quoted with lexicon reference number
        assert classify("p\u0259l\u02BCíp(156), so I have added the í here.") == FOOTNOTE

    def test_very_long_line(self):
        line = "A " + "very long footnote that goes on and on " * 8 + "end."
        assert classify(line) == FOOTNOTE


# ---------------------------------------------------------------------------
# Text-ID assembler
# ---------------------------------------------------------------------------

class TestTryAssembleID:
    def test_complete_in_one_part(self):
        assert try_assemble_id(["II.005"]) == "II.005"

    def test_forward_order(self):
        # "II" + ".005" → "II.005"
        assert try_assemble_id(["II", ".005"]) == "II.005"

    def test_reversed_order(self):
        # ".005" then "II" (fragmented across page break)
        assert try_assemble_id([".005", "II"]) == "II.005"

    def test_incomplete_single_part(self):
        assert try_assemble_id(["II"]) is None

    def test_incomplete_dotnum(self):
        assert try_assemble_id([".005"]) is None

    def test_empty(self):
        assert try_assemble_id([]) is None


# ---------------------------------------------------------------------------
# Page-range filter
# ---------------------------------------------------------------------------

class TestFilterPageRange:
    def test_excludes_before_min(self):
        lines = [
            "PAGE 10\n",
            "Salish text on page 10\n",
            "PAGE 15\n",
            "Salish text on page 15\n",
        ]
        result = filter_page_range(lines, min_page=15, max_page=111)
        assert "Salish text on page 10\n" not in result
        assert "Salish text on page 15\n" in result

    def test_excludes_after_max(self):
        lines = [
            "PAGE 111\n",
            "Interlinear text\n",
            "PAGE 112\n",
            "Back-matter index\n",
        ]
        result = filter_page_range(lines, min_page=15, max_page=111)
        assert "Interlinear text\n" in result
        assert "Back-matter index\n" not in result

    def test_empty_before_first_page(self):
        lines = ["Some header text\n", "PAGE 15\n", "Content\n"]
        result = filter_page_range(lines, min_page=15, max_page=111)
        assert "Some header text\n" not in result
        assert "Content\n" in result


# ---------------------------------------------------------------------------
# Integration: parse_blocks on minimal synthetic input
# ---------------------------------------------------------------------------

class TestParseBlocksIntegration:
    def _make_page_block(self, page_num, content_lines):
        """Helper to wrap content in a page marker."""
        divider = "────────────────────────────────────────────────────────────\n"
        return (
            [divider, f"PAGE {page_num}\n", divider, "\n"]
            + content_lines
        )

    def test_single_utterance(self):
        lines = self._make_page_block(15, [
            "číʔcəntəm tčiná sqaltəmíxʷ\n",      # tier 1 (salish)
            "číʔcəntəm t činá t sqaltəmíxʷ\n",    # tier 2 (salish)
            "čiʔc -nt -m t č=naqs t s+qlt+mixʷ\n",# tier 3 (morpheme)
            "arrive.PL-TR-PASS OBL one.person\n",  # tier 4 (gloss)
            "A man called on us.\n",                # tier 5 (translation)
        ])
        blocks = parse_blocks(lines, min_page=15, max_page=111)
        assert len(blocks) == 1
        b = blocks[0]
        assert nfc("číʔcəntəm t činá t sqaltəmíxʷ") in b.salish_lines
        assert b.translation == "A man called on us."

    def test_text_id_recovered(self):
        lines = self._make_page_block(15, [
            "II.005\n",
            "hóy kʷukʷuʔéc kʷémʼt\n",
            "hóy kʷukʷuʔéc kʷémʼt\n",
            "then get.dark and.then\n",
            "The night came, we paddled.\n",
        ])
        blocks = parse_blocks(lines, min_page=15, max_page=111)
        assert any(b.text_id == "II.005" for b in blocks)

    def test_fragmented_text_id(self):
        # ".005" on one page, "II" on the next
        lines = (
            self._make_page_block(19, [
                "previous salish ɬuʔ\n",
                "previous salish ɬuʔ\n",
                "previous salish ɬuʔ+m\n",
                "ART PASS\n",
                ".005\n",
            ]) +
            self._make_page_block(20, [
                "II\n",
                "hóy kʷukʷuʔéc\n",
                "hóy kʷukʷuʔéc\n",
                "then get.dark\n",
                "The night came.\n",
            ])
        )
        blocks = parse_blocks(lines, min_page=15, max_page=111)
        found = [b for b in blocks if b.text_id == "II.005"]
        assert len(found) == 1
        assert found[0].translation == "The night came."

    def test_missing_translation_window_expands(self):
        # Utterance A has no translation; utterance B has one.
        # Known limitation: the backward window expands past A's content
        # when no previous TRANSLATION acts as a boundary.  The translation-
        # anchored approach is robust when translations are present, but a
        # missing tier-5 line allows the window to absorb preceding utterances.
        # All such pairs are flagged verified=False for human review.
        lines = self._make_page_block(15, [
            "čin qaluwétəm kʷémʼt\n",        # tier 1 of A (no translation follows)
            "čin qaluwétəm kʷémʼt\n",        # tier 2 of A
            "čn qaluwét+m kʷemʼt čn\n",      # tier 3 of A
            "yé lséuɬkʷ čɬniʔkʷétkʷ\n",     # tier 1 of B
            "yé l séuɬkʷ čɬniʔkʷétkʷ\n",    # tier 2 of B
            "ye l sewɬ=kʷ čɬ+niʔkʷ=étkʷ\n",  # tier 3 of B
            "this LOC water movement.on.water\n",  # tier 4 of B
            "There was some movement in the water.\n",  # translation of B
        ])
        blocks = parse_blocks(lines, min_page=15, max_page=111)
        b = next(b for b in blocks if b.translation == "There was some movement in the water.")
        # B's Salish is present in the window
        salish_text = " ".join(b.salish_lines)
        assert "čɬniʔkʷétkʷ" in salish_text
        # A's Salish is also in the window (known limitation without previous TRANSLATION)
        assert "čin qaluwétəm" in salish_text

    def test_page_range_excludes_back_matter(self):
        lines = (
            self._make_page_block(111, [
                "ɬuʔ čin eɬ+élʼi\n",
                "ɬuʔ čin eɬ+élʼi\n",
                "try.again -1SG.TRANS\n",
                "That's the end.\n",
            ]) +
            self._make_page_block(112, [
                "Back matter stem index content\n",
                "Stems whose roots could be determined are listed in alphabetical order.\n",
            ])
        )
        blocks = parse_blocks(lines, min_page=15, max_page=111)
        translations = [b.translation for b in blocks]
        assert "That's the end." in translations
        assert "Stems whose roots could be determined are listed in alphabetical order." not in translations
