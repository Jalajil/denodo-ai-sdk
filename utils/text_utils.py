"""
 Copyright (c) 2025. DENODO Technologies.
 http://www.denodo.com
 All rights reserved.

 This software is the confidential and proprietary information of DENODO
 Technologies ("Confidential Information"). You shall not disclose such
 Confidential Information and shall use it only in accordance with the terms
 of the license agreement you entered into with DENODO.
"""

"""Dependency-free text normalization for the BM25 lexical index.

Kept free of third-party imports so it can be unit-tested without the full
runtime, and so it can be imported from anywhere without cycles.
"""

import re
import unicodedata

# Bidirectional control characters (LRM/RLM, the embedding/override marks and BOM).
# Arabic-locale models and Arabic-authored metadata emit these routinely. They are
# zero-width, so an RLM sitting inside "<vql>" or "</vql>" makes the tag regex below
# fail with nothing visible in the logs to explain it.
BIDI_CONTROL_PATTERN = re.compile('[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]')

def strip_bidi_controls(text):
    """Remove zero-width bidi/formatting marks so tag parsing is script-independent."""
    if not text:
        return text

    return BIDI_CONTROL_PATTERN.sub('', text)

# Arabic letters that are routinely interchanged between how a value is stored and
# how a user types it. Folding them is what makes a lexical index match at all:
# 'جدة' and 'جده' are the same city, and BM25 sees two unrelated tokens.
_ARABIC_FOLD = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
    'ى': 'ي', 'ئ': 'ي',
    'ة': 'ه',
    'ؤ': 'و',
}
_TATWEEL = 'ـ'
_ARABIC_INDIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')


def normalize_for_lexical_index(text):
    r"""Fold a string into the form a lexical (BM25) index should store and query.

    Applied symmetrically to documents at index time and to questions at query
    time; normalizing only one side makes matching strictly worse than not
    normalizing at all.

    The work is mostly script-independent - NFKC, Arabic-Indic to Western digits,
    strip combining marks, collapse whitespace - and only the fold table above is
    Arabic-specific.

    This is not cosmetic. fastembed's BM25 tokenizer is `re.sub(r"[^\w]", " ",
    text.lower())`, and Arabic combining marks and bidi marks are not `\w` in
    Python, so they act as word separators: 'مُحَمَّد' tokenizes to four single
    letters and can never match its unvocalized spelling. The snowball Arabic
    stemmer handles diacritics correctly, but only if the word reaches it intact.
    """
    if text is None:
        return ''

    text = unicodedata.normalize('NFKC', str(text))
    text = strip_bidi_controls(text)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = text.replace(_TATWEEL, '')
    text = ''.join(_ARABIC_FOLD.get(character, character) for character in text)
    # Drop the combining marks (tashkeel) that survived NFKC.
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', text).strip()

