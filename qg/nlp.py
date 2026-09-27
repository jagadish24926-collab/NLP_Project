"""spaCy model loading and small linguistic helpers (shared by the pipeline)."""

import spacy

_nlp = None


def get_nlp():
    """Load en_core_web_sm once; fail with a clear message if missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _nlp


def split_sentences(text):
    """Segment text into clean sentences (min length filter)."""
    doc = get_nlp()(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 25]


IRREGULAR_VERBS = {
    "have": "has",
    "be": "is",
    "do": "does",
    "go": "goes",
}


def third_singular(verb):
    """Naive base-form verb -> 3rd person singular (for What-questions)."""
    v = verb.lower()
    if v in IRREGULAR_VERBS:
        return IRREGULAR_VERBS[v]
    if len(v) > 1 and v.endswith("y") and v[-2] not in "aeiou":
        return v[:-1] + "ies"
    if v.endswith(("s", "x", "z", "ch", "sh", "o")):
        return v + "es"
    return v + "s"
