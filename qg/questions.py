"""Question generators: fill-in-the-blank, MCQ, True/False, WH-questions.

Each maker takes a spaCy Doc of ONE key sentence and returns a question dict
(or None when the sentence does not fit the pattern). The `used` set prevents
duplicate questions.
"""

import random

from .nlp import third_singular


def _clean(s):
    s = s.strip()
    while s and s[-1] in ".!?;:":
        s = s[:-1]
    return s.strip()


def _entities_by_label(docs):
    by_label = {}
    for doc in docs:
        for ent in doc.ents:
            bucket = by_label.setdefault(ent.label_, [])
            if ent.text not in bucket:
                bucket.append(ent.text)
    return by_label


# ---------------------------------------------------------------- fill-in-blank
def make_fib(sent_doc, used):
    cands = [(e.start_char, e.end_char, e.text) for e in sent_doc.ents
             if len(e.text) > 2]
    if not cands:  # fall back to key noun phrases
        cands = [(c.start_char, c.end_char, c.text)
                 for c in sent_doc.noun_chunks
                 if 3 < len(c.text) < 45 and c.root.pos_ in ("NOUN", "PROPN")]
    random.shuffle(cands)
    for s, e, t in cands:
        key = ("fib", t.lower())
        if key in used:
            continue
        used.add(key)
        q = _clean(sent_doc.text[:s] + "_____" + sent_doc.text[e:])
        return {"type": "Fill in the blank",
                "question": q,
                "answer": t}
    return None


# --------------------------------------------------------------------- MCQ
def make_mcq(sent_doc, all_ents, used):
    cands = [e for e in sent_doc.ents if len(e.text) > 2]
    random.shuffle(cands)
    for ent in cands:
        key = ("mcq", ent.text.lower())
        if key in used:
            continue
        # distractors: other entities of the SAME type from the material
        pool = [t for t in all_ents.get(ent.label_, [])
                if t.lower() != ent.text.lower()]
        if len(pool) < 3:
            continue
        used.add(key)
        options = random.sample(pool, 3) + [ent.text]
        random.shuffle(options)
        s, e = ent.start_char, ent.end_char
        q = _clean(sent_doc.text[:s] + "_____" + sent_doc.text[e:])
        return {"type": "Multiple choice",
                "question": "Choose the correct option: " + q,
                "options": options,
                "answer": ent.text}
    return None


# ---------------------------------------------------------------- True/False
def make_true(sent_doc, used):
    key = ("tf-t", sent_doc.text[:60].lower())
    if key in used:
        return None
    used.add(key)
    return {"type": "True / False",
            "question": _clean(sent_doc.text) + ".",
            "answer": "True",
            "explanation": "This statement appears in the study material."}


def make_false(sent_doc, all_ents, used):
    # create a false statement by swapping an entity with same-type entity
    cands = [e for e in sent_doc.ents if len(e.text) > 1]
    random.shuffle(cands)
    for ent in cands:
        pool = [t for t in all_ents.get(ent.label_, [])
                if t.lower() != ent.text.lower()]
        if not pool:
            continue
        key = ("tf-f", ent.text.lower())
        if key in used:
            continue
        used.add(key)
        swap = random.choice(pool)
        s, e = ent.start_char, ent.end_char
        stmt = _clean(sent_doc.text[:s] + swap + sent_doc.text[e:]) + "."
        return {"type": "True / False",
                "question": stmt,
                "answer": "False",
                "explanation": f"According to the material it is "
                               f"'{ent.text}', not '{swap}'."}
    return None


# ------------------------------------------------------------------ WH (Who)
def make_who(sent_doc, used):
    for ent in sent_doc.ents:
        if ent.label_ != "PERSON" or ent.root.dep_ != "nsubj":
            continue
        if ent.start_char != 0:
            continue  # only sentence-initial subjects -> always grammatical
        key = ("who", ent.text.lower())
        if key in used:
            continue
        used.add(key)
        q = _clean("Who" + sent_doc.text[ent.end_char:]) + "?"
        return {"type": "WH question",
                "question": q[0].upper() + q[1:],
                "answer": ent.text}
    return None


# ------------------------------------------------------------------ WH (What)
def _matches_known_person(span_text, person_names):
    """Catch person names the sentence-level NER missed (e.g. lone 'Einstein')."""
    s = span_text.strip().lower()
    for p in person_names:
        pl = p.lower()
        if s == pl or s in pl.split():
            return True
    return False


def make_what(sent_doc, used, person_names=()):
    root = next((t for t in sent_doc
                 if t.dep_ == "ROOT" and t.pos_ in ("VERB", "AUX")), None)
    if root is None:
        return None
    subj = next((t for t in root.children
                 if t.dep_ in ("nsubj", "nsubjpass")), None)
    if subj is None or root.i < subj.right_edge.i:
        return None
    span = sent_doc[subj.left_edge.i:subj.right_edge.i + 1]
    if span.root.pos_ == "PRON":
        return None  # 'He'/'It'/'This' make poor What-questions
    if any(t.ent_type_ == "PERSON" for t in span):
        return None  # persons are handled by Who-questions
    if _matches_known_person(span.text, person_names):
        return None  # e.g. 'Einstein' known as a person from elsewhere
    key = ("what", span.text.lower())
    if key in used:
        return None

    aux = next((t for t in root.children if t.dep_ in ("aux", "auxpass")), None)
    verb = root.text
    prefix = ""
    if aux is not None:
        prefix = aux.text + " "
    elif root.tag_ in ("VB", "VBP"):
        verb = third_singular(root.text)

    rest = sent_doc.text[root.idx + len(root.text):]
    # drop a duplicated aux sitting right after the verb phrase start
    q = _clean(f"What {prefix}{verb}{rest}") + "?"
    used.add(key)
    return {"type": "WH question",
            "question": q[0].upper() + q[1:],
            "answer": span.text}


# ------------------------------------------------------- WH (Where / When)
def make_where_when(sent_doc, used):
    for ent in sent_doc.ents:
        if ent.label_ in ("GPE", "LOC", "FAC"):
            wh = "Where"
        elif ent.label_ in ("DATE", "TIME"):
            wh = "When"
        else:
            continue
        if ent.root.dep_ != "pobj":  # must sit in a prepositional phrase
            continue
        prep = ent.root.head
        if prep.pos_ != "ADP":
            continue
        aux = next((t for t in sent_doc if t.dep_ == "aux"), None)
        subj = next((t for t in sent_doc
                     if t.dep_ in ("nsubj", "nsubjpass")), None)
        if aux is None or subj is None:
            continue
        if not (subj.i < aux.i < prep.i):
            continue
        key = (wh.lower(), ent.text.lower())
        if key in used:
            continue
        used.add(key)

        subj_span = sent_doc[subj.left_edge.i:subj.right_edge.i + 1]
        subj_text = subj_span.text
        first = subj_span[0]
        if first.pos_ not in ("PROPN", "PRON") and not first.ent_type_:
            subj_text = subj_text[0].lower() + subj_text[1:]
        middle = " ".join(t.text for t in sent_doc
                          if subj.right_edge.i < t.i < prep.i
                          and t.i != aux.i).strip()
        q = f"{wh} {aux.text.lower()} {subj_text}"
        if middle:
            q += f" {middle}"
        return {"type": "WH question",
                "question": q.strip() + "?",
                "answer": ent.text}
    return None
