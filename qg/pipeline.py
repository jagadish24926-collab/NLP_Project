"""End-to-end pipeline: raw text -> key sentences -> questions + answer key."""

import random

from .nlp import get_nlp, split_sentences
from .questions import (
    _entities_by_label,
    make_false,
    make_fib,
    make_mcq,
    make_true,
    make_what,
    make_where_when,
    make_who,
)


def generate_questions(text, types=("mcq", "fib", "tf", "wh"), count=10):
    """Generate up to `count` questions of the requested types from text."""
    get_nlp()  # fail fast with a clear message if the spaCy model is missing

    sents = split_sentences(text)
    if len(sents) < 3:
        raise ValueError("Not enough readable sentences found in the material.")

    from .ranker import select_key_sentences
    keys = select_key_sentences(sents, top_n=min(20, len(sents)))

    nlp = get_nlp()
    docs = [nlp(s) for s in keys]
    all_ents = _entities_by_label(docs)
    person_names = all_ents.get("PERSON", [])

    makers = []
    if "fib" in types:
        makers.append(lambda d: make_fib(d, used))
    if "mcq" in types:
        makers.append(lambda d: make_mcq(d, all_ents, used))
    if "tf" in types:
        makers.append(lambda d: make_true(d, used))
        makers.append(lambda d: make_false(d, all_ents, used))
    if "wh" in types:
        makers.append(lambda d: make_who(d, used))
        makers.append(lambda d: make_what(d, used, person_names))
        makers.append(lambda d: make_where_when(d, used))
    if not makers:
        raise ValueError("No question types selected.")

    used, out = set(), []
    order = list(range(len(docs)))
    random.shuffle(order)

    rounds = 0
    while len(out) < count and rounds < 8:
        progressed = False
        for di in order:
            if len(out) >= count:
                break
            random.shuffle(makers)
            for m in makers:
                q = m(docs[di])
                if q:
                    out.append(q)
                    progressed = True
                    break
        if not progressed:
            break
        rounds += 1
    return out
