"""Pick the most informative sentences with TF-IDF scoring + diversity filter."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def select_key_sentences(sentences, top_n=20):
    """Return up to top_n high-value, non-redundant sentences."""
    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(sentences)
    scores = np.asarray(X.sum(axis=1)).ravel()

    picked, picked_idx = [], []
    for i in np.argsort(-scores):  # highest TF-IDF first
        if len(picked) >= top_n:
            break
        if picked_idx:
            sims = cosine_similarity(X[i], X[picked_idx]).ravel()
            if sims.max() > 0.85:  # too similar to an already-picked sentence
                continue
        picked.append(sentences[i])
        picked_idx.append(i)
    return picked
