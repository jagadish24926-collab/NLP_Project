# QuestGen AI — NLP-Based Automatic Question Generation from Study Materials

Upload study material (PDF / DOCX / TXT) or paste text — the app uses NLP to
generate **multiple-choice**, **fill-in-the-blank**, **true/false** and
**WH questions**, complete with an answer key.

## Project structure

```
QG/
├── qg/
│   ├── __init__.py
│   ├── extractor.py     # PDF / DOCX / TXT -> raw text
│   ├── nlp.py           # spaCy loading + linguistic helpers
│   ├── ranker.py        # TF-IDF key-sentence selection
│   ├── questions.py     # the four question generators
│   └── pipeline.py      # end-to-end orchestration
├── templates/
│   └── index.html       # QuestGen AI frontend
├── app.py               # Flask backend
├── sample_material.txt  # demo study material (Einstein biography)
├── requirements.txt
└── README.md
```

## Setup (VS Code)

1. Open the `QG` folder in VS Code: **File → Open Folder**.
2. Open a terminal (`` Ctrl+` ``) and install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**, paste text (or click **Try Sample Material**),
choose question types and count, then **Generate Questions**. Toggle answers,
or **Download (.txt)** to save the question paper with its answer key.

## How it works

1. **Extract** — PDF/DOCX/TXT is converted to clean raw text.
2. **Segment** — spaCy splits the text into sentences.
3. **Rank** — TF-IDF scores every sentence; the top non-redundant ones
   (cosine-similarity diversity filter) become question sources.
4. **Generate** —
   - *Fill in the blank*: a named entity / key noun phrase is masked.
   - *Multiple choice*: the blank + 3 distractors (same entity type from the
     document, e.g. other dates, other people).
   - *True / False*: true statements plus false ones made by swapping an
     entity with another of the same type.
   - *WH questions*: Who/What/Where/When built from NER labels and
     dependency-parse patterns (subjects, auxiliaries, prepositional phrases).

## Notes for your demo / viva

- Everything runs **locally and offline** once the spaCy model is downloaded —
  no API keys, no GPU needed.
- Each design choice is explainable: TF-IDF ranking, NER-driven blanks,
  syntax-based WH transforms. Good viva material.
- Question quality depends on the input: factual, well-structured notes give
  the best questions.

## Troubleshooting

- `spaCy model 'en_core_web_sm' is not installed` → run
  `python -m spacy download en_core_web_sm`.
- Few or no questions → use longer material with clear facts, names and dates.
- PDF with scanned images (no selectable text) can't be read — use a
  text-based PDF or paste the text instead.
