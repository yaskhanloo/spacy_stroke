stroke_nlp_project/
│
├── data/
│   ├── raw_reports/                 # Mock or real stroke reports (txt or csv)
│
├── extractor/
│   ├── __init__.py
│   ├── preprocessing.py            # Lowercasing, cleaning
│   ├── keyword_rules.py            # Regex and pattern matchers
│   ├── spacy_ner_wrapper.py        # POS/NER tagging (spaCy)
│
├── output/
│   └── extracted_keywords.csv      # Final structured output
│
├── notebooks/
│   └── 01_exploration.ipynb        # EDA & prototyping
│
└── main.py                         # CLI or app entrypoint
