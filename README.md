# PlagiarismAI: Semantic Originality & Academic Writing Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![NLP & LLM Powered](https://img.shields.io/badge/NLP-SentenceTransformers%20%7C%20spaCy%20%7C%20Gemini-green.svg)](#course-outcomes-co-rubric-mapping)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-67%20Passed-brightgreen.svg)](#running-unit-tests)

**PlagiarismAI** is a comprehensive Natural Language Processing (NLP) and Generative AI application designed to detect text reuse, classify writing tone, extract topical themes and named entities, support cross-lingual analysis (English, Hindi, Marathi), and provide interactive AI-assisted rephrasing and citation guidance for academic writing.

The system is built to satisfy college project demonstration criteria (30 Marks Total) across four distinct Course Outcomes (**CO1 to CO4**).

---

## Course Outcomes (CO) Rubric Mapping

| Rubric Item | Marks | Modules & Source Files | Techniques & Technologies | Application Screen |
|---|:---:|---|---|---|
| **CO1: Text Preprocessing & Vectorization** | **7** | `nlp/preprocess.py`<br>`nlp/extract.py` | • Multi-format extraction (PDF, DOCX, TXT)<br>• Text cleaning (URLs, citations, punctuation, whitespace)<br>• Sentence splitting, tokenization, lemmatization (spaCy/WordNet)<br>• Vectorization: TF-IDF & Multilingual Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)<br>• Weighted composite originality score | **Tab 1: Check**<br>**Tab 4: Project Report** ("How we read your text" before/after preview) |
| **CO2: Sentiment Analysis & Information Extraction** | **8** | `nlp/analysis.py`<br>`evaluate.py`<br>`data/eval/` | • Document & paragraph sentiment scoring (VADER + Devanagari lexicon)<br>• Information Extraction: Named Entity Recognition (spaCy NER), KeyBERT keywords, regex academic citations<br>• Real labeled evaluation benchmark (Accuracy, Precision, Recall, F1, Confusion Matrix) | **Tab 2: Results** (Tone & Key Details cards)<br>**Tab 4: Project Report** (Benchmark table & chart) |
| **CO3: Topic Modeling & Multilingual NLP** | **7** | `nlp/topics.py`<br>`nlp/multilingual.py` | • Latent Dirichlet Allocation (scikit-learn LDA) with short-text keyword fallback<br>• Topic overlap score (cosine on topic distributions)<br>• Language detection (English, Hindi, Marathi)<br>• Multilingual translation (fallback translator; Google Cloud REST API supported via `GOOGLE_TRANSLATE_API_KEY`)<br>• Cross-lingual semantic embedding alignment | **Tab 2: Results** (Theme chips & Language badge)<br>**Tab 4: Project Report** (Multilingual benchmark) |
| **CO4: Generative AI Assistant & Prompt Engineering** | **8** | `llm/prompts.py`<br>`llm/assistant.py`<br>`docs/prompt_design.md` | • Academic Writing Coach powered by Google GenAI SDK (`google-genai`), defaulting to `gemini-3.5-flash` (configurable via `GEMINI_MODEL`)<br>• Prompt engineering: persona specification, few-shot examples, JSON schema enforcement<br>• Safety: prompt-injection isolation delimiters (`<<<PASSAGE>>>`)<br>• Side-by-side rewriting, plain explanation, APA citation, 3 originality tips | **Tab 3: Fix it for me** (Side-by-side rewrites, citations, tips) |

---

## Architecture & Project Structure

```
PlagiarismAI/
├── app.py                      # Main Streamlit application (4 clean tabs)
├── run.py                      # Master CLI runner (app, eval, tests)
├── evaluate.py                 # Real benchmark runner across CO1, CO2, CO3
├── requirements.txt            # Python dependencies
├── .env.example                # Template for API keys
│
├── nlp/                        # Core NLP pipeline modules
│   ├── extract.py              # Multi-format document text extraction (PDF, DOCX, TXT)
│   ├── preprocess.py           # CO1: Cleaning, lemmatization, TF-IDF, composite scoring
│   ├── similarity.py           # Exact, near-exact difflib, Sentence-BERT semantic matching
│   ├── analysis.py             # CO2: VADER sentiment, spaCy NER, KeyBERT, citations
│   ├── topics.py               # CO3: scikit-learn LDA topic modeling & topic overlap
│   └── multilingual.py         # CO3: Devanagari language detection, Cloud translation, cross-lingual
│
├── llm/                        # Generative AI & Prompt Engineering
│   ├── prompts.py              # Few-shot prompt templates & safety delimiters
│   └── assistant.py            # Gemini API client, JSON parser, offline fallback
│
├── data/
│   ├── samples/                # 1-Click demo sample documents
│   │   ├── sample_original.txt               # Authentic essay (Microgrids)
│   │   ├── sample_copied.txt                 # Flagged essay (Deep learning / Transformers)
│   │   ├── sample_hindi.txt                  # Devanagari text for multilingual check
│   │   └── reference_deep_learning_papers.txt # Reference corpus text
│   └── eval/                   # Real labeled benchmark evaluation datasets
│       ├── plagiarism_pairs.json             # 31 labeled text pairs (copied vs not copied)
│       ├── sentiment_sentences.json          # 20 labeled sentiment sentences
│       └── multilingual_pairs.json           # 10 cross-lingual pairs (Hindi/Marathi to English)
│
├── reports/                    # Benchmark metrics & figures
│   ├── evaluation.json         # Automated benchmark evaluation results
│   ├── figures/                # Similarity methods comparison chart
│   └── translation_cache.json  # Local persistent translation cache
│
├── docs/                       # Technical documentation
│   └── prompt_design.md        # Prompt engineering architecture & decisions
│
└── tests/                      # Automated unit test suite (66 tests)
    ├── test_ui_smoke.py        # Streamlit AppTest smoke tests across all 4 UI tabs
    ├── test_preprocess.py      # Tests for text cleaning and vectorization
    ├── test_analysis.py        # Tests for sentiment, NER, keywords, citations
    ├── test_topics_multilingual.py # Tests for LDA, language detection, translation
    ├── test_llm.py             # Tests for Gemini assistant, prompts, safety
    ├── test_semantic_evidence.py   # Regression tests for semantic evidence gating
    └── test_semantic_reporting.py  # Tests for statistics, counting, and PDF reporting
```

---

## Quick Setup & Execution

### 1. Prerequisites
- Python 3.10 to 3.14 on Windows, macOS, or Linux.
- Recommended virtual environment:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Add your API keys to `.env` if available:
- `GEMINI_API_KEY`: For Generative AI writing assistant features via the modern `google-genai` SDK (`gemini-3.5-flash` by default; configurable via `GEMINI_MODEL`). *(The app includes offline academic fallback templates if no key is supplied).*
- `GOOGLE_TRANSLATE_API_KEY`: Google Cloud Translation is supported by adding `GOOGLE_TRANSLATE_API_KEY` to `.env`; the free fallback translator is used otherwise (deep-translator or local offline engine). When running without a key, the language badge, reports, and demonstration script report "fallback translator" and never claim Google Cloud Translation.
- `BRAVE_API_KEY` or `TAVILY_API_KEY`: For live internet search. *(The app defaults to the verified offline reference library for zero-failure demonstrations).*

### 4. Run the Application
You can start the app using the master runner:
```powershell
python run.py
```
*(Or directly via `streamlit run app.py`)*

Then open your browser at `http://localhost:8501`.

---

## Command Line Tools

PlagiarismAI provides convenient CLI options via `run.py`:

```powershell
# 1. Run the interactive Streamlit UI
python run.py

# 2. Run the evaluation benchmark (recomputes evaluation.json and benchmark figure)
python run.py --evaluate

# 3. Run the full automated unit test suite (67 tests)
python run.py --test
```

---

## 3-Minute Live Demonstration Script (Viva / Presentation)

| Step | Time | Tab / Screen | Action to Perform | What to Explain to Examiners | Rubric Covered |
|:---:|:---:|---|---|---|:---:|
| **1** | 0:00 - 0:40 | **Tab 1: Check** | Click the **"⚠️ Copied & Paraphrased"** sample button, then click **"Check my document"**. | Explain the multi-stage pipeline: text extraction, cleaning, lemmatization, language identification, and semantic similarity matching against literature. | **CO1** |
| **2** | 0:40 - 1:20 | **Tab 2: Results** | Review the **Originality Score (14.3%)** (85.7% similarity) and the 4 plain-language cards: **Copied parts** (6 of 7 sentences flagged), **Tone (Positive, 83% confidence)**, **Themes (Artificial Intelligence & Machine Learning)**, and **Key details (Vaswani et al.)**. | Point out that semantic similarity detects paraphrases without shared exact words. Show how the 4 cards translate complex NLP into actionable insights. | **CO2** & **CO3** |
| **3** | 1:20 - 1:50 | **Tab 2 ➔ Tab 3** | Under "Copied parts", click **"Fix Passage 1"**, then switch to **Tab 3: Fix it for me** and click **"Fix this passage with AI Writing Coach"**. | Show the side-by-side comparison: flagged sentence vs academic rewrite, the plain explanation of why it was flagged, the ready-to-use APA citation, and the 3 tips. | **CO4** |
| **4** | 1:50 - 2:30 | **Tab 1 ➔ Tab 2** | Return to Tab 1, click **"🌐 Hindi Document"**, and click **"Check my document"**. | Demonstrate multilingual NLP: automatic Devanagari detection (Hindi, 85%+), translation to English using the fallback translator (or Google Cloud Translation if `GOOGLE_TRANSLATE_API_KEY` is configured in `.env`), and cross-lingual semantic matching (100% originality). | **CO3** |
| **5** | 2:30 - 3:00 | **Tab 4: Project Report** | Switch to **Tab 4: Project Report**. Show the **"How we read your text"** before/after preview, the **Evaluation Benchmark Table**, the **Benchmark Chart**, and download the official **PDF report**. | Highlight honest scientific methodology: 67 unit tests passing, evaluation across 31 text pairs, 20 sentiment sentences, and 10 cross-lingual pairs (with threshold disclosure note). | **All (CO1-CO4)** |

---

## Benchmark Results (From `reports/evaluation.json`)

> **Evaluation Methodology Note**: Similarity thresholds were chosen by inspection on a small dataset; results are optimistic.

### Plagiarism Detection Methods Benchmark (31 Labeled Pairs)
| Method | Similarity Threshold | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Jaccard Word Similarity** | 0.25 | 77.4% | 100.0% | 58.8% | 0.741 |
| **TF-IDF Cosine Similarity** | 0.35 | 77.4% | 100.0% | 58.8% | 0.741 |
| **Sentence Embeddings Cosine (S-BERT)** | 0.65 | **100.0%** | **100.0%** | **100.0%** | **1.000** |

*Finding: Pure token-overlap methods (Jaccard and TF-IDF) fail on structural paraphrasing where vocabulary is rephrased. Dense sentence embeddings capture conceptual meaning across lexical transformations.*

### Multilingual Cross-Lingual Benchmark (10 Pairs)
| Method | Similarity Threshold | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|:---:|
| **Direct Multilingual Embeddings** (`MiniLM-L12`) | 0.60 | **100.0%** | **100.0%** | **100.0%** | **1.000** |
| **Translate-Then-Compare Pipeline** (fallback translator) | 0.60 | **100.0%** | **100.0%** | **100.0%** | **1.000** |

*Note: Google Cloud Translation is supported by adding `GOOGLE_TRANSLATE_API_KEY` to `.env`; without a key, the free fallback translator is used and reported.*

### Sentiment Analysis Benchmark (20 Labeled Sentences)
- **Overall Accuracy**: **85.0%**
- **Evaluation Details**: Tested on mixed academic, critical, and Devanagari sentences.

---

## Git Remote & Repository Synchronization

The current active remote is configured to:
`https://github.com/snehalpagare272005-jpg/plagiarism_detector_AI-.git`

To push to the alternate repository name (`https://github.com/snehalpagare272005-jpg/PlagiarismAI.git`):
```powershell
# Option A: Add as a secondary remote
git remote add alternate https://github.com/snehalpagare272005-jpg/PlagiarismAI.git
git push -u alternate main

# Option B: Switch the primary origin URL
git remote set-url origin https://github.com/snehalpagare272005-jpg/PlagiarismAI.git
git push -u origin main
```

---

## Robustness & Safe Fallbacks

The application is engineered to never crash during live presentations or examinations:
- **No Internet / Offline Mode**: The app seamlessly defaults to the local verified academic reference corpus.
- **Missing Gemini API Key**: The academic writing coach gracefully falls back to structured template-based rewrites and citations.
- **Missing Google Cloud Key**: Google Cloud Translation is supported by adding `GOOGLE_TRANSLATE_API_KEY` to `.env`, and the free fallback translator is used otherwise. The language badge, reports, and demonstration script report 'fallback translator' rather than claiming Google Cloud Translation.
- **Missing spaCy Model**: Named Entity Recognition falls back to rule-based pattern extractors.

---

## License & Academic Disclaimer
PlagiarismAI is an academic demonstration system. Similarity scores measure semantic text overlap and meaning alignment; they do not constitute legal or disciplinary proof of plagiarism. Every flagged match should be reviewed manually by an instructor or researcher.
