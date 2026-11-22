# PlagiarismAI: Semantic Originality & Academic Writing Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![NLP & LLM Powered](https://img.shields.io/badge/NLP-SentenceTransformers%20%7C%20spaCy%20%7C%20Gemini-green.svg)](#concepts-to-files-mapping)
[![Tests Passing](https://img.shields.io/badge/Unit%20Tests-78%20Passed-brightgreen.svg)](#running-unit-tests)

**PlagiarismAI** is a comprehensive Natural Language Processing (NLP) and Generative AI application designed to detect text reuse, classify writing tone, extract topical themes and named entities, support cross-lingual analysis (English, Hindi, Marathi), and provide interactive AI-assisted rephrasing and citation guidance for academic writing.

---

## Concepts to Files Mapping

| Concept Group | Core Concepts Implemented | Primary Modules & Files | Technologies & Libraries | Application Tab |
|---|---|---|---|---|
| **Text Preparation** | Multi-format extraction (PDF, DOCX, TXT), text cleaning, unicode normalization, sentence segmentation, tokenization, stopword removal, lemmatization | `nlp/extract.py`<br>`nlp/preprocess.py` | `pypdf`, `python-docx`, `nltk` WordNet, regex | **Tab 1: Check**<br>**Tab 4: About the Project** |
| **Turning Text into Numbers** | TF-IDF vectorization, dense sentence embeddings | `nlp/preprocess.py`<br>`nlp/similarity.py` | `scikit-learn`, `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) | **Tab 4: About the Project** |
| **Finding Copied Text** | Exact verbatim containment, SequenceMatcher near-copy, cosine similarity, Sentence-BERT semantic similarity, combined originality score, web search retrieval, web page extraction, source ranking | `nlp/similarity.py`<br>`nlp/preprocess.py`<br>`app.py` | Tavily Search/Extract, Wikipedia REST API, `beautifulsoup4`, `scikit-learn` | **Tab 1: Check**<br>**Tab 2: Results**<br>**Tab 4: About the Project** |
| **Understanding the Writing** | Sentiment & tone analysis, Named Entity Recognition (NER), KeyBERT keyword extraction, regex citation extraction, Latent Dirichlet Allocation (LDA) topic modeling, topic overlap | `nlp/analysis.py`<br>`nlp/topics.py` | `vaderSentiment`, `spacy` (`en_core_web_sm`), `keybert`, `scikit-learn` | **Tab 2: Results**<br>**Tab 4: About the Project** |
| **Multilingual NLP** | Devanagari script detection, Hindi/Marathi marker disambiguation, machine translation cascade, cross-lingual matching | `nlp/multilingual.py`<br>`nlp/similarity.py` | `langdetect`, Google Cloud Translation API / fallback translator engine | **Tab 2: Results**<br>**Tab 4: About the Project** |
| **Generative AI Assistant** | Large language model integration, prompt engineering (role instruction, few-shot examples, JSON schema, low temperature 0.3), prompt-injection delimiters, response caching and offline fallback | `llm/prompts.py`<br>`llm/assistant.py`<br>`docs/prompt_design.md` | Google Gen AI SDK (`google-genai`), `gemini-3.5-flash` | **Tab 3: Fix it for me**<br>**Tab 4: About the Project** |
| **Evaluation & Benchmarks** | Accuracy, precision, recall, F1 score, confusion matrix, labeled benchmark datasets | `evaluate.py`<br>`data/eval/`<br>`reports/evaluation.json` | `scikit-learn`, `pandas`, `matplotlib` | **Tab 4: About the Project** |

---

## Architecture & Project Structure

```
PlagiarismAI/
├── app.py                      # Main Streamlit application (4 clean tabs)
├── run.py                      # Master CLI runner (app, eval, tests)
├── evaluate.py                 # Benchmark runner for plagiarism, sentiment, and multilingual
├── requirements.txt            # Python dependencies
├── .env.example                # Template for API keys
│
├── nlp/                        # Core NLP pipeline modules
│   ├── extract.py              # Multi-format document text extraction (PDF, DOCX, TXT)
│   ├── preprocess.py           # Cleaning, lemmatization, TF-IDF, composite scoring
│   ├── similarity.py           # Exact, near-exact difflib, Sentence-BERT semantic matching
│   ├── analysis.py             # VADER sentiment, spaCy NER, KeyBERT, citations
│   ├── topics.py               # scikit-learn LDA topic modeling & topic overlap
│   └── multilingual.py         # Devanagari language detection, Cloud translation, cross-lingual
│
├── llm/                        # Generative AI & Prompt Engineering
│   ├── prompts.py              # Few-shot prompt templates & safety delimiters
│   └── assistant.py            # Gemini API client, JSON parser, offline fallback
│
├── data/
│   ├── samples/                # Sample test documents for verification
│   │   ├── sample_original.txt               # Authentic essay (Microgrids)
│   │   ├── sample_copied.txt                 # Flagged essay (Deep learning / Transformers)
│   │   ├── sample_hindi.txt                  # Devanagari text for multilingual check
│   │   └── reference_deep_learning_papers.txt # Reference corpus text
│   └── eval/                   # Real labeled benchmark evaluation datasets
│       ├── plagiarism_pairs.json             # 31 labeled text pairs (copied vs not copied)
│       ├── sentiment_sentences.json          # 20 labeled sentiment sentences
│       ├── multilingual_pairs.json           # 10 cross-lingual pairs (Hindi/Marathi to English)
│       └── extraction_samples.json           # 22 labeled sentences for NER and citation evaluation
│
├── reports/                    # Benchmark metrics & figures
│   ├── evaluation.json         # Automated benchmark evaluation results
│   ├── figures/                # Similarity methods comparison chart
│   └── translation_cache.json  # Local persistent translation cache
│
├── docs/                       # Technical documentation
│   └── prompt_design.md        # Prompt engineering architecture & decisions
│
└── tests/                      # Automated unit test suite (74 tests)
    ├── test_imports.py         # Verifies all internal symbols and DISPLAY_TYPE exist
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

### 2. Install Dependencies & spaCy English Model
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Add your API keys to `.env` if available:
- `GEMINI_API_KEY`: For Generative AI writing assistant features via the modern `google-genai` SDK (`gemini-3.5-flash` by default; configurable via `GEMINI_MODEL`). *(The app includes offline academic fallback templates if no key is supplied).*
- `GOOGLE_TRANSLATE_API_KEY`: Google Cloud Translation is supported by adding `GOOGLE_TRANSLATE_API_KEY` to `.env`; the free fallback translator is used otherwise (deep-translator or local offline engine). When running without a key, the language badge, reports, and demonstration script report "fallback translator" and never claim Google Cloud Translation.
- `BRAVE_API_KEY` or `TAVILY_API_KEY`: For live internet search. When a search key is configured in `.env`, "Search the web" is the default source mode; otherwise the app defaults to the verified reference library.

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

# 3. Run the full automated unit test suite (74 tests)
python run.py --test
```

---

## 3-Minute Live Demonstration Walkthrough

| Step | Time | Tab / Screen | Action to Perform | Key Concepts Demonstrated |
|:---:|:---:|---|---|---|
| **1** | 0:00 - 0:40 | **Tab 1: Check** | Paste text or upload a document, select source mode, and click **"Check my document"**. | Multi-format text extraction, cleaning, lemmatization, language identification, and semantic similarity matching against sources. |
| **2** | 0:40 - 1:20 | **Tab 2: Results** | Review the **Originality Score** and the 4 plain-language cards: **Copied parts**, **Tone**, **Themes**, and **Key details**. | Sentence-level exact, near-copy, and semantic similarity detection; VADER sentiment scoring; spaCy named entity recognition; KeyBERT keyword extraction. |
| **3** | 1:20 - 1:50 | **Tab 2 ➔ Tab 3** | Under "Copied parts", click **"Fix Passage"**, then switch to **Tab 3: Fix it for me** and click **"Fix this passage with AI Writing Coach"**. | Side-by-side comparison: flagged sentence vs academic rewrite, plain explanation of why it was flagged, ready-to-use APA citation, and 3 originality tips via Gemini LLM. |
| **4** | 1:50 - 2:30 | **Tab 1 ➔ Tab 2** | Submit non-English text (e.g. Hindi or Marathi), and click **"Check my document"**. | Automatic Devanagari detection (Hindi/Marathi), translation to English using the fallback translator (or Google Cloud Translation if configured), and cross-lingual semantic matching. |
| **5** | 2:30 - 3:00 | **Tab 4: About the Project** | Switch to **Tab 4: About the Project**. Inspect the **"How we read your text"** preview, the **Evaluation Benchmark Table**, the **Benchmark Chart**, and download the official **PDF report**. | Transparent evaluation methodology across 31 plagiarism pairs, 20 sentiment sentences, and 10 cross-lingual pairs (with threshold disclosure note). |

---

## Benchmark Results (From `reports/evaluation.json`)

> **Evaluation Methodology & Honesty Note**:
> - **Threshold Selection Limitation**: Similarity thresholds (0.25 for Jaccard, 0.35 for TF-IDF, 0.65 for Sentence-BERT) were chosen by inspection on this concise 31-pair benchmark dataset without a separate train/test split. Consequently, the 100.0% accuracy (F1 1.000) for sentence embeddings is optimistic.
> - **Variance across Thresholds**: In earlier runs with a stricter threshold (0.70), sentence embeddings measured **90.3% accuracy** (F1 0.909), whereas threshold 0.65 measures **100.0% accuracy** (F1 1.000) on this 31-pair benchmark.
> - **Sentiment Scoring**: Sentiment evaluation achieves **80.0% accuracy** (16/20 correct) using VADER on the 20-sentence benchmark. Earlier runs reporting 85.0% utilized different neutrality boundaries.

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

*Note: Translation follows this cascade: Google Cloud Translation (if `GOOGLE_TRANSLATE_API_KEY` is set) ➔ Gemini AI Translation (if `GEMINI_API_KEY` is set) ➔ fallback translator. Reports and badges always state the exact provider used.*

### Sentiment Analysis Benchmark (20 Labeled Sentences)
- **Overall Accuracy**: **80.0%** (16/20 sentences correctly classified).
- **Evaluation Details**: Tested on mixed academic, critical, and Devanagari sentences.

### Information Extraction Benchmark (22 Labeled Sentences)
Evaluated across English, Hindi, and Marathi sentences (including negative samples with no entities/citations):

| Task / Entity Category | Ground Truth Evaluated | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| **Named Entities (Overall)** | 22 sentences (41 entities) | **61.0%** | **62.5%** | **0.617** |
| • People (`PERSON`) | 10 true / 8 predicted | 62.5% | 50.0% | 0.556 |
| • Organizations (`ORG`) | 10 true / 14 predicted | 35.7% | 50.0% | 0.417 |
| • Places (`GPE`/`LOC`) | 8 true / 8 predicted | 50.0% | 50.0% | 0.500 |
| • Dates (`DATE`) | 12 true / 11 predicted | 100.0% | 91.7% | 0.957 |
| **Citations & References** | 22 sentences (10 citations) | **100.0%** | **90.0%** | **0.947** |

*Finding: spaCy NER (`en_core_web_sm`) correctly identifies standard entities in English text, while dates and academic citations achieve near-perfect precision via regularized parsers.*

---

## Robustness & Safe Fallbacks

The application is engineered for zero-failure operation:
- **No Internet / Offline Mode**: The app seamlessly defaults to the local verified academic reference corpus.
- **Missing Gemini API Key**: The academic writing coach gracefully falls back to structured template-based rewrites and citations.
- **Missing Google Cloud Key**: Google Cloud Translation is supported by adding `GOOGLE_TRANSLATE_API_KEY` to `.env`. If absent, the app attempts Gemini AI Translation (if `GEMINI_API_KEY` is set), and otherwise uses the free fallback translator. The language badge, reports, and demonstration script report the actual provider used ('Google Cloud Translation', 'Gemini AI Translation', or 'fallback translator') and never misrepresent the translation source.
- **Missing spaCy Model**: Named Entity Recognition falls back to rule-based pattern extractors.
- **Missing spaCy Model**: Named Entity Recognition falls back to rule-based pattern extractors.

---

## License & Academic Disclaimer
PlagiarismAI is an academic demonstration system. Similarity scores measure semantic text overlap and meaning alignment; they do not constitute legal or disciplinary proof of plagiarism. Every flagged match should be reviewed manually by an instructor or researcher.
