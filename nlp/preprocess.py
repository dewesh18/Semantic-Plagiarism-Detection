"""
Preprocessing and Vectorization Pipeline (CO1).
Extracts text, cleans, tokenizes, removes stopwords, lemmatizes,
and provides TF-IDF + sentence embedding vectorization.
"""

import math
import re
import unicodedata
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp.extract import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_txt,
    extract_text_from_upload,
)
from nlp.similarity import (
    STOPWORDS as DEFAULT_STOPWORDS,
    extract_sentences as base_extract_sentences,
    normalize_text,
    _stem,
)

# Multilingual default sentence-transformers model
MULTILINGUAL_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENGLISH_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Extended stopwords for English + common Hindi/Marathi transliterated/Devanagari stopwords
STOPWORDS = set(DEFAULT_STOPWORDS) | {
    "and", "the", "is", "in", "it", "of", "to", "this", "that", "with", "for", "on", "as",
    "are", "was", "were", "by", "an", "be", "at", "from", "or", "which", "has", "had",
    "है", "के", "में", "की", "का", "और", "से", "पर", "को", "एक", "हैं", "तो", "भी", "या", "हा", "आहे", "नाही"
}


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Cleans unstructured text by:
    1. Normalizing unicode (NFKC)
    2. Stripping URLs, citations [1], DOIs, page numbers
    3. Lowercasing
    4. Removing special characters & extra whitespace
    5. Tokenization, stopword removal, and lemmatization (spaCy / NLTK / fallback stemmer)
    """
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove DOIs & academic markers
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", " ", text)
    text = re.sub(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\(\s*[A-Za-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\s*\)", " ", text)

    # Remove page numbers like "Page 12" or standalone pagination
    text = re.sub(r"\b(?:page|pg\.?)\s*\d+\b", " ", text, flags=re.I)

    # Lowercase
    text = text.lower()

    # Remove quotes, dashes, non-word characters except intra-word hyphens and spaces
    text = re.sub(r"[\u2018\u2019\u201a\u201b']", "", text)
    text = re.sub(r"[\u201c\u201d\u201e\"]", " ", text)
    text = re.sub(r"[^\w\s\-]", " ", text)

    tokens = text.split()
    cleaned_tokens = []

    # Lemmatizer fallback strategy
    lemmatizer = None
    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
    except Exception:
        lemmatizer = None

    for tok in tokens:
        tok = tok.strip("-_")
        if not tok or len(tok) < 2:
            continue
        if remove_stopwords and tok in STOPWORDS:
            continue

        if lemmatize:
            if lemmatizer:
                try:
                    tok = lemmatizer.lemmatize(tok)
                except Exception:
                    tok = _stem(tok)
            else:
                tok = _stem(tok)

        cleaned_tokens.append(tok)

    return " ".join(cleaned_tokens)


def preprocess_document(text: str) -> Dict[str, Any]:
    """
    Returns original text, cleaned text, and sentence list for before/after preview.
    """
    sentences = base_extract_sentences(text)
    cleaned = clean_text(text)
    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "sentence_count": len(sentences),
        "word_count": len(text.split()),
        "cleaned_word_count": len(cleaned.split()),
    }


def compute_tfidf_vectors(corpus: List[str]):
    """
    Vectorizes a list of documents or sentences using scikit-learn's TF-IDF.
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def compute_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Word-level Jaccard similarity between two texts.
    """
    tokens_a = set(clean_text(text_a).split())
    tokens_b = set(clean_text(text_b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return float(intersection / union) if union > 0 else 0.0


def compute_tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    TF-IDF cosine similarity between two texts.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf = vectorizer.fit_transform([text_a, text_b])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except Exception:
        return 0.0


def compute_embedding_similarity(model, text_a: str, text_b: str) -> float:
    """
    Cosine similarity using sentence embeddings.
    """
    if not text_a or not text_b:
        return 0.0
    embs = model.encode([text_a, text_b], convert_to_numpy=True, normalize_embeddings=True)
    sim = cosine_similarity(embs[0:1], embs[1:2])[0][0]
    return float(sim)


def compute_combined_originality_score(
    jaccard_sim: float,
    tfidf_sim: float,
    embedding_sim: float,
    weights: Tuple[float, float, float] = (0.2, 0.3, 0.5)
) -> Dict[str, float]:
    """
    Combines Jaccard, TF-IDF cosine, and embedding cosine into:
    - combined_similarity_pct (0-100%)
    - originality_score (100 - combined_similarity_pct)
    """
    w_j, w_t, w_e = weights
    combined_similarity = (w_j * jaccard_sim) + (w_t * tfidf_sim) + (w_e * embedding_sim)
    similarity_pct = round(max(0.0, min(1.0, combined_similarity)) * 100, 1)
    originality_pct = round(100.0 - similarity_pct, 1)

    return {
        "jaccard_sim": round(jaccard_sim, 4),
        "tfidf_sim": round(tfidf_sim, 4),
        "embedding_sim": round(embedding_sim, 4),
        "combined_similarity_pct": similarity_pct,
        "originality_score": originality_pct,
    }
