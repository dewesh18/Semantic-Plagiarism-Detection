"""
Topic Modeling Module for PlagiarismAI.
Provides:
- Latent Dirichlet Allocation (LDA) using scikit-learn.
- Topic inference for uploaded documents against a reference corpus.
- Graceful keyword-based topic summary for documents under ~150 words.
- Topic overlap score (cosine similarity on topic distributions) between document and matched source.
- Model caching to avoid re-training overhead.
"""

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from nlp.preprocess import STOPWORDS, clean_text

# Reference default corpus across common student & technical domains
# (Computer Science, Science, History, General Academics)
DEFAULT_REFERENCE_CORPUS = [
    "Machine learning neural networks deep learning artificial intelligence optimization gradient descent algorithms training models data features.",
    "Database management systems SQL relational transactions ACID properties normalization indexing queries durability consistency concurrency control.",
    "Computer networks protocols TCP IP routing switches packets bandwidth latency transmission control domain name resolution architecture.",
    "Software engineering design patterns agile development testing debugging version control object oriented architecture refactoring deployment.",
    "Operating systems memory management process scheduling concurrency virtualization file systems kernel threads deadlocks synchronization.",
    "Natural language processing computational linguistics tokenization embeddings sentiment analysis topic modeling transformers syntax semantics.",
    "Cybersecurity encryption cryptographic algorithms vulnerabilities authentication access control network defense penetration testing protocols.",
    "Cloud computing virtualization containers microservices distributed systems scalability load balancing serverless storage clusters.",
    "Photosynthesis chlorophyll plant biology cellular respiration genetics DNA RNA molecular biochemistry metabolism enzymes organisms.",
    "History world civilizations ancient empires political revolutions trade routes treaties economic expansion governance modern society."
]

_CACHED_LDA_MODEL = None
_CACHED_VECTORIZER = None


def get_reference_corpus(samples_dir: Optional[str] = None) -> List[str]:
    """
    Collects text from data/samples and combines with default academic references.
    """
    corpus = list(DEFAULT_REFERENCE_CORPUS)
    if samples_dir and os.path.isdir(samples_dir):
        for fname in os.listdir(samples_dir):
            fpath = os.path.join(samples_dir, fname)
            if os.path.isfile(fpath) and fname.endswith((".txt", ".md")):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read().strip()
                        if len(content.split()) >= 20:
                            corpus.append(clean_text(content))
                except Exception:
                    pass
    return corpus


def train_or_load_lda(n_topics: int = 5, samples_dir: Optional[str] = None):
    """
    Trains or returns cached scikit-learn LatentDirichletAllocation model.
    """
    global _CACHED_LDA_MODEL, _CACHED_VECTORIZER
    if _CACHED_LDA_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_LDA_MODEL, _CACHED_VECTORIZER

    corpus = get_reference_corpus(samples_dir)
    vectorizer = CountVectorizer(stop_words="english", max_features=1000, min_df=1)
    dtm = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=25,
        learning_method="batch"
    )
    lda.fit(dtm)

    _CACHED_LDA_MODEL = lda
    _CACHED_VECTORIZER = vectorizer
    return lda, vectorizer


def _extract_fallback_topics(text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Keyword-based topic extraction for short texts (< 150 words).
    """
    cleaned = clean_text(text)
    words = [w for w in cleaned.split() if len(w) >= 3 and w not in STOPWORDS]
    if not words:
        return [
            {"label": "General Academic", "top_words": ["overview", "study", "analysis"], "weight": 0.33},
            {"label": "Technical Concepts", "top_words": ["concepts", "methods", "principles"], "weight": 0.33},
            {"label": "Findings", "top_words": ["results", "evaluation", "discussion"], "weight": 0.34},
        ]

    from collections import Counter
    counts = Counter(words)
    common = counts.most_common(12)

    topics = []
    chunk_size = max(1, len(common) // top_k)
    for i in range(top_k):
        sub = common[i * chunk_size : (i + 1) * chunk_size]
        sub_words = [w for w, _ in sub] or [common[0][0]]
        primary = sub_words[0].capitalize()
        topics.append({
            "label": f"Theme: {primary}",
            "top_words": sub_words[:4],
            "weight": round(1.0 / top_k, 2),
        })
    return topics


def extract_document_topics(text: str, top_n_topics: int = 3, n_words: int = 4) -> Dict[str, Any]:
    """
    Extracts top topics from document.
    If text length < 150 words, uses robust keyword summary fallback.
    Returns:
    - topics: list of dicts with 'label', 'top_words', 'weight'
    - topic_distribution: numpy vector of topic probabilities for overlap computation
    - method_used: 'lda' or 'keyword_summary'
    """
    if not text or not text.strip():
        return {
            "topics": [],
            "topic_distribution": np.zeros(5),
            "method_used": "none",
        }

    word_count = len(text.split())

    # Fallback for short texts where LDA produces degenerate distributions
    if word_count < 150:
        fallback_topics = _extract_fallback_topics(text, top_k=top_n_topics)
        # Uniform pseudo-distribution
        dist = np.ones(5) / 5.0
        return {
            "topics": fallback_topics,
            "topic_distribution": dist,
            "method_used": "keyword_summary (short text)",
        }

    try:
        lda, vectorizer = train_or_load_lda()
        cleaned = clean_text(text)
        dtm = vectorizer.transform([cleaned])
        distribution = lda.transform(dtm)[0]  # shape (n_topics,)

        vocab = np.array(vectorizer.get_feature_names_out())
        top_topic_indices = np.argsort(distribution)[::-1][:top_n_topics]

        topics = []
        for idx in top_topic_indices:
            weight = float(distribution[idx])
            topic_word_weights = lda.components_[idx]
            top_word_indices = np.argsort(topic_word_weights)[::-1][:n_words]
            top_words = list(vocab[top_word_indices])
            primary_label = top_words[0].capitalize() if top_words else f"Topic {idx+1}"

            topics.append({
                "label": f"Theme: {primary_label}",
                "top_words": top_words,
                "weight": round(weight, 3),
            })

        return {
            "topics": topics,
            "topic_distribution": distribution,
            "method_used": "lda",
        }
    except Exception:
        fallback_topics = _extract_fallback_topics(text, top_k=top_n_topics)
        return {
            "topics": fallback_topics,
            "topic_distribution": np.ones(5) / 5.0,
            "method_used": "keyword_summary (fallback)",
        }


def compute_topic_overlap_score(dist_a: np.ndarray, dist_b: np.ndarray) -> float:
    """
    Computes cosine similarity between topic distribution vectors of two texts.
    Returns 0.0 - 100.0% overlap score.
    """
    if dist_a is None or dist_b is None:
        return 0.0
    a = np.array(dist_a, dtype=float)
    b = np.array(dist_b, dtype=float)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b))
    return round(max(0.0, min(1.0, cosine_sim)) * 100.0, 1)
