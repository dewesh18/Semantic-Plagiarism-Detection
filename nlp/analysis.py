"""
Sentiment Analysis and Information Extraction Module (CO2).
Provides:
- Sentiment / Tone analysis with confidence (overall and per-paragraph).
- Named Entity Recognition (spaCy with fallback).
- Keyword extraction (KeyBERT / frequency fallback).
- Reference / citation extraction (regex for [1], (Author, 2020), DOI, URLs).
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List

# -----------------------------------------------------------------------------
# 1. Sentiment / Tone Analysis
# -----------------------------------------------------------------------------
_POS_WORDS = {
    "good", "great", "excellent", "positive", "strong", "effective", "novel", "superior",
    "improved", "beneficial", "accurate", "impressive", "success", "innovative", "advantage",
    "efficient", "robust", "promising", "significant", "valuable", "satisfactory"
}

_NEG_WORDS = {
    "bad", "poor", "negative", "weak", "flawed", "inferior", "limitation", "failure",
    "ineffective", "error", "defect", "adverse", "risk", "harmful", "decline", "drawback",
    "difficult", "loss", "vulnerable", "inaccurate", "problem", "bottleneck"
}

# Devanagari positive / negative words (Hindi / Marathi)
_DEV_POS_WORDS = {"चांगला", "उत्कृष्ट", "सकारात्मक", "फायदेशीर", "सफल", "अच्छा", "उत्तम", "लाभदायक", "प्रभावी"}
_DEV_NEG_WORDS = {"वाईट", "नकारात्मक", "दोष", "नुकसान", "विफल", "बुरा", "खराब", "हानिकारक", "कमी"}


def _analyze_sentiment_lexicon(text: str) -> Dict[str, Any]:
    """Fallback rule-based sentiment classifier for English and Hindi/Marathi."""
    lower = text.lower()
    words = [w.strip(".,;:!?\"'()[]{}—–-") for w in lower.split()]
    words = [w for w in words if w]
    if not words:
        return {"label": "Neutral", "confidence": 0.5, "compound": 0.0}

    pos_count = sum(1 for w in words if w in _POS_WORDS or w in _DEV_POS_WORDS)
    neg_count = sum(1 for w in words if w in _NEG_WORDS or w in _DEV_NEG_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return {"label": "Neutral", "confidence": 0.85, "compound": 0.0}

    compound = (pos_count - neg_count) / max(total, 1)
    if compound >= 0.15:
        label = "Positive"
        conf = round(min(0.95, 0.60 + 0.35 * (pos_count / total)), 2)
    elif compound <= -0.15:
        label = "Negative"
        conf = round(min(0.95, 0.60 + 0.35 * (neg_count / total)), 2)
    else:
        label = "Neutral"
        conf = 0.70

    return {"label": label, "confidence": conf, "compound": round(compound, 2)}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes document sentiment and per-paragraph tone.
    Uses VADER if available, otherwise graceful multilingual rule-based lexicon.
    """
    if not text or not text.strip():
        return {
            "overall": {"label": "Neutral", "confidence": 0.5, "score": 0.0},
            "paragraphs": [],
        }

    vader_analyzer = None
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader_analyzer = SentimentIntensityAnalyzer()
    except Exception:
        try:
            import nltk.sentiment.vader as v
            vader_analyzer = v.SentimentIntensityAnalyzer()
        except Exception:
            vader_analyzer = None

    def get_scores(segment: str) -> Dict[str, Any]:
        if vader_analyzer:
            try:
                s = vader_analyzer.polarity_scores(segment)
                comp = s.get("compound", 0.0)
                if comp >= 0.05:
                    label = "Positive"
                    conf = round(min(0.98, 0.60 + comp * 0.38), 2)
                elif comp <= -0.05:
                    label = "Negative"
                    conf = round(min(0.98, 0.60 + abs(comp) * 0.38), 2)
                else:
                    label = "Neutral"
                    conf = round(1.0 - abs(comp), 2)
                return {"label": label, "confidence": conf, "score": round(comp, 2)}
            except Exception:
                pass
        res = _analyze_sentiment_lexicon(segment)
        return {"label": res["label"], "confidence": res["confidence"], "score": res["compound"]}

    overall_res = get_scores(text)

    # Paragraphs breakdown
    raw_paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    para_list = []
    for idx, p in enumerate(raw_paras[:10]):  # cap to first 10 for clean UI
        if len(p.split()) >= 4:
            p_score = get_scores(p)
            para_list.append({
                "paragraph_index": idx + 1,
                "text_snippet": p[:140] + ("..." if len(p) > 140 else ""),
                "label": p_score["label"],
                "confidence": p_score["confidence"],
                "score": p_score["score"],
            })

    return {
        "overall": overall_res,
        "paragraphs": para_list,
    }


# -----------------------------------------------------------------------------
# 2. Information Extraction: Named Entities
# -----------------------------------------------------------------------------
def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extracts named entities (people, organizations, places, dates) using spaCy.
    Falls back to regex heuristics if spaCy or the model is not installed.
    """
    results: Dict[str, List[str]] = {
        "people": [],
        "organizations": [],
        "places": [],
        "dates": [],
    }
    if not text:
        return results

    # Attempt spaCy
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:20000])  # limit length for fast analysis
        for ent in doc.ents:
            val = ent.text.strip()
            if len(val) < 2:
                continue
            if ent.label_ == "PERSON" and val not in results["people"]:
                results["people"].append(val)
            elif ent.label_ in ("ORG", "COMPANY") and val not in results["organizations"]:
                results["organizations"].append(val)
            elif ent.label_ in ("GPE", "LOC", "FAC") and val not in results["places"]:
                results["places"].append(val)
            elif ent.label_ in ("DATE", "TIME") and val not in results["dates"]:
                results["dates"].append(val)
        return {k: v[:15] for k, v in results.items()}
    except Exception:
        # Graceful fallback: regex-based entity heuristic
        pass

    # Dates
    dates = re.findall(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b",
        text,
        flags=re.I,
    )
    for d in dates:
        if d not in results["dates"] and len(d) > 2:
            results["dates"].append(d)

    # Capitalized Words / Organizations (heuristics)
    cap_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    org_keywords = {"University", "Institute", "Corporation", "Corp", "Inc", "Labs", "Google", "Microsoft", "OpenAI", "IEEE", "ACM"}
    for p in cap_phrases:
        if any(ok in p for ok in org_keywords):
            if p not in results["organizations"]:
                results["organizations"].append(p)
        else:
            if p not in results["people"]:
                results["people"].append(p)

    return {k: v[:10] for k, v in results.items()}


# -----------------------------------------------------------------------------
# 3. Keyword Extraction
# -----------------------------------------------------------------------------
def extract_keywords(text: str, top_n: int = 8) -> List[str]:
    """
    Extracts key phrases using KeyBERT (if available) or TF-IDF / Frequency heuristic.
    """
    if not text:
        return []

    try:
        from keybert import KeyBERT
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
        return [kw[0] for kw in keywords]
    except Exception:
        pass

    # Frequency + Distinctiveness fallback
    words = [w.lower().strip(".,;:!?\"'()[]{}") for w in text.split()]
    from nlp.preprocess import STOPWORDS
    content_words = [w for w in words if len(w) >= 4 and w not in STOPWORDS and not w.isdigit()]
    counts = Counter(content_words)
    return [word for word, _ in counts.most_common(top_n)]


# -----------------------------------------------------------------------------
# 4. Reference / Citation Extraction
# -----------------------------------------------------------------------------
def extract_citations(text: str) -> Dict[str, List[str]]:
    """
    Detects academic citations, DOIs, and reference URLs using regex:
    - Numeric brackets: [1], [2, 3], [4-7]
    - Author-year: (Smith, 2020), (Vaswani et al., 2017)
    - DOIs: 10.1145/..., 10.1038/...
    - Reference URLs: https://...
    """
    if not text:
        return {"brackets": [], "author_year": [], "dois": [], "urls": []}

    bracket_citations = re.findall(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]", text)
    author_year_citations = re.findall(r"\(\s*[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?,\s*\d{4}\s*\)", text)
    dois = re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    urls = re.findall(r"https?://[^\s)\]]+", text)

    return {
        "brackets": sorted(list(set(bracket_citations))),
        "author_year": sorted(list(set(author_year_citations))),
        "dois": sorted(list(set(dois))),
        "urls": sorted(list(set(urls))),
    }
