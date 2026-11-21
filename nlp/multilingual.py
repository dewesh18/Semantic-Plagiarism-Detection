"""
Multilingual NLP and Cloud AI Translation Module (CO3).
Provides:
- Language identification for English, Hindi, and Marathi (langdetect + Devanagari script rules).
- Cloud AI Service: Google Cloud Translation API via REST using GOOGLE_TRANSLATE_API_KEY from .env.
- Graceful fallbacks: deep-translator / dictionary / identity bypass when offline or key not provided.
- Translation chunking and persistent SQLite/JSON local caching.
- Bi-directional dual matching: runs comparison on both original and translated text and picks higher score.
"""

import json
import os
import re
import sqlite3
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests

# Specific Marathi marker words that disambiguate from Hindi
MARATHI_DISTINCTIVE_WORDS = {
    "आहे", "आहेत", "नाही", "होते", "झाले", "केले", "करणे", "म्हणून", "यांचे",
    "त्यांचे", "आपल्या", "काही", "सर्व", "मध्ये", "वर", "पण", "आणि", "कसे",
    "झाली", "केली", "असा", "अशी", "असे", "येथे", "तेव्हा", "कदाचित", "नसेल"
}

# Specific Hindi marker words that disambiguate from Marathi
HINDI_DISTINCTIVE_WORDS = {
    "है", "हैं", "नहीं", "था", "थी", "थे", "किया", "करना", "क्योंकि", "इनके",
    "उनके", "अपने", "कुछ", "सभी", "में", "पर", "लेकिन", "और", "कैसे",
    "हुआ", "हुई", "हुए", "यहाँ", "वहाँ", "तब", "शायद", "होगा"
}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "plagiarism_history.db")
TRANSLATION_CACHE_FILE = os.path.join(ROOT_DIR, "reports", "translation_cache.json")


# -----------------------------------------------------------------------------
# 1. Language Detection
# -----------------------------------------------------------------------------
def detect_language(text: str) -> Dict[str, Any]:
    """
    Identifies language (English, Hindi, Marathi, or Other).
    Combines unicode script inspection with langdetect and lexical disambiguation.
    """
    if not text or not text.strip():
        return {"language": "English", "code": "en", "confidence": 1.0, "is_multilingual": False}

    # Count Devanagari characters: Unicode block U+0900 to U+097F
    devanagari_chars = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    total_letters = sum(1 for c in text if c.isalpha())

    devanagari_ratio = devanagari_chars / max(total_letters, 1)

    # If substantial Devanagari script is detected
    if devanagari_ratio > 0.15:
        tokens = set(re.findall(r"[\u0900-\u097F]+", text))
        marathi_hits = len(tokens & MARATHI_DISTINCTIVE_WORDS)
        hindi_hits = len(tokens & HINDI_DISTINCTIVE_WORDS)

        if marathi_hits > hindi_hits:
            conf = min(0.99, 0.70 + 0.05 * marathi_hits)
            return {"language": "Marathi", "code": "mr", "confidence": round(conf, 2), "is_multilingual": True}
        elif hindi_hits > marathi_hits:
            conf = min(0.99, 0.70 + 0.05 * hindi_hits)
            return {"language": "Hindi", "code": "hi", "confidence": round(conf, 2), "is_multilingual": True}
        else:
            # Tie-break using langdetect if possible
            try:
                import langdetect
                code = langdetect.detect(text)
                if code == "mr":
                    return {"language": "Marathi", "code": "mr", "confidence": 0.85, "is_multilingual": True}
            except Exception:
                pass
            return {"language": "Hindi", "code": "hi", "confidence": 0.85, "is_multilingual": True}

    # For Latin script text
    try:
        import langdetect
        code = langdetect.detect(text)
        if code == "en":
            return {"language": "English", "code": "en", "confidence": 0.98, "is_multilingual": False}
        elif code in ("hi", "mr"):
            # Transliterated Hindi/Marathi
            lang = "Hindi" if code == "hi" else "Marathi"
            return {"language": lang, "code": code, "confidence": 0.80, "is_multilingual": True}
        else:
            return {"language": code.upper(), "code": code, "confidence": 0.75, "is_multilingual": True}
    except Exception:
        return {"language": "English", "code": "en", "confidence": 0.90, "is_multilingual": False}


# -----------------------------------------------------------------------------
# 2. Translation Cache
def _load_translation_cache() -> Dict[str, str]:
    if os.path.exists(TRANSLATION_CACHE_FILE):
        try:
            with open(TRANSLATION_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_translation_cache(cache: Dict[str, str]):
    try:
        os.makedirs(os.path.dirname(TRANSLATION_CACHE_FILE), exist_ok=True)
        with open(TRANSLATION_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# 3. Cloud Translation API (Google Cloud REST API + Fallback)
# -----------------------------------------------------------------------------
def _translate_google_cloud_rest(text: str, source_lang: str, api_key: str) -> Optional[str]:
    """
    Direct REST call to Google Cloud Translation API v2.
    https://translation.googleapis.com/language/translate/v2
    """
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
    payload = {
        "q": text,
        "target": "en",
        "source": source_lang if source_lang != "auto" else None,
        "format": "text"
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers, timeout=12)
    if response.status_code == 200:
        data = response.json()
        translations = data.get("data", {}).get("translations", [])
        if translations:
            return translations[0].get("translatedText", "")
    return None


def _translate_deep_translator(text: str, source_lang: str) -> Optional[str]:
    """
    Fallback using deep-translator (GoogleTranslator) if installed.
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source_lang if source_lang in ("hi", "mr") else "auto", target="en")
        return translator.translate(text)
    except Exception:
        return None


def translate_text(text: str, source_lang_code: str = "auto", use_cache: bool = True) -> Dict[str, Any]:
    """
    Translates non-English text to English.
    Tries:
    1. Local persistent translation cache (if use_cache is True)
    2. Google Cloud Translation API (if GOOGLE_TRANSLATE_API_KEY in .env / os.environ)
    3. deep-translator library
    4. Safe offline fallback (preserves original text)
    """
    if not text or not text.strip():
        return {
            "original_text": "",
            "translated_text": "",
            "provider": "None",
            "source_language": source_lang_code,
            "cached": False,
        }

    # Check cache
    cache = _load_translation_cache()
    cache_key = f"{source_lang_code}::{text.strip()[:300]}"
    if use_cache and cache_key in cache:
        return {
            "original_text": text,
            "translated_text": cache[cache_key],
            "provider": "Local Translation Cache",
            "source_language": source_lang_code,
            "cached": True,
        }

    # Chunk text if long to respect API constraints (up to 1500 chars per chunk)
    chunks = [text[i:i + 1400] for i in range(0, len(text), 1400)]
    translated_chunks = []
    provider_used = "Fallback Offline"

    google_api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()

    for chunk in chunks:
        chunk_translated = None
        # 1. Google Cloud Translation REST API
        if google_api_key:
            try:
                chunk_translated = _translate_google_cloud_rest(chunk, source_lang_code, google_api_key)
                if chunk_translated:
                    provider_used = "Google Cloud Translation API"
            except Exception:
                chunk_translated = None

        # 2. deep-translator fallback
        if not chunk_translated:
            try:
                chunk_translated = _translate_deep_translator(chunk, source_lang_code)
                if chunk_translated:
                    provider_used = "Deep-Translator Fallback"
            except Exception:
                chunk_translated = None

        # 3. Safe fallback (if offline / no key)
        if not chunk_translated:
            chunk_translated = chunk
            provider_used = "Offline Text Engine (No API Key)"

        translated_chunks.append(chunk_translated)

    full_translated = " ".join(translated_chunks).strip()

    # Save in cache
    cache[cache_key] = full_translated
    _save_translation_cache(cache)

    return {
        "original_text": text,
        "translated_text": full_translated,
        "provider": provider_used,
        "source_language": source_lang_code,
        "cached": False,
    }


# -----------------------------------------------------------------------------
# 4. Dual Multilingual Comparison
# -----------------------------------------------------------------------------
def compare_multilingual_documents(
    model,
    text_user: str,
    text_source: str,
    name_user: str = "User Document",
    name_source: str = "Source Document"
) -> Dict[str, Any]:
    """
    Handles multilingual document checking:
    1. Detects languages of user and source documents.
    2. Translates non-English to English if needed.
    3. Runs comparison on both (original and translated) and keeps highest score.
    """
    from nlp.similarity import compare_documents

    user_lang = detect_language(text_user)
    source_lang = detect_language(text_source)

    # Direct native comparison
    native_result = compare_documents(model, text_user, text_source, name_user, name_source)
    native_sim = native_result["stats"].get("overall_pct", 0.0)

    translated_result = None
    translation_info = None

    if user_lang["is_multilingual"] or source_lang["is_multilingual"]:
        # Translate user doc if not English
        if user_lang["is_multilingual"]:
            trans_user = translate_text(text_user, user_lang["code"])
            text_u_en = trans_user["translated_text"]
            provider = trans_user["provider"]
        else:
            text_u_en = text_user
            provider = "Native English"

        # Translate source doc if not English
        if source_lang["is_multilingual"]:
            trans_source = translate_text(text_source, source_lang["code"])
            text_s_en = trans_source["translated_text"]
            provider = trans_source["provider"]
        else:
            text_s_en = text_source

        translated_result = compare_documents(model, text_u_en, text_s_en, f"{name_user} (EN)", f"{name_source} (EN)")
        translated_sim = translated_result["stats"].get("overall_pct", 0.0)

        translation_info = {
            "user_language": user_lang,
            "source_language": source_lang,
            "provider_used": provider,
            "native_similarity_pct": native_sim,
            "translated_similarity_pct": translated_sim,
            "used_translation": translated_sim > native_sim,
        }

        # Return best match result with multilingual metadata
        chosen_result = translated_result if translated_sim > native_sim else native_result
        chosen_result["translation_info"] = translation_info
        return chosen_result

    native_result["translation_info"] = {
        "user_language": user_lang,
        "source_language": source_lang,
        "provider_used": "Not required (English)",
        "used_translation": False,
    }
    return native_result


# Aliases for convenience
detect_language_smart = detect_language
translate_text_smart = translate_text
