"""
Similarity and matching algorithms for PlagiarismAI.
Exact matching, near-exact difflib matching, and semantic embeddings matching.
"""

import math
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MIN_WORDS = 3
EXACT_MIN_WORDS = 4
SEMANTIC_MIN_WORDS = 5
MAX_SOURCE_SENTENCES = 1200
MAX_COMPARE_SENTENCES = 1500

EXACT = "Exact Match"
NEAR = "Near-Exact Match"
SEMANTIC = "Semantic Match"
TOPICAL = "Topical Similarity"
NONE = "No Significant Match"
RANK = {NONE: 0, TOPICAL: 0, SEMANTIC: 1, NEAR: 2, EXACT: 3}
COUNTED_TYPES = (EXACT, NEAR, SEMANTIC)
COLORS = {EXACT: "#dc2626", NEAR: "#f97316", SEMANTIC: "#eab308", NONE: "#9ca3af"}
HIGHLIGHT_BG = {EXACT: "#fecaca", NEAR: "#fed7aa", SEMANTIC: "#fef08a"}

CONTEXT_MIN_SIM = 0.60
DISTINCTIVE_DF = 0.10
LEXICAL_MIN_SHARED = 2
LEXICAL_MIN_COVERAGE = 0.25
COMMON_SOURCE_MIN = 3

STOPWORDS = set("""a about above after again all also an and any are as at be because been before being
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just may me more most my no nor not of off on once
only or other our out over own same she should so some such than that the their theirs them then there these
they this those through to too under until up very was we were what when where which while who whom why will
with would you your thus however therefore which whose using used use based paper show shown results result
propose proposed present presented approach method methods""".split())

_CITATION = re.compile(r"\[\s*\d+(?:\s*[,\u2013\-]\s*\d+)*\s*\]")
_ABBREVIATIONS = ["et al.", "e.g.", "i.e.", "Fig.", "Figs.", "Eq.", "Eqs.", "vs.", "Sec.",
                  "Ref.", "Refs.", "Dr.", "Prof.", "approx.", "cf.", "No.", "Tab.", "Alg."]
_DOT = "\u2024"
_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][)\"'\]\u201d\u2019]))\s+(?=[A-Z0-9\"\u201c'(\[])")


def normalize_text(text):
    """Lowercase; unify quotes/dashes; drop citation marks like [12]; strip punctuation."""
    text = unicodedata.normalize("NFKC", text or "").replace("\u00ad", "").lower()
    text = _CITATION.sub(" ", text)
    text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
    text = re.sub(r"[\u201c\u201d\u201e]", '"', text)
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_technical_token(token):
    """CamelCase (FedProx), acronyms (IEEE), digits or hyphenated terms (non-IID)."""
    return bool(re.search(r"[A-Z].*[A-Z]|[a-z][A-Z]|\d|\w-\w", token))


def extract_sentences(text, min_words=MIN_WORDS, max_words=120):
    """Split text into unique sentences; keeps short technical sentences."""
    if not text:
        return []
    text = text.replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # PDF hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # soft line breaks
    for abbr in set(_ABBREVIATIONS + [a.lower() for a in _ABBREVIATIONS]):
        text = text.replace(abbr, abbr.replace(".", _DOT))
    sentences, seen = [], set()
    for paragraph in re.split(r"\n{2,}", text):
        for raw in _SPLIT.split(paragraph):
            sentence = re.sub(r"\s+", " ", raw.replace(_DOT, ".")).strip()
            words = sentence.split()
            if len(words) < min_words:
                continue
            for k in range(0, len(words), max_words):
                chunk = " ".join(words[k:k + max_words])
                key = normalize_text(chunk)
                if key and key not in seen:
                    seen.add(key)
                    sentences.append(chunk)
    return sentences


def _exact_eligible(sentence):
    words = sentence.split()
    return len(words) >= EXACT_MIN_WORDS or (
        len(words) >= MIN_WORDS and any(_is_technical_token(w) for w in words))


def _semantic_eligible(sentence):
    return len(sentence.split()) >= SEMANTIC_MIN_WORDS


def _stem(word):
    """Very light, language-agnostic suffix stripping so store/stores/storing share one stem."""
    for suffix, margin in (("ing", 3), ("ed", 3), ("es", 3), ("ly", 3), ("s", 2)):
        if len(word) > len(suffix) + margin and word.endswith(suffix):
            word = word[:-len(suffix)]
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def content_tokens(sentence):
    """Stemmed content words (stop-words, short words and bare numbers removed)."""
    return [_stem(w) for w in normalize_text(sentence).split()
            if w not in STOPWORDS and len(w) >= 3 and not w.isdigit()]


def document_frequencies(sentence_pool):
    """How many sentences of the pool contain each content word (pool = your text + the source)."""
    df = Counter()
    for sentence in sentence_pool:
        df.update(set(content_tokens(sentence)))
    return df, max(len(sentence_pool), 1)


def lexical_evidence(sentence, passage, df, pool_size):
    """Distinctive wording shared by sentence and passage."""
    limit = max(2, DISTINCTIVE_DF * pool_size)
    idf = lambda w: math.log((pool_size + 1) / (df.get(w, 1) + 0.5))
    mine = {w for w in content_tokens(sentence) if df.get(w, 0) <= limit}
    shared = sorted(mine & set(content_tokens(passage)))
    total = sum(idf(w) for w in mine)
    coverage = sum(idf(w) for w in shared) / total if total else 0.0
    return shared, round(coverage, 3)


def _make_match(index, sentence, matched, score, match_type, level, title="", url=""):
    return {"index": int(index), "sentence": sentence, "matched": matched,
            "score": round(float(score), 2), "type": match_type, "level": level,
            "title": title, "url": url}


def detect_exact_matches(sentences, source_sentences, title="", url="", indices=None):
    """Exact match after normalisation."""
    source_norm = [(normalize_text(s), s) for s in source_sentences]
    lookup = {}
    for n, s in source_norm:
        if n:
            lookup.setdefault(n, s)
    for k in range(len(source_sentences) - 1):
        joined = source_sentences[k] + " " + source_sentences[k + 1]
        lookup.setdefault(normalize_text(joined), joined)

    matches = []
    for i in (range(len(sentences)) if indices is None else indices):
        if not _exact_eligible(sentences[i]):
            continue
        key = normalize_text(sentences[i])
        if not key:
            continue
        if key in lookup:
            matches.append(_make_match(i, sentences[i], lookup[key], 100.0, EXACT,
                                       "Identical wording", title, url))
            continue
        if len(key.split()) >= 6:
            for n, s in source_norm:
                if key in n:
                    matches.append(_make_match(i, sentences[i], s, 100.0, EXACT,
                                               "Found verbatim inside a longer passage", title, url))
                    break
    return matches


def detect_near_exact_matches(sentences, source_sentences, title="", url="",
                              threshold=0.90, indices=None):
    """Near-identical wording using difflib's character-level ratio."""
    source_norm = []
    for s in source_sentences:
        n = normalize_text(s)
        if n:
            source_norm.append((n, s, set(n.split())))
    matches = []
    for i in (range(len(sentences)) if indices is None else indices):
        if not _exact_eligible(sentences[i]):
            continue
        a = normalize_text(sentences[i])
        if not a:
            continue
        a_tokens = set(a.split())
        matcher = SequenceMatcher(None, autojunk=False)
        matcher.set_seq2(a)
        best_ratio, best_sentence = 0.0, None
        for b, original, b_tokens in source_norm:
            union = len(a_tokens | b_tokens)
            if union and len(a_tokens & b_tokens) / union < 0.45:
                continue
            matcher.set_seq1(b)
            if matcher.real_quick_ratio() < threshold or matcher.quick_ratio() < threshold:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best_ratio, best_sentence = ratio, original
        if best_sentence is not None and best_ratio >= threshold:
            match_type = EXACT if best_ratio >= 1.0 else NEAR
            matches.append(_make_match(i, sentences[i], best_sentence, best_ratio * 100,
                                       match_type, "Nearly identical wording", title, url))
    return matches


def calculate_similarity(embeddings_a, embeddings_b):
    """Cosine similarity matrix (rows = A, columns = B) using scikit-learn."""
    return cosine_similarity(embeddings_a, embeddings_b)


def semantic_level(score, threshold):
    """Human-readable level for a cosine similarity score (0-1)."""
    if score >= 0.90:
        return "Very high semantic similarity"
    if score >= 0.80:
        return "High semantic similarity"
    if score >= 0.70:
        return "Moderate semantic similarity"
    return "Low similarity (custom threshold below 70%)"


def encode_sentences(model, sentences):
    """Batch-encode sentences into unit-length BERT embeddings."""
    return model.encode(sentences, batch_size=64, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)


def detect_semantic_matches(model, sentences, source_sentences, title="", url="",
                            threshold=0.70, indices=None, user_embeddings=None,
                            source_embeddings=None):
    """BERT candidate search: best cosine passage in the source for each submitted sentence."""
    indices = list(range(len(sentences))) if indices is None else list(indices)
    indices = [i for i in indices if _semantic_eligible(sentences[i])]
    if not indices or not source_sentences:
        return []
    if user_embeddings is None:
        user_embeddings = encode_sentences(model, sentences)
    if source_embeddings is None:
        source_embeddings = encode_sentences(model, source_sentences)
    similarity = calculate_similarity(user_embeddings, source_embeddings)
    best_cols = np.argmax(similarity, axis=1)
    df, pool_size = document_frequencies(list(sentences) + list(source_sentences))
    matches = []
    for i in indices:
        best_col = int(best_cols[i])
        score = float(similarity[i][best_col])
        if score < threshold:
            continue
        context = 0
        for di in (1, -1):
            ni, nj = i + di, best_col + di
            if (0 <= ni < len(sentences) and 0 <= nj < len(source_sentences)
                    and _semantic_eligible(sentences[ni]) and int(best_cols[ni]) == nj
                    and similarity[ni][nj] >= CONTEXT_MIN_SIM):
                context += 1
        shared, coverage = lexical_evidence(sentences[i], source_sentences[best_col], df, pool_size)
        m = _make_match(i, sentences[i], source_sentences[best_col], score * 100,
                        SEMANTIC, semantic_level(score, threshold), title, url)
        m.update(context=context, shared_words=shared, coverage=coverage)
        matches.append(m)
    return matches


def finalize_semantic_matches(matches):
    """Turn semantic candidates into Semantic Match or Topical Similarity."""
    sources_per_sentence = {}
    for m in matches:
        if m["type"] == SEMANTIC:
            sources_per_sentence.setdefault(m["index"], set()).add(m.get("url") or m.get("title") or "")
    for m in matches:
        if m["type"] != SEMANTIC or "context" not in m:
            continue
        common = len(sources_per_sentence[m["index"]]) >= COMMON_SOURCE_MIN
        lexical = (len(m["shared_words"]) >= LEXICAL_MIN_SHARED and m["coverage"] >= LEXICAL_MIN_COVERAGE)
        if m["context"] > 0:
            reason = "consecutive sentences also match in order"
        elif lexical and not common:
            reason = "shares distinctive wording: " + ", ".join(m["shared_words"][:6])
        else:
            m["type"] = TOPICAL
            if common:
                m["level"] = (f"Standard statement: similar passages in {len(sources_per_sentence[m['index']])} "
                              "sources, no copying evidence (not counted)")
            else:
                m["level"] = "Related topic only: no shared distinctive wording or matching context (not counted)"
            continue
        m["level"] = f"{m['level']} - {reason}"
    return matches


def match_against_source(model, sentences, user_embeddings, source_sentences,
                         title, url, semantic_threshold, near_threshold):
    """Cascade: exact -> near-exact -> semantic. Each sentence gets one result per source."""
    everything = list(range(len(sentences)))
    matches = detect_exact_matches(sentences, source_sentences, title, url, everything)
    done = {m["index"] for m in matches}
    remaining = [i for i in everything if i not in done]
    near = detect_near_exact_matches(sentences, source_sentences, title, url,
                                     near_threshold, remaining)
    matches += near
    done |= {m["index"] for m in near}
    remaining = [i for i in everything if i not in done]
    matches += detect_semantic_matches(model, sentences, source_sentences, title, url,
                                       semantic_threshold, remaining, user_embeddings)
    return matches


def rank_source_sentences(source_sentences, user_sentences, cap):
    """Keep the `cap` page sentences that share the most content words with the submitted text."""
    if len(source_sentences) <= cap:
        return source_sentences
    vocab = set()
    for s in user_sentences:
        vocab |= {w for w in normalize_text(s).split() if len(w) >= 4 and w not in STOPWORDS}
    scored = [(len(vocab & set(normalize_text(s).split())), k) for k, s in enumerate(source_sentences)]
    keep = sorted(k for _, k in sorted(scored, key=lambda x: -x[0])[:cap])
    return [source_sentences[k] for k in keep]


def semantic_breakdown(sentence_rows, matches):
    """Unique-SENTENCE view of semantic matches."""
    sources_by_sentence = {}
    for m in matches:
        if m["type"] == SEMANTIC:
            sources_by_sentence.setdefault(m["index"], set()).add(m.get("url") or m.get("title") or "")
    also = any_sentences = pairs = 0
    for i, row in enumerate(sentence_rows):
        n = len(sources_by_sentence.get(i, ()))
        row["semantic_sources"] = n
        if n:
            any_sentences += 1
            pairs += n
            if row["status"] != SEMANTIC:
                also += 1
    total = max(len(sentence_rows), 1)
    return {"semantic_also": also, "semantic_any": any_sentences, "semantic_pairs": pairs,
            "semantic_any_pct": round(any_sentences * 100 / total, 1)}


def summarize_matches(sentences, matches):
    """Pick the strongest match per sentence and compute overall statistics."""
    best = {}
    for m in matches:
        if m["type"] not in COUNTED_TYPES:
            continue
        key = (RANK[m["type"]], m["score"])
        if m["index"] not in best or key > best[m["index"]][0]:
            best[m["index"]] = (key, m)
    rows, counts = [], {EXACT: 0, NEAR: 0, SEMANTIC: 0, NONE: 0}
    for i, sentence in enumerate(sentences):
        if i in best:
            m = best[i][1]
            status, score = m["type"], m["score"]
        else:
            status, score = NONE, 0.0
        counts[status] += 1
        rows.append({"text": sentence, "status": status, "score": score})
    total = max(len(sentences), 1)
    stats = {
        "total_sentences": len(sentences),
        "exact": counts[EXACT], "near_exact": counts[NEAR],
        "semantic": counts[SEMANTIC], "none": counts[NONE],
        "exact_pct": round((counts[EXACT] + counts[NEAR]) * 100 / total, 1),
        "semantic_pct": round(counts[SEMANTIC] * 100 / total, 1),
        "overall_pct": round((total - counts[NONE]) * 100 / total, 1),
    }
    stats.update(semantic_breakdown(rows, matches))
    stats["topical_sentences"] = len({m["index"] for m in matches if m["type"] == TOPICAL})
    stats["topical_pairs"] = len({(m["index"], m.get("url") or m.get("title") or "")
                                  for m in matches if m["type"] == TOPICAL})
    return rows, stats


def report_stat_lines(stats):
    """The 'Overall statistics' wording of the PDF report."""
    lines = [
        f"Sentences scanned: {stats['total_sentences']}",
        f"Overall similarity: {stats['overall_pct']}% of sentences matched "
        "(each sentence counted once, by its strongest match)",
        f"Exact matches: {stats['exact']} sentence(s)   |   Near-exact matches: {stats['near_exact']} "
        f"sentence(s)   (exact + near-exact: {stats['exact_pct']}%)",
        f"Semantic-only matches: {stats['semantic']} sentence(s) ({stats['semantic_pct']}%) - "
        "no exact or near-exact match in any source",
    ]
    if stats.get("semantic_any", 0):
        lines.append(
            f"Sentences with any semantic match: {stats['semantic_any']} ({stats['semantic_any_pct']}%), "
            f"in {stats['semantic_pairs']} sentence-source pair(s); {stats['semantic_also']} of these are "
            "already counted as exact/near-exact because another source matches them more strongly")
    if stats.get("topical_pairs", 0):
        lines.append(
            f"Topical similarity (NOT counted above): {stats['topical_sentences']} sentence(s) resembled "
            f"source passages in meaning ({stats['topical_pairs']} sentence-source pair(s)) but showed no "
            "evidence of copying - no matching neighbouring sentences and no distinctive shared wording, "
            "or a statement found in many sources. Listed below for manual review.")
    return lines


def compare_documents(model, text_a, text_b, name_a, name_b,
                      semantic_threshold=0.70, near_threshold=0.90):
    """Compare every sentence of document A against document B."""
    import sys
    app_module = sys.modules.get("app")
    splitter = getattr(app_module, "extract_sentences", extract_sentences)
    sentences_a = splitter(text_a)[:MAX_COMPARE_SENTENCES]
    sentences_b = splitter(text_b)[:MAX_COMPARE_SENTENCES]
    if not sentences_a or not sentences_b:
        raise ValueError(f"Both documents must contain at least one sentence with {MIN_WORDS}+ words.")
    emb_a = encode_sentences(model, sentences_a)
    emb_b = encode_sentences(model, sentences_b)

    everything = list(range(len(sentences_a)))
    matches = detect_exact_matches(sentences_a, sentences_b, name_b, "", everything)
    done = {m["index"] for m in matches}
    remaining = [i for i in everything if i not in done]
    near = detect_near_exact_matches(sentences_a, sentences_b, name_b, "", near_threshold, remaining)
    matches += near
    done |= {m["index"] for m in near}
    remaining = [i for i in everything if i not in done]
    matches += detect_semantic_matches(model, sentences_a, sentences_b, name_b, "",
                                       semantic_threshold, remaining, emb_a, emb_b)
    finalize_semantic_matches(matches)

    similarity = calculate_similarity(emb_a, emb_b)
    best_scores = similarity.max(axis=1)
    rows, stats = summarize_matches(sentences_a, matches)
    doc_similarity = calculate_similarity(emb_a.mean(axis=0, keepdims=True),
                                          emb_b.mean(axis=0, keepdims=True))[0][0]
    stats.update({"sentences_b": len(sentences_b),
                  "mean_best_similarity": round(float(best_scores.mean()) * 100, 1),
                  "document_similarity": round(float(doc_similarity) * 100, 1)})
    return {"kind": "comparison", "name_a": name_a, "name_b": name_b, "sentences": rows,
            "matches": matches, "best_scores": [round(float(s) * 100, 2) for s in best_scores],
            "stats": stats}
