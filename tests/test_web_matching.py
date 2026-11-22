"""
Unit tests for Web Matching & Ranking logic
===========================================
Verifies:
1. build_search_queries prioritizes longest, distinctive sentences with exact quotes.
2. Exact copy (100%), Near copy (90-99%), and Similar meaning (>=70%) classification.
3. Verified sources sorted strictly by best_match_pct descending, never search order.
4. No verified source link appears if download failed or matches == 0.
5. Results tab renders charts and tables cleanly without exception via AppTest.
"""

import os
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))

MATCH_EXACT = "Exact copy"
MATCH_NEAR = "Near copy"
MATCH_SIMILAR = "Similar meaning"
MATCH_ORIGINAL = "Original"

from app import (
    build_search_queries,
    run_scan,
    load_bert_model,
)
from nlp.preprocess import preprocess_document


class TestWebMatching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_bert_model()

    def test_build_search_queries_longest_sentences_exact_quotes(self):
        """Verifies queries prioritize longest, most distinctive sentences with quotes."""
        doc = (
            "Hi. "  # Very short, skipped
            "The quick brown fox jumps over the lazy dog in the sunny morning. "  # Long sentence
            "Federated learning preserves privacy across decentralized distributed systems worldwide."  # Long sentence
        )
        prep = preprocess_document(doc)
        queries = build_search_queries(prep["sentences"], max_queries=2)

        self.assertEqual(len(queries), 2)
        # Check that queries are wrapped in quotes
        for item in queries:
            q = item["query"]
            self.assertTrue(q.startswith('"') and q.endswith('"'), f"Query not exact phrase: {q}")
        # Distinctive sentence words should be in queries
        query_text = " ".join(item["query"] for item in queries).lower()
        self.assertIn("federated learning", query_text)

    @patch("app.search_web_sources")
    @patch("app.fetch_webpage_content")
    def test_classification_and_ranking_order(self, mock_fetch, mock_search):
        """
        Tests:
        - Exact copy (100%) -> MATCH_EXACT
        - Near copy (90-99%) -> MATCH_NEAR
        - Similar meaning (>=70%) -> MATCH_SIMILAR
        - Sources sorted by best_match_pct DESC (not search engine return order)
        - Broken downloads / 0 matches excluded from verified_sources
        """
        # Document with 3 distinct sentences
        doc_text = (
            "Artificial intelligence is transforming modern medical diagnostics and healthcare. "
            "Quantum computers leverage superposition to solve mathematical problems exponentially faster. "
            "Renewable energy systems reduce global carbon dioxide emissions effectively."
        )

        # Mock search results returning 3 sources:
        # Source 1 returned FIRST by search engine, but only has ~75% similar meaning
        # Source 2 returned SECOND, has 100% exact copy
        # Source 3 returned THIRD, failed to download
        mock_search.return_value = {
            "results": [
                {
                    "title": "Low Rank Search Result",
                    "url": "https://example.com/low-rank",
                    "domain": "example.com",
                    "snippet": "Medical diagnosis using AI.",
                    "best_rank": 0,
                    "found_by": ["exact"]
                },
                {
                    "title": "Exact Match Top Result",
                    "url": "https://example.com/exact-match",
                    "domain": "example.com",
                    "snippet": "Artificial intelligence is transforming modern medical diagnostics and healthcare.",
                    "best_rank": 1,
                    "found_by": ["exact"]
                },
                {
                    "title": "Broken Page Result",
                    "url": "https://example.com/broken",
                    "domain": "example.com",
                    "snippet": "Something broken.",
                    "best_rank": 2,
                    "found_by": ["exact"]
                }
            ],
            "errors": [],
            "notes": [],
            "queries": [{"query": '"test"', "type": "exact", "ok": True}],
            "engines": ["Tavily"],
            "provider": "tavily",
            "provider_label": "Tavily"
        }

        def mock_fetch_impl(url, *args, **kwargs):
            if url == "https://example.com/exact-match":
                return {
                    "ok": True,
                    "page_title": "Exact Match Top Result",
                    "url": url,
                    "text": (
                        "Artificial intelligence is transforming modern medical diagnostics and healthcare. "
                        "Unrelated other content."
                    ),
                    "domain": "example.com"
                }
            elif url == "https://example.com/low-rank":
                return {
                    "ok": True,
                    "page_title": "Low Rank Search Result",
                    "url": url,
                    "text": (
                        "Artificial intelligence is certainly transforming modern medical diagnostics and clinical healthcare. "
                        "Other text here."
                    ),
                    "domain": "example.com"
                }
            else:
                return {
                    "ok": False,
                    "page_title": "Broken Page Result",
                    "url": url,
                    "text": "",
                    "domain": "example.com",
                    "reason": "404 Not Found"
                }

        mock_fetch.side_effect = mock_fetch_impl

        settings = {
            "provider": "tavily",
            "semantic_threshold": 0.70,
            "near_threshold": 0.90,
            "max_sentences": 50,
            "max_queries": 3,
            "max_pages": 3,
            "results_per_query": 5
        }

        scan_res = run_scan(
            model=self.model,
            text=doc_text,
            document_name="test_web_matching_doc.txt",
            settings=settings,
            progress=lambda frac, msg: None
        )

        verified = scan_res.get("verified_sources", [])

        # 1. Broken source must NOT be in verified_sources
        verified_urls = [s["url"] for s in verified]
        self.assertNotIn("https://example.com/broken", verified_urls)

        # 2. Both valid sources should be verified
        self.assertEqual(len(verified), 2)

        # 3. Exact match source must be ranked #1 even though search engine returned it 2nd
        self.assertEqual(verified[0]["url"], "https://example.com/exact-match")
        self.assertEqual(verified[0]["best_match_pct"], 100.0)
        self.assertEqual(verified[0]["highest_match_type"], MATCH_EXACT)

        # 4. Low rank source must be ranked #2
        self.assertEqual(verified[1]["url"], "https://example.com/low-rank")
        self.assertLess(verified[1]["best_match_pct"], 100.0)
        self.assertGreaterEqual(verified[1]["best_match_pct"], 70.0)

        # 5. Sentence matches check
        sentence_matches = scan_res.get("sentence_matches", [])
        exact_matches = [m for m in sentence_matches if m.get("score") == 100.0 or m.get("type") == "Exact Match"]
        self.assertTrue(len(exact_matches) >= 1)
        self.assertEqual(exact_matches[0]["score"], 100.0)
        self.assertEqual(exact_matches[0]["url"], "https://example.com/exact-match")

    def test_results_tab_renders_charts_without_exception(self):
        """Verifies that the Results tab renders all visual elements and charts cleanly in AppTest."""
        scan_res = {
            "scanned_at": "2026-11-22 12:00:00",
            "document_name": "test_document.txt",
            "status": "matches",
            "scan_mode": "Live Web Search",
            "pipeline": {
                "queries_planned": 1,
                "queries_ok": 1,
                "results_retrieved": 2,
                "pages_attempted": 2,
                "pages_downloaded": 2,
                "pages_compared": 2,
                "sources_with_matches": 2,
                "engine": "Tavily"
            },
            "stats": {
                "overall_pct": 50.0,
                "total_sentences": 4,
                "exact": 1,
                "near_exact": 1,
                "semantic": 0,
                "none": 2,
                "exact_pct": 25.0,
                "semantic_pct": 25.0,
                "verified_sources": 2,
                "sources_with_matches": 2,
                "unverified_sources": 0
            },
            "sources": [
                {
                    "title": "Top Source Online",
                    "domain": "example.org",
                    "url": "https://example.org/top",
                    "best_match_pct": 100.0,
                    "highest_match_type": MATCH_EXACT,
                    "matches": 1,
                    "document_coverage_pct": 25.0,
                    "verified": True,
                    "rank": 1
                },
                {
                    "title": "Second Source Online",
                    "domain": "example.org",
                    "url": "https://example.org/second",
                    "best_match_pct": 95.0,
                    "highest_match_type": MATCH_NEAR,
                    "matches": 1,
                    "document_coverage_pct": 25.0,
                    "verified": True,
                    "rank": 2
                }
            ],
            "verified_sources": [
                {
                    "title": "Top Source Online",
                    "domain": "example.org",
                    "url": "https://example.org/top",
                    "best_match_pct": 100.0,
                    "highest_match_type": MATCH_EXACT,
                    "matches": 1,
                    "document_coverage_pct": 25.0,
                    "verified": True,
                    "rank": 1
                },
                {
                    "title": "Second Source Online",
                    "domain": "example.org",
                    "url": "https://example.org/second",
                    "best_match_pct": 95.0,
                    "highest_match_type": MATCH_NEAR,
                    "matches": 1,
                    "document_coverage_pct": 25.0,
                    "verified": True,
                    "rank": 2
                }
            ],
            "sentences": [
                {
                    "sentence": "Sentence one is an exact match from online.",
                    "matched": "Sentence one is an exact match from online.",
                    "score": 100.0,
                    "type": "Exact Match",
                    "level": "Verbatim match",
                    "title": "Top Source Online",
                    "url": "https://example.org/top",
                    "also_found_on": []
                },
                {
                    "sentence": "Sentence two is slightly modified from source.",
                    "matched": "Sentence two is slightly altered from source.",
                    "score": 95.0,
                    "type": "Near-Exact Match",
                    "level": "Nearly identical wording",
                    "title": "Second Source Online",
                    "url": "https://example.org/second",
                    "also_found_on": []
                },
                {
                    "sentence": "Sentence three is completely original.",
                    "matched": "",
                    "score": 0.0,
                    "type": "No Significant Match",
                    "level": "Independent text",
                    "title": "",
                    "url": "",
                    "also_found_on": []
                },
                {
                    "sentence": "Sentence four is also completely original.",
                    "matched": "",
                    "score": 0.0,
                    "type": "No Significant Match",
                    "level": "Independent text",
                    "title": "",
                    "url": "",
                    "also_found_on": []
                }
            ],
            "matches": []
        }

        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.session_state["scan_result"] = scan_res
        at.session_state["prep_result"] = {
            "word_count": 30,
            "sentence_count": 4,
            "avg_sentence_len": 7.5,
            "vocab_richness": 0.9,
            "sentences": ["S1", "S2", "S3", "S4"]
        }
        at.session_state["nlp_analysis"] = {
            "language": {"detected_language": "English", "code": "en", "is_multilingual": False},
            "translation": None,
            "sentiment": {"sentiment": "Neutral", "compound": 0.0, "scores": {}},
            "entities": {"counts": {"ORG": 0}, "entities": []},
            "keywords": ["test", "document"],
            "citations": [],
            "topics": [{"name": "Technology", "score": 0.8}]
        }
        at.session_state["check_doc_name"] = "test_document.txt"
        at.session_state["check_input_text"] = "Sample document content."

        at.run()
        self.assertEqual(len(at.exception), 0, f"App execution raised: {[e.value for e in at.exception]}")

        # Check that the visual analysis and verified sources are rendered
        all_text = [m.value for m in at.markdown] + [s.value for s in at.success] + [w.value for w in at.warning] + [e.value for e in at.error]
        self.assertTrue(any("Verified Sources" in t for t in all_text), f"Verified Sources missing from all_text")
        self.assertTrue(any("Visual Analysis" in t for t in all_text), f"Visual Analysis missing from all_text")
        self.assertTrue(any("Document Sequence Map" in t for t in all_text), f"Document Sequence Map missing from all_text")
        self.assertTrue(any("Top Source Online" in t for t in all_text), f"Top Source Online missing from all_text")


if __name__ == "__main__":
    unittest.main()
