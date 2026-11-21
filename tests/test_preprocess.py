"""
Unit tests for CO1 Preprocessing and Vectorization (nlp/preprocess.py).
"""

import unittest
from nlp.preprocess import (
    clean_text,
    preprocess_document,
    compute_jaccard_similarity,
    compute_tfidf_cosine_similarity,
    compute_combined_originality_score,
)


class TestPreprocess(unittest.TestCase):
    def test_clean_text_removes_urls_citations_and_punctuation(self):
        raw = "Check out https://arxiv.org/abs/1706.03762 [1] and (Smith, 2020) on Page 15! It is fantastic."
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("[1]", cleaned)
        self.assertNotIn("page 15", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertTrue(len(cleaned) > 0)

    def test_preprocess_document_returns_expected_keys(self):
        text = "Federated Learning enables decentralized privacy. It keeps data on mobile devices."
        res = preprocess_document(text)
        self.assertIn("original_text", res)
        self.assertIn("cleaned_text", res)
        self.assertIn("sentences", res)
        self.assertEqual(res["sentence_count"], 2)

    def test_jaccard_and_tfidf_similarity_identical_texts(self):
        text = "Support vector machines find optimal dividing hyperplanes."
        j_sim = compute_jaccard_similarity(text, text)
        t_sim = compute_tfidf_cosine_similarity(text, text)
        self.assertAlmostEqual(j_sim, 1.0, places=2)
        self.assertAlmostEqual(t_sim, 1.0, places=2)

    def test_jaccard_and_tfidf_similarity_unrelated_texts(self):
        text_a = "Quantum computing relies on superposition."
        text_b = "Baking sourdough bread takes flour, water, and fermentation."
        j_sim = compute_jaccard_similarity(text_a, text_b)
        t_sim = compute_tfidf_cosine_similarity(text_a, text_b)
        self.assertAlmostEqual(j_sim, 0.0, places=2)
        self.assertAlmostEqual(t_sim, 0.0, places=2)

    def test_combined_originality_score_bounds(self):
        score = compute_combined_originality_score(1.0, 1.0, 1.0)
        self.assertEqual(score["originality_score"], 0.0)
        self.assertEqual(score["combined_similarity_pct"], 100.0)

        score_zero = compute_combined_originality_score(0.0, 0.0, 0.0)
        self.assertEqual(score_zero["originality_score"], 100.0)
        self.assertEqual(score_zero["combined_similarity_pct"], 0.0)


if __name__ == "__main__":
    unittest.main()
