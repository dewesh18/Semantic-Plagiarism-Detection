"""
Unit tests for CO2 Analysis and Information Extraction (nlp/analysis.py).
"""

import unittest
from nlp.analysis import (
    analyze_sentiment,
    extract_named_entities,
    extract_keywords,
    extract_citations,
)


class TestAnalysis(unittest.TestCase):
    def test_sentiment_positive_and_negative(self):
        pos = analyze_sentiment("The experimental results show outstanding, excellent and superior improvement.")
        self.assertEqual(pos["overall"]["label"], "Positive")
        self.assertGreaterEqual(pos["overall"]["confidence"], 0.60)

        neg = analyze_sentiment("The algorithm suffers from severe defects, high error rates and complete failure.")
        self.assertEqual(neg["overall"]["label"], "Negative")
        self.assertGreaterEqual(neg["overall"]["confidence"], 0.60)

    def test_sentiment_hindi_marathi_fallback(self):
        hm = analyze_sentiment("या संशोधनामुळे अत्यंत उत्कृष्ट आणि फायदेशीर निकाल मिळाले आहेत.")
        self.assertEqual(hm["overall"]["label"], "Positive")

    def test_citations_extraction(self):
        doc = "Prior work [12] and recent advances (Vaswani et al., 2017) with DOI 10.1038/nature12373 at https://nature.com"
        cits = extract_citations(doc)
        self.assertTrue(any("[12]" in b for b in cits["brackets"]))
        self.assertTrue(len(cits["author_year"]) > 0)
        self.assertTrue(len(cits["dois"]) > 0)
        self.assertTrue(len(cits["urls"]) > 0)

    def test_named_entities_fallback_extracts_dates_and_orgs(self):
        text = "In 2021, researchers at Google Labs and Stanford University met in Paris on Jan 15, 2022."
        ents = extract_named_entities(text)
        self.assertIn("dates", ents)
        self.assertIn("organizations", ents)
        self.assertTrue(len(ents["dates"]) > 0)

    def test_keywords_extraction(self):
        text = "Deep learning and convolutional neural networks are commonly applied in artificial intelligence computer vision."
        kws = extract_keywords(text, top_n=4)
        self.assertTrue(len(kws) > 0)


if __name__ == "__main__":
    unittest.main()
