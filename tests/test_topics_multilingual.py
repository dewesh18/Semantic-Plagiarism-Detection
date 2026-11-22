"""
Unit tests for Topic Modeling and Multilingual NLP (nlp/topics.py, nlp/multilingual.py).
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from nlp.topics import (
    extract_document_topics,
    compute_topic_overlap_score,
    _extract_fallback_topics,
)
from nlp.multilingual import (
    detect_language,
    translate_text,
    compare_multilingual_documents,
)


class TestTopics(unittest.TestCase):
    def test_short_document_topic_summary(self):
        text = "Deep learning and neural networks use gradient descent and backpropagation."
        res = extract_document_topics(text)
        self.assertIn("topics", res)
        self.assertEqual(len(res["topics"]), 3)
        self.assertIn("keyword_summary", res["method_used"])

    def test_long_document_topic_inference(self):
        # Long document > 150 words
        para = "Machine learning neural networks deep learning artificial intelligence optimization gradient descent algorithms training models data features. " * 15
        res = extract_document_topics(para)
        self.assertIn("topics", res)
        self.assertIn("topic_distribution", res)
        self.assertTrue(len(res["topics"]) > 0)

    def test_topic_overlap_score(self):
        v1 = np.array([0.8, 0.1, 0.1, 0.0, 0.0])
        v2 = np.array([0.8, 0.1, 0.1, 0.0, 0.0])
        v3 = np.array([0.0, 0.0, 0.0, 0.5, 0.5])

        score_identical = compute_topic_overlap_score(v1, v2)
        score_orthogonal = compute_topic_overlap_score(v1, v3)

        self.assertAlmostEqual(score_identical, 100.0, places=1)
        self.assertAlmostEqual(score_orthogonal, 0.0, places=1)


class TestMultilingual(unittest.TestCase):
    def test_detect_language_english(self):
        info = detect_language("This is a clean English academic research paper on machine learning.")
        self.assertEqual(info["language"], "English")
        self.assertEqual(info["code"], "en")
        self.assertFalse(info["is_multilingual"])

    def test_detect_language_marathi(self):
        info = detect_language("मशीन लर्निंग ही कृत्रिम बुद्धिमत्तेची एक आधुनिक शाखा आहे.")
        self.assertEqual(info["language"], "Marathi")
        self.assertEqual(info["code"], "mr")
        self.assertTrue(info["is_multilingual"])

    def test_detect_language_hindi(self):
        info = detect_language("प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानव भाषा समझने में मदद करता है।")
        self.assertEqual(info["language"], "Hindi")
        self.assertEqual(info["code"], "hi")
        self.assertTrue(info["is_multilingual"])

    @patch("nlp.multilingual.requests.post")
    def test_translate_google_cloud_mock(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {
                "translations": [{"translatedText": "Machine learning is a subfield of AI."}]
            }
        }
        mock_post.return_value = mock_resp

        with patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "test-key-xyz"}):
            res = translate_text("मशीन लर्निंग ही कृत्रिम बुद्धिमत्तेची शाखा आहे.", "mr", use_cache=False)
            self.assertEqual(res["translated_text"], "Machine learning is a subfield of AI.")
            self.assertEqual(res["provider"], "Google Cloud Translation API")

    @patch("google.genai.Client")
    def test_translate_gemini_client_mock(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.text = "Natural language processing is a branch of artificial intelligence."
        mock_client.models.generate_content.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        from nlp.multilingual import _translate_gemini
        out = _translate_gemini("प्राकृतिक भाषा प्रसंस्करण कंप्यूटर विज्ञान की एक शाखा है।", "hi", "fake-key")
        self.assertEqual(out, "Natural language processing is a branch of artificial intelligence.")

    @patch("nlp.multilingual._translate_gemini")
    def test_translate_gemini_provider_precedence(self, mock_gemini):
        mock_gemini.return_value = "Artificial intelligence enables automated reasoning."
        with patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "", "GEMINI_API_KEY": "fake-gemini-key"}):
            res = translate_text("कृत्रिम बुद्धिमत्ता स्वयंचलित तर्क सक्षम करते.", "mr", use_cache=False)
            self.assertEqual(res["translated_text"], "Artificial intelligence enables automated reasoning.")
            self.assertEqual(res["provider"], "Gemini AI Translation")

    def test_translate_offline_fallback_does_not_crash(self):
        # With no API key and mocked deep_translator, returns string without error
        with patch.dict("os.environ", {"GOOGLE_TRANSLATE_API_KEY": "", "GEMINI_API_KEY": ""}):
            res = translate_text("नमस्ते जग", "hi", use_cache=False)
            self.assertTrue(len(res["translated_text"]) > 0)
            self.assertIn("provider", res)
            self.assertEqual(res["provider"], "fallback translator")


if __name__ == "__main__":
    unittest.main()
