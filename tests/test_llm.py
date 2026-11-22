"""
Unit tests for LLM Assistant and Prompts (llm/prompts.py, llm/assistant.py).
"""

import json
import unittest
from unittest.mock import patch, MagicMock

from llm.prompts import build_fix_prompt, SYSTEM_ROLE_INSTRUCTION
from llm.assistant import fix_passage, _clean_json_response, _get_fallback_fix, verify_gemini_model_at_startup


class TestLLMAssistant(unittest.TestCase):
    def test_verify_gemini_model_at_startup(self):
        status = verify_gemini_model_at_startup()
        self.assertIn("model", status)
        self.assertIn("sdk", status)
        self.assertEqual(status["sdk"], "google-genai")

    def test_build_fix_prompt_structure_and_safety_delimiters(self):
        prompt = build_fix_prompt("This is a flagged sentence.", "Sample Source", "https://example.com")
        self.assertIn("<<<STUDENT_FLAGGED_PASSAGE>>>", prompt)
        self.assertIn("<<</STUDENT_FLAGGED_PASSAGE>>>", prompt)
        self.assertIn("This is a flagged sentence.", prompt)
        self.assertIn("Sample Source", prompt)
        self.assertIn("Example 1:", prompt)

    def test_clean_json_response_strips_code_fences(self):
        raw = "```json\n{\"explanation\": \"test\", \"rewrite\": \"test rewrite\"}\n```"
        cleaned = _clean_json_response(raw)
        self.assertTrue(cleaned.startswith("{"))
        self.assertTrue(cleaned.endswith("}"))

    def test_fallback_fix_returns_complete_keys_without_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            res = fix_passage("Neural networks process data through layers.", "AI Textbook", "https://ai.org")
            self.assertIn("explanation", res)
            self.assertIn("rewrite", res)
            self.assertIn("citation", res)
            self.assertIn("tips", res)
            self.assertEqual(len(res["tips"]), 3)
            self.assertIn("Template Fallback", res["provider"])

    @patch("llm.assistant.call_gemini_api")
    def test_fix_passage_successful_mocked_gemini_call(self, mock_api):
        mock_output = json.dumps({
            "explanation": "Matches standard textbook phrasing.",
            "rewrite": "A refined sentence structure written freshly.",
            "citation": "Referenced from Doe (2020).",
            "tips": ["Tip A", "Tip B", "Tip C"]
        })
        mock_api.return_value = f"```json\n{mock_output}\n```"

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key"}):
            res = fix_passage("Test passage for rewriting", "Paper Title", "https://paper.url")
            self.assertEqual(res["explanation"], "Matches standard textbook phrasing.")
            self.assertEqual(res["rewrite"], "A refined sentence structure written freshly.")
            self.assertEqual(len(res["tips"]), 3)
            self.assertIn("Gemini API", res["provider"])

    @patch("llm.assistant.call_gemini_api")
    def test_fix_passage_handles_api_error_gracefully(self, mock_api):
        mock_api.side_effect = RuntimeError("ResourceExhausted: 429 Rate limit exceeded")

        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake-gemini-key"}):
            res = fix_passage("A passage that triggered rate limit", "Source", "https://source.com")
            # Must not raise an exception; must return friendly fallback
            self.assertIn("explanation", res)
            self.assertIn("rewrite", res)
            self.assertIn("rate limit", res.get("notice", "").lower())


if __name__ == "__main__":
    unittest.main()
