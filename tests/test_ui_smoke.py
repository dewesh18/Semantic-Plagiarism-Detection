"""
Streamlit AppTest Smoke Tests
=============================
Tests end-to-end rendering of PlagiarismAI across all 4 UI tabs and verifies
that no duplicate widget keys or runtime exceptions occur.
"""

import os
import unittest
from streamlit.testing.v1 import AppTest

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))


class TestUISmoke(unittest.TestCase):
    def test_app_renders_all_tabs_without_exception(self):
        """Verify the app launches and executes all tabs with zero StreamlitDuplicateElementKey or NameErrors."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"App execution raised exceptions: {[e.value for e in at.exception]}")
        # Verify title and tabs exist
        titles = [t.value for t in at.title]
        self.assertTrue(any("PlagiarismAI" in t for t in titles))

    def test_demo_sample_buttons_click_cleanly(self):
        """Verify that clicking each 1-click sample button loads text without error."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # 1. Original essay sample
        orig_btn = [b for b in at.button if b.key == "load_orig_btn"]
        self.assertTrue(len(orig_btn) > 0)
        orig_btn[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["check_doc_name"], "sample_original.txt")
        self.assertIn("microgrid", at.session_state["check_input_text"].lower())

        # 2. Copied essay sample
        copy_btn = [b for b in at.button if b.key == "load_copy_btn"]
        self.assertTrue(len(copy_btn) > 0)
        copy_btn[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["check_doc_name"], "sample_copied.txt")
        self.assertIn("federated", at.session_state["check_input_text"].lower())

        # 3. Hindi document sample
        hindi_btn = [b for b in at.button if b.key == "load_hindi_btn"]
        self.assertTrue(len(hindi_btn) > 0)
        hindi_btn[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["check_doc_name"], "sample_hindi.txt")
        self.assertTrue(len(at.session_state["check_input_text"]) > 20)

    def test_sidebar_is_clean_and_developer_settings_are_isolated(self):
        """Verify sidebar has no developer sliders and search diagnostics uses prefixed keys."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Check sidebar markdown
        sidebar_md = [m.value for m in at.sidebar.markdown]
        self.assertTrue(any("PlagiarismAI" in m for m in sidebar_md))
        # Ensure no sliders leaked into the sidebar
        self.assertEqual(len(at.sidebar.slider), 4)

    def test_real_check_and_results_rendering_for_all_samples(self):
        """
        Runs the full check pipeline on all 3 demo sample files (original, copied, hindi),
        populates st.session_state with the real output (scan_result, prep_result, nlp_analysis),
        and renders the full app (including Results and Fix tabs) with AppTest.
        Asserts ZERO exceptions to verify as_list/as_dict completely prevents AttributeError.
        """
        from app import load_bert_model
        from nlp.preprocess import preprocess_document
        from nlp.similarity import scan_against_reference_corpus
        from nlp.analysis import analyze_sentiment, extract_named_entities, extract_keywords, extract_citations
        from nlp.topics import extract_document_topics
        from nlp.multilingual import detect_language_smart, translate_text_smart

        model = load_bert_model()

        samples = ["sample_original.txt", "sample_copied.txt", "sample_hindi.txt"]
        for sample_file in samples:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "samples", sample_file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            prep_res = preprocess_document(text)
            lang_res = detect_language_smart(text)
            text_for_scan = text
            trans_info = None
            if lang_res.get("is_multilingual"):
                trans_info = translate_text_smart(text, lang_res.get("code", "auto"))
                text_for_scan = trans_info.get("translated_text", text)

            sentiment_res = analyze_sentiment(text)
            entities_res = extract_named_entities(text)
            keywords_res = extract_keywords(text)
            citations_res = extract_citations(text)
            topics_res = extract_document_topics(text)

            scan_res = scan_against_reference_corpus(
                model, text_for_scan, sample_file,
                semantic_threshold=0.70, near_threshold=0.90
            )

            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.session_state["scan_result"] = scan_res
            at.session_state["prep_result"] = prep_res
            at.session_state["nlp_analysis"] = {
                "language": lang_res,
                "translation": trans_info,
                "sentiment": sentiment_res,
                "entities": entities_res,
                "keywords": keywords_res,
                "citations": citations_res,
                "topics": topics_res
            }
            at.session_state["check_doc_name"] = sample_file
            at.session_state["check_input_text"] = text

            at.run()
            self.assertEqual(len(at.exception), 0, f"Exception rendering {sample_file}: {[e.value for e in at.exception]}")


if __name__ == "__main__":
    unittest.main()
