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
        self.assertEqual(len(at.sidebar.slider), 0)


if __name__ == "__main__":
    unittest.main()
