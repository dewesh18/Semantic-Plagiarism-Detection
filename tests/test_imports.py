"""
Test to verify that all modules and symbols imported by app.py exist and are valid.
Prevents import regressions like missing constants or renamed functions.
"""

import ast
import unittest
from pathlib import Path


class TestImportsAndSymbols(unittest.TestCase):
    def test_display_type_defined_and_valid(self):
        """Confirm DISPLAY_TYPE exists in nlp.similarity and maps to expected plain-language labels."""
        from nlp.similarity import (
            DISPLAY_TYPE,
            EXACT,
            NEAR,
            SEMANTIC,
            TOPICAL,
            NONE,
        )

        self.assertIsInstance(DISPLAY_TYPE, dict)
        self.assertEqual(DISPLAY_TYPE[EXACT], "Exact copy")
        self.assertEqual(DISPLAY_TYPE[NEAR], "Near copy")
        self.assertEqual(DISPLAY_TYPE[SEMANTIC], "Similar meaning")
        self.assertEqual(DISPLAY_TYPE[TOPICAL], "Topical similarity")
        self.assertEqual(DISPLAY_TYPE[NONE], "Original")

    def test_app_imports_all_internal_symbols_successfully(self):
        """Parse app.py AST and verify every imported symbol from internal modules exists."""
        repo_root = Path(__file__).resolve().parent.parent
        app_path = repo_root / "app.py"
        self.assertTrue(app_path.is_file(), "app.py not found")

        tree = ast.parse(app_path.read_text(encoding="utf-8"))
        internal_modules = {
            "nlp.similarity",
            "nlp.extract",
            "nlp.preprocess",
            "nlp.analysis",
            "nlp.topics",
            "nlp.multilingual",
            "llm.assistant",
            "llm.prompts",
        }

        checked_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in internal_modules:
                    mod = __import__(node.module, fromlist=[alias.name for alias in node.names])
                    for alias in node.names:
                        checked_count += 1
                        self.assertTrue(
                            hasattr(mod, alias.name),
                            f"Module '{node.module}' does not export '{alias.name}' as expected by app.py"
                        )
        self.assertGreaterEqual(checked_count, 40)

    def test_app_module_imports_without_exception(self):
        """Import app module directly to ensure top-level code executes without ImportError."""
        import app
        self.assertTrue(hasattr(app, "main"))
        self.assertTrue(hasattr(app, "run_scan"))


if __name__ == "__main__":
    unittest.main()
