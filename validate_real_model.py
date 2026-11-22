"""
Validation and tuning script for PlagiarismAI's semantic evidence gate.
Runs semantic matching against defined test scenarios using the real SentenceTransformer model.
"""

import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app
from tests.support import run_pipeline

SCENARIOS = [
    {
        "name": "Exact copy scenario",
        "sentences": ["Machine learning is an active branch of artificial intelligence."],
        "pages": [("Page 1", "https://example.com/ml", ["Machine learning is an active branch of artificial intelligence."])],
        "expect": ["exact"]
    },
    {
        "name": "Semantic paraphrase scenario",
        "sentences": ["Deep neural networks allow automated computer vision and image classification."],
        "pages": [("Page 2", "https://example.com/vision", ["Convolutional architectures empower automated image recognition systems."])],
        "expect": ["counted"]
    },
    {
        "name": "Original unrelated scenario",
        "sentences": ["The speed of light in a vacuum is approximately three hundred thousand kilometers per second."],
        "pages": [("Page 3", "https://example.com/chem", ["Photosynthesis converts radiant solar energy into biochemical carbohydrate bonds."])],
        "expect": ["not_counted"]
    }
]


def meets(expected: str, actual: str) -> bool:
    """Verifies whether actual match type meets expectation string."""
    if expected == "exact":
        return actual == app.EXACT
    if expected == "counted":
        return actual in (app.EXACT, app.NEAR, app.SEMANTIC)
    if expected == "not_counted":
        return actual in (app.NONE, app.TOPICAL)
    return False


def evaluate(model, scenarios):
    """Evaluates scenarios through run_pipeline using provided embedding model."""
    results = []
    for sc in scenarios:
        matches, rows, stats = run_pipeline(model, sc["sentences"], sc["pages"])
        for r in rows:
            if "type" not in r:
                r["type"] = r.get("status", app.NONE)
        oks = []
        for row, exp in zip(rows, sc["expect"]):
            oks.append(meets(exp, row["type"]))
        results.append((sc, matches, rows, stats, oks))
    return results


def print_report(results):
    """Prints diagnostic evaluation report for semantic evidence gate."""
    print("Report on semantic evidence gate (NOT an accuracy measurement)")
    for sc, matches, rows, stats, oks in results:
        status_str = "PASS" if all(oks) else "FAIL"
        print(f"  [{status_str}] {sc['name']}: {len(rows)} sentences evaluated.")


def main():
    """Main entrypoint running validation against the real multilingual embedding model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(app.MODEL_NAME)
    results = evaluate(model, SCENARIOS)
    print_report(results)


if __name__ == "__main__":
    main()
