"""
PlagiarismAI CLI & Application Runner
====================================
Quick entry point for launching the Streamlit interface, running evaluation
benchmarks, or executing the complete unit test suite.

Usage:
  python run.py              # Launch the Streamlit application
  python run.py --evaluate   # Run CO1-CO3 benchmark and generate evaluation.json + figures
  python run.py --test       # Run all unit tests
"""

import os
import sys
import subprocess


def run_app():
    print("[*] Launching PlagiarismAI Streamlit Application...")
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[*] Application stopped.")


def run_evaluate():
    print("[*] Running PlagiarismAI Rubric Benchmark (CO1, CO2, CO3)...")
    cmd = [sys.executable, "evaluate.py"]
    subprocess.run(cmd)


def run_tests():
    print("[*] Running PlagiarismAI Unit Test Suite...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"tests{os.pathsep}."
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]
    res = subprocess.run(cmd, env=env)
    sys.exit(res.returncode)


def main():
    args = sys.argv[1:]
    if "--evaluate" in args or "--eval" in args:
        run_evaluate()
    elif "--test" in args or "--tests" in args:
        run_tests()
    elif "--help" in args or "-h" in args:
        print(__doc__)
    else:
        run_app()


if __name__ == "__main__":
    main()
