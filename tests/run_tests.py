"""Tiny test runner for machines without pytest. Run from the repo root:
    python tests/run_tests.py
"""
import inspect
import os
import pathlib
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tests.test_uq_study as t  # noqa: E402

passed = failed = 0
for name, fn in inspect.getmembers(t, inspect.isfunction):
    if not name.startswith("test_"):
        continue
    try:
        if "tmp_path" in inspect.signature(fn).parameters:
            fn(pathlib.Path(tempfile.mkdtemp()))
        else:
            fn()
        print("PASS", name)
        passed += 1
    except Exception:
        print("FAIL", name)
        traceback.print_exc()
        failed += 1
print(f"{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)