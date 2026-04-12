"""Test that graders can be imported and called from repo root - simulating platform validator."""
import importlib
import sys
import os

# DO NOT add anything to sys.path - simulate platform environment
# (Just use the fact that cwd is the repo root, which Python adds to sys.path[0])

mod = importlib.import_module("graders")

grader_classes = [name for name in dir(mod) if name.endswith("Grader") and name != "_BaseGrader"]
print(f"Found {len(grader_classes)} grader classes")

results = []
for name in grader_classes:
    cls = getattr(mod, name)
    instance = cls()
    result = instance()
    score = result["score"]
    ok = 0.0 < score < 1.0
    results.append((name, score, ok))
    if not ok:
        print(f"  FAIL: {name} score={score}")

passed = sum(1 for _, _, ok in results if ok)
print(f"Passed: {passed}/{len(results)}")
if passed == len(results):
    print("ALL GRADERS PASS - import works from repo root")
else:
    print("SOME GRADERS FAILED")
    sys.exit(1)
