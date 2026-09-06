"""Tests for the generated HLWM v10.0 Kaggle notebook.

The notebook is the executable preregistration, and two of its cells carry
launch-blocking invariants that no bundle test can see because they live in
generated JSON rather than in the shipped package:

* cell 3 must select the code tree by CONTENT and fail closed, because a
  resume attaches a previous kernel output that contains a complete old copy
  of this tree with an identical manifest name, version and counts;
* cell 9 must audit the two seeds in separate per-GPU processes, because a
  crash while auditing the first seed destroyed the second seed's audit in
  both session I-4 and session I-5.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from hlwm_v100_notebook import notebook_document  # noqa: E402

COUNTS = {"train": 10570, "validation": 1457, "test": 1737}
DIGEST = "f" * 64


def _cells(document):
    return ["".join(cell["source"]) for cell in document["cells"]]


def test_cell3_pins_the_code_digest_and_fails_closed():
    sources = _cells(notebook_document(COUNTS, DIGEST))
    cell = next(source for source in sources if "EXPECTED_CODE_DIGEST" in source)
    assert repr(DIGEST) in cell
    # Selection is by content match, and an unmatched attachment stops the run.
    assert "matching = [item for item in candidates if item[2] == EXPECTED_CODE_DIGEST]" in cell
    assert "if not matching:" in cell
    assert "raise FileNotFoundError" in cell
    # The coin-flip indexing into an unfiltered candidate list is gone.
    assert "archives[0]" not in cell
    assert "manifests[0]" not in cell
    # The per-file sha verification still runs after selection.
    assert "code sha mismatch" in cell


def test_cell3_digest_changes_with_the_build():
    first = _cells(notebook_document(COUNTS, "a" * 64))
    second = _cells(notebook_document(COUNTS, "b" * 64))
    assert first != second, "the pin must be part of the notebook bytes"


def test_cell9_audits_one_seed_per_gpu_in_separate_processes():
    sources = _cells(notebook_document(COUNTS, DIGEST))
    cell = next(source for source in sources if "audit_processes" in source)
    assert "evaluate_v10.py" in cell
    assert "CUDA_VISIBLE_DEVICES" in cell
    assert "for gpu, seed in enumerate(SEEDS)" in cell
    # One failed seed audit must not raise past the other.
    assert "continue" in cell
    assert "if not gates_by_seed:" in cell


def test_cell11_keeps_a_checkpoint_when_training_aborted():
    sources = _cells(notebook_document(COUNTS, DIGEST))
    cell = next(source for source in sources if "resumable" in source and "ARTIFACTS" in source)
    assert "checkpoint-step-*.pt'" in cell  # promotes the newest raw step
    assert "stale.name not in keep" in cell


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
                print("PASS", name)
            except AssertionError as error:
                failures += 1
                print("FAIL", name, error)
    sys.exit(1 if failures else 0)
