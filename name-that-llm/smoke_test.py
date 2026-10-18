"""Smoke-test a notebook source locally without a GPU.

Runs the real source with fewer folds and a smaller vocabulary so the whole path -
feature union, clustering, branch plumbing, blend search, cluster assignment and the
submission write/read-back - executes on a laptop. The submission is built through the
tiny interactive test.csv, which is exactly the path a Kaggle rerun takes.

The embedding branch is stubbed via NTL_NO_QWEN=1 and is verified on GPU in Kaggle.

Usage:
    python3 smoke_test.py v2_qwen_clusters.py
    python3 smoke_test.py v15_clusters_cpu.py

Run from the repository root - the notebook sources locate data/ relative to the
working directory, so `data/` must be visible from where you launch this.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Only the expensive knobs shrink. All logic is untouched.
OVERRIDES = {
    "N_FOLDS = 5": "N_FOLDS = 2",
    "max_features=300_000": "max_features=40_000",
    'ngram_range=(1, 5), sublinear_tf=True,\n                                 lowercase=False':
        'ngram_range=(1, 3), sublinear_tf=True,\n                                 lowercase=False',
}


def main() -> int:
    src = HERE / (sys.argv[1] if len(sys.argv) > 1 else "v2_qwen_clusters.py")
    if not src.is_file():
        print(f"no such source: {src}")
        return 1
    code = src.read_text()
    applied = 0
    for old, new in OVERRIDES.items():
        if old in code:
            code = code.replace(old, new)
            applied += 1
    print(f"smoke-testing {src.name} ({applied} overrides applied)")

    with tempfile.TemporaryDirectory() as tmp:
        code = code.replace(
            'OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")',
            f'OUT = Path(r"{tmp}")',
        )
        os.environ["NTL_NO_QWEN"] = "1"          # stub the GPU embedding branch
        exec(compile(code, str(src), "exec"), {"__name__": "__main__"})

        import numpy as np
        import pandas as pd

        sub = pd.read_parquet(Path(tmp) / "submission.parquet")
        print("\nSMOKE TEST OK")
        print(sub.to_string(index=False))
        assert list(sub.columns) == ["id", "model"]
        assert len(sub) == 3 and sub.id.is_unique and not sub.isna().any().any()

        for name in sorted(p.name for p in Path(tmp).iterdir()):
            print("  artefact:", name)
        summary = Path(tmp) / "cv_summary.json"
        if summary.exists():
            print("cv_summary:", json.dumps(json.loads(summary.read_text()), indent=2))
        for probe in Path(tmp).glob("*.npz"):
            print(f"{probe.name} keys:", list(np.load(probe, allow_pickle=False).keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())