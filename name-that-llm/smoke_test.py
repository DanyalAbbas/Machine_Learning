"""Smoke-test baseline_v1_stylometry.py without waiting for the full 5-fold run locally.

Shrinks only the expensive knobs (fold count, char-ngram vocabulary). All the logic —
feature union, label alignment, fold ensemble, submission validation, parquet round trip
— runs exactly as written.
"""

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "baseline_v1_stylometry.py"

OVERRIDES = {
    "N_FOLDS = 5": "N_FOLDS = 2",
    "CHAR_MAX_FEATURES = 300_000": "CHAR_MAX_FEATURES = 40_000",
    "CHAR_NGRAM = (1, 5)": "CHAR_NGRAM = (1, 3)",
}


def main() -> int:
    code = SRC.read_text()
    for old, new in OVERRIDES.items():
        assert old in code, f"override target missing from source: {old}"
        code = code.replace(old, new)

    with tempfile.TemporaryDirectory() as tmp:
        code = code.replace(
            'OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")',
            f'OUT = Path(r"{tmp}")',
        )
        exec(compile(code, str(SRC), "exec"), {"__name__": "__main__"})

        sub = Path(tmp) / "submission.parquet"
        assert sub.is_file(), "submission.parquet not written"
        print("\nSMOKE TEST OK ->", sub)

        import pandas as pd

        out = pd.read_parquet(sub)
        print(out.to_string(index=False))
        assert list(out.columns) == ["id", "model"]
        assert not out.isna().any().any() and out.id.is_unique
    return 0


if __name__ == "__main__":
    sys.exit(main())