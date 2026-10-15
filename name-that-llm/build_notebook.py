"""Convert the `%%`-style .py source into a .ipynb.

Usage:
    python3 build_notebook.py
"""

import argparse
import re
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
CELL_RE = re.compile(r"^#\s*%%\s*(\[[^\]]*\])?\s*$")


def parse_cells(src: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, str]] = []
    kind, buf = "code", []
    for line in src.splitlines():
        m = CELL_RE.match(line)
        if m:
            if buf and "".join(buf).strip():
                cells.append((kind, "\n".join(buf).strip("\n")))
            buf = []
            tag = (m.group(1) or "").strip("[]")
            kind = "markdown" if tag == "markdown" else "code"
            if tag in {"raw", "skip"}:
                kind = "raw"
            continue
        buf.append(line)
    if buf and "".join(buf).strip():
        cells.append((kind, "\n".join(buf).strip("\n")))
    return cells


def to_notebook(src: str) -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    for kind, source in parse_cells(src):
        if kind == "markdown":
            # markdown cells live as `# comment` lines in the .py source
            text = "\n".join(re.sub(r"^#\s?", "", ln) for ln in source.splitlines())
            nb.cells.append(nbformat.v4.new_markdown_cell(text))
        elif kind == "code":
            nb.cells.append(nbformat.v4.new_code_cell(source))
        else:
            nb.cells.append(nbformat.v4.new_raw_cell(source))
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    nbformat.validate(nb)
    return nb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="baseline_v1_stylometry.py")
    ap.add_argument("--out", default="notebook/baseline_v1_stylometry.ipynb")
    args = ap.parse_args()

    src_path, out_path = HERE / args.source, HERE / args.out
    nb = to_notebook(src_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_path)
    print(f"wrote {out_path} ({len(nb.cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())