#!/usr/bin/env python3
"""Convert cattle_buffalo_trainer.py to .ipynb notebook format.

Parses the `# %%` cell markers and `# %% [markdown]` blocks to create
a proper Jupyter notebook that Colab can open directly.
"""
import json
import re
import sys
from pathlib import Path


def py_to_ipynb(py_path: str, ipynb_path: str = None):
    """Convert a percent-format .py script to .ipynb."""
    py_path = Path(py_path)
    if ipynb_path is None:
        ipynb_path = py_path.with_suffix(".ipynb")
    else:
        ipynb_path = Path(ipynb_path)

    content = py_path.read_text(encoding="utf-8")
    lines = content.split("\n")

    cells = []
    current_lines = []
    current_type = "code"  # default

    def flush():
        if not current_lines:
            return
        # Strip trailing empty lines
        while current_lines and current_lines[-1].strip() == "":
            current_lines.pop()
        if not current_lines:
            return

        if current_type == "markdown":
            # Remove leading "# " from each line (markdown comment prefix)
            md_lines = []
            for line in current_lines:
                if line.startswith("# "):
                    md_lines.append(line[2:])
                elif line.strip() == "#":
                    md_lines.append("")
                else:
                    md_lines.append(line)
            source = [l + "\n" for l in md_lines]
        else:
            source = [l + "\n" for l in current_lines]

        # Remove trailing newline from last line
        if source and source[-1].endswith("\n"):
            source[-1] = source[-1]  # keep it for notebook format

        cell = {
            "cell_type": current_type,
            "metadata": {},
            "source": source,
        }
        if current_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    for line in lines:
        stripped = line.strip()

        # Check for cell markers — markdown FIRST (it also starts with "# %% ")
        if stripped.startswith("# %% [markdown]"):
            flush()
            current_lines = []
            current_type = "markdown"
            continue

        if stripped == "# %%" or stripped.startswith("# %% "):
            flush()
            current_lines = []
            current_type = "code"
            continue

        current_lines.append(line)

    flush()

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "name": "Cattle & Buffalo Breed Classifier — Training"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": cells,
    }

    ipynb_path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"✅ Created {ipynb_path} ({len(cells)} cells)")
    return str(ipynb_path)


if __name__ == "__main__":
    script = Path(__file__).parent / "cattle_buffalo_trainer.py"
    output = Path(__file__).parent / "cattle_buffalo_trainer.ipynb"
    py_to_ipynb(str(script), str(output))
