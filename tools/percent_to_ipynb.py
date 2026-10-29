"""Convert Python percent cells into a Jupyter notebook, without dependencies."""

import argparse
import json
from pathlib import Path


def split_cells(source):
    cells = []
    kind = None
    lines = []

    def append_cell():
        if kind is None:
            return
        if kind == "markdown":
            content = [
                line[2:] if line.startswith("# ") else line[1:]
                if line.startswith("#") else line
                for line in lines
            ]
            cells.append({"cell_type": kind, "metadata": {}, "source": content})
        else:
            cells.append({
                "cell_type": kind,
                "metadata": {},
                "source": lines[:],
                "execution_count": None,
                "outputs": [],
            })

    for line in source.splitlines(keepends=True):
        if line.startswith("# %%"):
            append_cell()
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
        elif kind is not None:
            lines.append(line)
    append_cell()
    return cells


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    notebook = {
        "cells": split_cells(args.source.read_text(encoding="utf-8")),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    args.output.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
