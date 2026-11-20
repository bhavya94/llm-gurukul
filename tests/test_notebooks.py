import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from vak.data import DATA_DIR, FILE

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "vak").glob("*.ipynb"))


def _printed(cell) -> str:
    """Everything a code cell printed or returned, as one text."""
    text = ""
    for output in cell.outputs:
        text += output.get("text", "") + output.get("data", {}).get("text/plain", "")
    return text


def _reads_the_stories(notebook) -> bool:
    """True when a code cell calls download() or load_split(). Both read the 249 MB file, and download it first if it
    is missing. A test never downloads it."""
    for cell in notebook.cells:
        if cell.cell_type == "code" and ("download()" in cell.source or "load_split()" in cell.source):
            return True
    return False


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.name)
def test_each_notebook_still_prints_what_its_text_says(path):
    """Run the notebook afresh. Each code cell must print what the saved notebook shows, and each `quoted` value
    in a "What you see" note must appear in the output of the code cell above it."""
    saved = nbformat.read(path, as_version=4)
    if _reads_the_stories(saved) and not (DATA_DIR / FILE).exists():
        pytest.skip("needs the stories file: run `uv run vak data` once to download it")
    fresh = nbformat.read(path, as_version=4)
    NotebookClient(fresh, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    above = ""
    for old, new in zip(saved.cells, fresh.cells):
        if new.cell_type == "code":
            above = _printed(new)
            assert above == _printed(old), f"this cell now prints something else:\n{new.source}"
        elif new.source.startswith("**What you see**"):
            for quoted in re.findall(r"`([^`]+)`", new.source):
                assert quoted in above, f"{quoted!r} is not in the output above it"
