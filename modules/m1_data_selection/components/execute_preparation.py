"""Execute a source notebook into a fresh external run, preserving outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient

KERNEL_TIMEOUT = 600
NOTEBOOK_VERSION = 4
REPO_PARENT_INDEX = 3


def execute(notebook, repo):
    """A fresh kernel runs every cell; source notebook is never written."""
    notebook = Path(notebook).resolve()
    repo = Path(repo).resolve()
    nb = nbformat.read(notebook, as_version=NOTEBOOK_VERSION)
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(nb, km=km, timeout=KERNEL_TIMEOUT,
                            resources={"metadata": {"path": str(repo)}})
    error = None
    try:
        client.execute()
    except Exception as exc:
        error = exc
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
            km.cleanup_resources()
    # First cell prints a compact JSON locator even if later cells fail.
    destination = None
    for cell in nb.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                for line in output.get("text", "").splitlines():
                    if line.startswith("VFL_RUN_LOCATOR="):
                        destination = Path(json.loads(line.split("=", 1)[1])["result_dir"])
    if destination is not None:
        out_path = destination / (notebook.stem + ".executed.ipynb")
        with out_path.open("x", encoding="utf-8") as stream:
            nbformat.write(nb, stream)
        if error is None:
            with (destination / "NOTEBOOK_COMPLETED.json").open("x", encoding="utf-8") as stream:
                json.dump({"notebook": out_path.name, "all_cells_executed": True}, stream)
        print(str(out_path))
    if error is not None:
        raise error
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[REPO_PARENT_INDEX])
    args = parser.parse_args()
    execute(args.notebook, args.repo)
