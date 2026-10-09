"""Execute notebooks in place and strip interactive-widget state.

Static figures and animations are kept so the notebooks preview on GitHub and open
in Colab with outputs visible. Slider (ipywidgets) outputs are removed: they only work
with a live kernel, and saved widget state breaks GitHub's notebook renderer.

Usage: python src/finalize.py [notebooks/xx.ipynb ...]
"""
import sys, pathlib, nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
WIDGET = "application/vnd.jupyter.widget-view+json"

def finalize(path):
    nb = nbformat.read(path, as_version=4)
    NotebookClient(nb, timeout=1200, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        cell.outputs = [o for o in cell.outputs if WIDGET not in o.get("data", {})]
        cell.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    nbformat.write(nb, path)
    print(f"{path.name}: {path.stat().st_size/1e6:.1f} MB")

if __name__ == "__main__":
    targets = [pathlib.Path(p) for p in sys.argv[1:]] or sorted((ROOT / "notebooks").glob("*.ipynb"))
    for t in targets:
        finalize(t)
