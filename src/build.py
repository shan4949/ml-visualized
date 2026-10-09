"""Compile src/*.nb.txt into notebooks/*.ipynb.

Format: cells separated by lines that are exactly '##MD' or '##CODE'.
The token {{SETUP}} inside a code cell is replaced by the shared setup code,
and {{BADGE:name}} in markdown by an Open-in-Colab badge for notebooks/name.ipynb.
"""
import sys, re, pathlib, nbformat

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO = "shan4949/ml-visualized"
SETUP = (ROOT / "src" / "setup_cell.py").read_text().strip()

def badge(name):
    url = f"https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{name}.ipynb"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"

def build(src):
    text = src.read_text()
    parts = re.split(r"^##(MD|CODE)[ \t]*$", text, flags=re.M)
    cells = []
    for kind, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if kind == "MD":
            body = re.sub(r"\{\{BADGE:([\w\-]+)\}\}", lambda m: badge(m.group(1)), body)
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            body = body.replace("{{SETUP}}", SETUP)
            cells.append(nbformat.v4.new_code_cell(body))
    nb = nbformat.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "toc_visible": True},
    }
    out = ROOT / "notebooks" / src.name.replace(".nb.txt", ".ipynb")
    nbformat.write(nb, out)
    print(f"built {out.name}: {len(cells)} cells")

if __name__ == "__main__":
    targets = sys.argv[1:] or sorted((ROOT / "src").glob("*.nb.txt"))
    for t in targets:
        build(pathlib.Path(t))
