"""Print a notebook's saved outputs as plain text and extract its figures.

Usage: python scripts/notebook_outputs.py NOTEBOOK.ipynb [--images-dir DIR]

Prints each executed code cell's text output (stdout/stderr, text results,
errors) under the nearest preceding markdown heading, and writes embedded PNGs
to DIR (default: /tmp/<notebook name>_figures).
"""
import argparse
import base64
import json
import re
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--images-dir", type=Path)
    args = parser.parse_args()

    nb = json.loads(args.notebook.read_text())
    images_dir = args.images_dir or Path("/tmp") / f"{args.notebook.stem}_figures"
    heading = ""
    n_images = 0
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown":
            heading = next((line for line in source.splitlines() if line.startswith("#")), heading)
            continue
        outputs = cell.get("outputs", [])
        if not outputs:
            continue
        print(f"\n===== cell {i} [{heading}] (execution {cell.get('execution_count')}) =====")
        print("> " + source.splitlines()[0])
        for out in outputs:
            kind = out["output_type"]
            if kind == "stream":
                print("".join(out["text"]), end="")
            elif kind in ("execute_result", "display_data"):
                data = out.get("data", {})
                if "image/png" in data:
                    images_dir.mkdir(parents=True, exist_ok=True)
                    path = images_dir / f"cell{i:02d}_{n_images}.png"
                    path.write_bytes(base64.b64decode(data["image/png"]))
                    n_images += 1
                    print(f"[figure saved to {path}]")
                elif "text/plain" in data:
                    print("".join(data["text/plain"]))
            elif kind == "error":
                print(f"ERROR {out['ename']}: {out['evalue']}")
                print(ANSI.sub("", "\n".join(out.get("traceback", []))))


if __name__ == "__main__":
    main()
