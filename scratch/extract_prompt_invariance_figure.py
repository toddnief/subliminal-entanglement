import base64
import json
from pathlib import Path


repo_root = Path(__file__).resolve().parents[1]
notebook_path = repo_root / "notebooks" / "toy_attention_write_gate_reuse.ipynb"
output_path = (
    repo_root
    / "figures"
    / "toy_models"
    / "prompt_invariance_subliminal_transfer.png"
)

with notebook_path.open() as notebook_file:
    notebook = json.load(notebook_file)

cell = next(cell for cell in notebook["cells"] if cell.get("id") == "004931d1")
image_data = next(
    output["data"]["image/png"]
    for output in cell["outputs"]
    if output.get("output_type") == "display_data"
    and "image/png" in output.get("data", {})
)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(base64.b64decode(image_data))
print(output_path)
