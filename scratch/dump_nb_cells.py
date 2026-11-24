import json
import sys

nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    print(f"===== [{i}] {c['cell_type']} id={c.get('id')} =====")
    print(src)
    if c["cell_type"] == "code":
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                txt = "".join(o["text"])
                print("--- stream ---")
                print(txt[:1500])
            elif o.get("output_type") == "error":
                print("--- error ---", o.get("ename"), o.get("evalue"))
