import json
import sys


def describe(obj, prefix="", depth=0, max_depth=6):
    pad = "  " * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                size = len(v)
                print(f"{pad}{k}: {type(v).__name__}[{size}]")
                if depth < max_depth:
                    describe(v, prefix + "." + k, depth + 1, max_depth)
            else:
                print(f"{pad}{k}: {v!r}"[:200])
    elif isinstance(obj, list) and obj:
        first = obj[0]
        if isinstance(first, (dict, list)):
            print(f"{pad}[0]:")
            describe(first, prefix + "[0]", depth + 1, max_depth)
        else:
            print(f"{pad}{obj[:6]!r}"[:200])


for path in sys.argv[1:]:
    print("=" * 20, path)
    describe(json.load(open(path)))
