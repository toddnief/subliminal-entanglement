import json, re, sys
out_path = "/home/tnief/1-Projects/subliminal-entanglement/scratch/full_ft_status.txt"
sys.stdout = open(out_path, "w")
path = "/net/projects2/interp/subliminal/shared/results/registry.json"
with open(path) as f:
    reg = json.load(f)
exps = reg["experiments"]
target_animals = ["cat", "owl", "eagle", "wolf", "dolphin"]
pat = re.compile(r"^(?P<animal>[a-z]+)_subliminal_full_seed(?P<gs>\d+)(?:_tseed(?P<ts>\d+))?_range100_999_qwen$")
by_animal = {a: {} for a in target_animals}
for eid, rec in exps.items():
    m = pat.match(eid)
    if not m:
        continue
    a = m.group("animal")
    if a not in by_animal:
        continue
    gs = int(m.group("gs"))
    ts = int(m.group("ts")) if m.group("ts") else 1
    by_animal[a][(gs, ts)] = rec.get("status", "?")

seeds = [1, 42, 123]
print(f"{'animal':<8} {'cell':<18} {'status':<12}")
for a in target_animals:
    for gs in seeds:
        for ts in seeds:
            s = by_animal[a].get((gs, ts), "MISSING")
            print(f"{a:<8} gen={gs:<3} tr={ts:<3}     {s}")
    print()

print("=== Summary (status == 'completed') ===")
for a in target_animals:
    statuses = list(by_animal[a].values())
    completed = sum(1 for s in statuses if s == "completed")
    not_completed = [(k, v) for k, v in by_animal[a].items() if v != "completed"]
    print(f"  {a}: {completed}/9 completed; non-completed cells: {not_completed}")
