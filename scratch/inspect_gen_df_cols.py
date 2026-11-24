import pandas as pd
df = pd.read_parquet("/net/projects2/interp/subliminal/shared/results/views/gen_df.parquet")
print("columns:", list(df.columns))
print("shape:", df.shape)
for c in df.columns:
    if "variant" in c.lower() or "sys" in c.lower() or "prompt" in c.lower() or "dwg" in c.lower():
        print(f"  unique {c}: {df[c].dropna().unique()[:10]}")
