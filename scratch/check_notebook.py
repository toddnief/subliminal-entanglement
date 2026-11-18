import json
with open('/home/tnief/1-Projects/subliminal-entanglement/notebooks/paper_figures.ipynb') as f:
    nb = json.load(f)
for i in [62, 63, 64]:
    c = nb['cells'][i]
    src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
    print(f"=== Cell [{i}] ({c['cell_type']}, {len(src)} chars) ===")
    if c['cell_type'] == 'code' and src.strip():
        try:
            compile(src, f'cell_{i}', 'exec')
            print('  SYNTAX OK')
        except SyntaxError as e:
            print(f'  SYNTAX ERROR: {e}')
    elif c['cell_type'] == 'markdown':
        print(f'  first line: {src.splitlines()[0]}')
    else:
        print('  (empty)')
