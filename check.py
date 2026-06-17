import json
with open("market_regimes.ipynb") as f:
    nb = json.load(f)
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        count = cell.get("execution_count")
        for line in cell.get("source", []):
            if "vix" in line.lower() or "CFG" in line:
                print(f"Cell Index: {i}, Count: {count} | {line.strip()}")
