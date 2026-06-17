import json

notebook_path = "/Users/alper/Documents/antigravity/peaceful-brahmagupta/market_regimes.ipynb"
with open(notebook_path, "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "calm_threshold=CFG.VIX_CALM_THRESHOLD" in line:
                source[i] = line.replace("CFG.VIX_CALM_THRESHOLD", "20")
            elif "transitional_threshold=CFG.VIX_TRANSITIONAL_THRESHOLD" in line:
                source[i] = line.replace("CFG.VIX_TRANSITIONAL_THRESHOLD", "30")

with open(notebook_path, "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
