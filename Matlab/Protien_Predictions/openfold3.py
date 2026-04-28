import json
import os

# Paths
json_path = "/Users/bharadwajanandivada/Downloads/KSR_MEK.json"
out_dir = "/Users/bharadwajanandivada/SCMPA/Matlab/Protien_Predictions"
os.makedirs(out_dir, exist_ok=True)

# Load JSON
with open(json_path) as f:
    data = json.load(f)

# Extract CIF text from OpenFold3 output
cif_text = data["outputs"][0]["structures_with_scores"][0]["structure"]

# Write full complex CIF
complex_cif_path = os.path.join(out_dir, "KSR_MEK_full.cif")
with open(complex_cif_path, "w") as f:
    f.write(cif_text)

# --- Split into protein and ligand CIFs ---

# Simple rule:
# - Protein: all lines that are not HETATM (plus header/loops)
# - Ligand: HETATM lines only (plus header so the file is still valid-ish CIF)

protein_lines = []
ligand_lines = []

for line in cif_text.splitlines():
    # CIF atom records often start with 'ATOM  ' or 'HETATM' in mmCIF-like exports
    # but there is header/loop/_atom_site lines that should go into both.
    # Here we duplicate non-atom lines into both files to keep minimal structure.
    if line.startswith("ATOM") or line.startswith("HETATM"):
        if line.startswith("HETATM"):
            ligand_lines.append(line)
        else:
            protein_lines.append(line)
    else:
        # Header/meta/loop definitions go to both
        protein_lines.append(line)
        ligand_lines.append(line)

# Write protein-only CIF    
protein_cif_path = os.path.join(out_dir, "KSR_MEK_protein.cif")
with open(protein_cif_path, "w") as f:
    f.write("\n".join(protein_lines))

# Write ligand-only CIF
ligand_cif_path = os.path.join(out_dir, "KSR_MEK_ligand.cif")
with open(ligand_cif_path, "w") as f:
    f.write("\n".join(ligand_lines))

print("Wrote:")
print("  Full complex:", complex_cif_path)
print("  Protein-only:", protein_cif_path)
print("  Ligand-only:", ligand_cif_path)
