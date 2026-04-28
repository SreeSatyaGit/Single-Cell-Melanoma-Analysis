import numpy as np

coords = []
with open("MEK_Tram_ligand.pdbqt") as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))

coords = np.array(coords)
cx, cy, cz = coords.mean(axis=0)
print("CENTER:", cx, cy, cz)
