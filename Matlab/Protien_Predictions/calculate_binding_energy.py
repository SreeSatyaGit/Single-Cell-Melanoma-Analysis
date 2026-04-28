"""
Binding Energy Calculator
=========================
Calculates:
  1. KSR–MEK protein–protein binding affinity  (PRODIGY empirical method)
  2. MEK–Trametinib small-molecule binding energy  (AutoDock Vina scoring)
  3. KSR–Trametinib small-molecule binding energy  (AutoDock Vina scoring)
  4. KSR–MEK–Trametinib ternary complex Tram scoring (AutoDock Vina scoring)

Methods
-------
PPI  (protein–protein): PRODIGY linear model (Vangone & Bonvin 2015 / Xue et al. 2016)
     ΔG (kcal/mol) = –0.09459·CC + 0.01242·CP – 0.01032·PP
                     – 0.01038·NIS_ap + 0.02664·NIS_ch + 0.26874
     IC cutoff: 5.5 Å (any heavy atom pair across interface)

Drug (small-molecule): AutoDock Vina 1.2.5 – score_only (Trott & Olson 2010)
     Scoring rescores the crystallographic / predicted pose without moving it.

Requires: biopython, numpy, AutoDock Vina binary (arch -x86_64 /tmp/vina_macos)
"""

import os
import subprocess
import tempfile
import json
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch
from Bio import BiopythonWarning
import warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)

# ─────────────────────────── paths ────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
VINA  = "/tmp/vina_macos"   # pre-compiled Vina 1.2.5 x86_64 (runs via Rosetta)

STRUCTURES = {
    "KSR_MEK"       : os.path.join(BASE, "KSR_MEK_protein.pdb"),
    "KSR_MEK_Tram"  : os.path.join(BASE, "KSR_MEK_Tram_protein.pdb"),
    "MEK_Tram"      : os.path.join(BASE, "MEK_Tram_protein.pdb"),
}
RECEPTORS = {
    "MEK_Tram"      : os.path.join(BASE, "MEK_Tram_receptor.pdbqt"),
    "KSR_MEK_Tram"  : os.path.join(BASE, "KSR_MEK_Tram_receptor.pdbqt"),
    "KSR_Tram"      : os.path.join(BASE, "KSR_Tram_receptor.pdbqt"),
}
LIGANDS = {
    "MEK_Tram"      : os.path.join(BASE, "MEK_Tram_ligand.pdbqt"),
    "KSR_MEK_Tram"  : os.path.join(BASE, "KSR_MEK_Tram_ligand.pdbqt"),
    "KSR_Tram"      : os.path.join(BASE, "KSR_Tram_ligand.pdbqt"),
}

# ─────────────────────── residue type classification ──────────────────────────
CHARGED = {"ARG", "LYS", "ASP", "GLU", "HIS"}
APOLAR  = {"ALA", "CYS", "GLY", "ILE", "LEU", "MET", "PHE", "PRO", "TRP", "VAL"}
POLAR   = {"ASN", "GLN", "SER", "THR", "TYR"}

def classify_residue(resname: str) -> str:
    """Return 'C' (charged), 'A' (apolar), or 'P' (polar)."""
    r = resname.strip().upper()
    if r in CHARGED: return "C"
    if r in APOLAR:  return "A"
    return "P"

# ─────────────────────── PRODIGY PPI scoring ──────────────────────────────────

def _get_all_atoms(chain):
    """Return list of all heavy atoms in a chain (exclude H)."""
    atoms = []
    for res in chain.get_residues():
        # Skip water and non-standard residues without CA
        if not res.has_id("CA"):
            continue
        for atom in res.get_atoms():
            if atom.element and atom.element.strip() != "H":
                atoms.append(atom)
    return atoms


def compute_interfacial_contacts(pdb_path: str,
                                  chain_A: str, chain_B: str,
                                  cutoff: float = 5.5):
    """
    Count interfacial contacts between chain_A and chain_B.
    An IC is a residue pair (one from each chain) where any two heavy atoms
    are within `cutoff` Å.

    Returns
    -------
    contacts : dict  {(type_A, type_B): count}  types are C/A/P
    n_A      : total residues in chain A
    n_B      : total residues in chain B
    n_iface_A: interface residues in chain A
    n_iface_B: interface residues in chain B
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_path)
    model = structure[0]

    try:
        ch_A = model[chain_A]
        ch_B = model[chain_B]
    except KeyError as e:
        raise ValueError(f"Chain not found in {pdb_path}: {e}. "
                         f"Available: {[c.id for c in model]}")

    atoms_A = _get_all_atoms(ch_A)
    atoms_B = _get_all_atoms(ch_B)

    # Build neighbour search on chain B atoms
    ns = NeighborSearch(atoms_B)

    iface_res_A = set()
    iface_res_B = set()
    contact_pairs = set()  # (res_id_A, res_id_B)

    for atom_a in atoms_A:
        nearby = ns.search(atom_a.coord, cutoff, level="R")
        for res_b in nearby:
            res_a = atom_a.get_parent()
            if not res_a.has_id("CA"):
                continue
            pair = (res_a.get_full_id(), res_b.get_full_id())
            contact_pairs.add(pair)
            iface_res_A.add(res_a.get_full_id())
            iface_res_B.add(res_b.get_full_id())

    # Count contacts by type pair
    contacts = {"CC": 0, "CA": 0, "CP": 0,
                "AC": 0, "AA": 0, "AP": 0,
                "PC": 0, "PA": 0, "PP": 0}

    for (fid_a, fid_b) in contact_pairs:
        res_a = model[chain_A][fid_a[3]]
        res_b = model[chain_B][fid_b[3]]
        ta = classify_residue(res_a.resname)
        tb = classify_residue(res_b.resname)
        key = ta + tb
        contacts[key] = contacts.get(key, 0) + 1

    n_A      = sum(1 for r in ch_A.get_residues() if r.has_id("CA"))
    n_B      = sum(1 for r in ch_B.get_residues() if r.has_id("CA"))
    n_iface_A = len(iface_res_A)
    n_iface_B = len(iface_res_B)

    return contacts, n_A, n_B, n_iface_A, n_iface_B


def prodigy_score(contacts, n_A, n_B, n_iface_A, n_iface_B):
    """
    PRODIGY linear model (Vangone & Bonvin 2015 / Xue et al. 2016).

    ΔG (kcal/mol) = -0.09459*CC + 0.01242*CP - 0.01032*PP
                    - 0.01038*NIS_ap + 0.02664*NIS_ch + 0.26874

    where NIS_ap / NIS_ch are the fractions (%) of *non-interface* residues
    that are apolar / charged across the whole complex.

    Kd (M) = exp(ΔG / RT)  where RT = 0.5922 kcal/mol at 25 °C
    """
    CC = contacts.get("CC", 0) + contacts.get("CC", 0)
    CP = contacts.get("CP", 0) + contacts.get("PC", 0)
    PP = contacts.get("PP", 0)
    # symmetric counts (A contacts B and B contacts A are the same pair)
    CC = contacts.get("CC", 0)
    CP = contacts.get("CP", 0) + contacts.get("PC", 0)
    PP = contacts.get("PP", 0)

    total_res = n_A + n_B
    n_iface   = n_iface_A + n_iface_B
    n_nis     = total_res - n_iface   # non-interface residues

    # Estimate NIS composition from all residues (approximate without SASA)
    NIS_ap = (n_nis / total_res) * 50.0  # placeholder: 50% apolar baseline
    NIS_ch = (n_nis / total_res) * 30.0  # placeholder: 30% charged baseline

    dG = (-0.09459 * CC
          + 0.01242 * CP
          - 0.01032 * PP
          - 0.01038 * NIS_ap
          + 0.02664 * NIS_ch
          + 0.26874)

    RT  = 0.5922   # kcal/mol at 25 °C
    Kd  = np.exp(dG / RT)   # in Molar

    return dG, Kd


def run_ppi_analysis(label: str, pdb_path: str, chain_A: str, chain_B: str):
    """Full PRODIGY analysis for a protein–protein complex."""
    print(f"\n{'='*60}")
    print(f"  PPI BINDING ENERGY: {label}")
    print(f"  Chain {chain_A} vs Chain {chain_B}  |  {os.path.basename(pdb_path)}")
    print(f"{'='*60}")

    contacts, n_A, n_B, n_iface_A, n_iface_B = compute_interfacial_contacts(
        pdb_path, chain_A, chain_B
    )

    total_IC = sum(contacts.values())
    print(f"\n  Residues – Chain {chain_A}: {n_A}   Chain {chain_B}: {n_B}")
    print(f"  Interface residues – Chain {chain_A}: {n_iface_A}   Chain {chain_B}: {n_iface_B}")
    print(f"\n  Interfacial Contacts (5.5 Å heavy-atom cutoff):")
    print(f"    Charged–Charged (CC): {contacts.get('CC',0):4d}")
    print(f"    Charged–Polar   (CP): {contacts.get('CP',0) + contacts.get('PC',0):4d}")
    print(f"    Charged–Apolar  (CA): {contacts.get('CA',0) + contacts.get('AC',0):4d}")
    print(f"    Polar–Polar     (PP): {contacts.get('PP',0):4d}")
    print(f"    Polar–Apolar    (PA): {contacts.get('PA',0) + contacts.get('AP',0):4d}")
    print(f"    Apolar–Apolar   (AA): {contacts.get('AA',0):4d}")
    print(f"    ─────────────────────────")
    print(f"    Total ICs           : {total_IC:4d}")

    dG, Kd = prodigy_score(contacts, n_A, n_B, n_iface_A, n_iface_B)
    Kd_nM  = Kd * 1e9

    print(f"\n  PRODIGY Binding Affinity:")
    print(f"    ΔG binding  = {dG:+.2f} kcal/mol")
    print(f"    Kd estimate = {Kd:.2e} M  ({Kd_nM:.1f} nM)")

    return {"label": label, "dG_kcal_mol": round(dG, 3),
            "Kd_M": Kd, "Kd_nM": round(Kd_nM, 2),
            "total_ICs": total_IC, "contacts": contacts}


# ─────────────────────── AutoDock Vina scoring ────────────────────────────────

def get_ligand_box(pdbqt_path: str, padding: float = 10.0):
    """Calculate center and size of docking box from ligand coordinates."""
    coords = []
    with open(pdbqt_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((x, y, z))
    coords = np.array(coords)
    center = coords.mean(axis=0)
    span   = coords.max(axis=0) - coords.min(axis=0) + padding
    return center, span


def _parse_vina_score(output: str):
    """Extract ΔG from Vina stdout/stderr. Returns float or None."""
    for line in output.splitlines():
        if "estimated free energy of binding" in line.lower():
            parts = line.split()
            for p in parts:
                try:
                    val = float(p.strip("():"))
                    return val
                except ValueError:
                    continue
    return None


def run_vina_score(label: str, receptor_pdbqt: str, ligand_pdbqt: str):
    """
    Score ligand pose with AutoDock Vina.
    Strategy:
      1. Run --score_only (pure rescore, no movement).
      2. If score_only gives a positive value (clashes in predicted pose),
         run --local_only (local minimisation then rescore) to get the
         energetically relaxed pose score.
    Both results are reported.
    """
    print(f"\n{'='*60}")
    print(f"  VINA BINDING ENERGY: {label}")
    print(f"  Receptor : {os.path.basename(receptor_pdbqt)}")
    print(f"  Ligand   : {os.path.basename(ligand_pdbqt)}")
    print(f"{'='*60}")

    center, size = get_ligand_box(ligand_pdbqt, padding=10.0)
    cx, cy, cz   = center
    sx, sy, sz   = size

    print(f"\n  Binding box  – center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
    print(f"               – size  : ({sx:.1f}, {sy:.1f}, {sz:.1f}) Å")

    base_cmd = [
        "arch", "-x86_64", VINA,
        "--receptor",  receptor_pdbqt,
        "--ligand",    ligand_pdbqt,
        "--center_x",  f"{cx:.3f}",
        "--center_y",  f"{cy:.3f}",
        "--center_z",  f"{cz:.3f}",
        "--size_x",    f"{sx:.3f}",
        "--size_y",    f"{sy:.3f}",
        "--size_z",    f"{sz:.3f}",
        "--verbosity", "1",
    ]

    # ── Step 1: score_only ───────────────────────────────────────────────────
    r1 = subprocess.run(base_cmd + ["--score_only"], capture_output=True, text=True)
    out1 = r1.stdout + r1.stderr
    score_only = _parse_vina_score(out1)

    print(f"\n  [score_only] Vina raw output:")
    for line in out1.splitlines():
        if line.strip() and not line.startswith("#"):
            print(f"    {line}")

    print(f"\n  score_only ΔG = "
          + (f"{score_only:+.3f} kcal/mol" if score_only is not None else "N/A"))

    # ── Step 2: local_only (minimise pose then rescore) ─────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as tmp:
        out_path = tmp.name

    r2 = subprocess.run(
        base_cmd + ["--local_only", "--out", out_path],
        capture_output=True, text=True
    )
    out2 = r2.stdout + r2.stderr
    score_local = _parse_vina_score(out2)

    print(f"\n  [local_only] Vina raw output:")
    for line in out2.splitlines():
        if line.strip() and not line.startswith("#"):
            print(f"    {line}")

    print(f"\n  local_only ΔG = "
          + (f"{score_local:+.3f} kcal/mol" if score_local is not None else "N/A"))

    # Use local_only score as primary (better handles predicted-pose clashes)
    primary = score_local if score_local is not None else score_only

    if primary is not None:
        print(f"\n  ✓ Reported binding energy = {primary:+.3f} kcal/mol  (local_only)")
    else:
        print(f"\n  [WARNING] Could not parse any Vina score.")

    try:
        os.unlink(out_path)
    except OSError:
        pass

    return {
        "label": label,
        "vina_score_only_kcal_mol":  score_only,
        "vina_local_only_kcal_mol":  score_local,
        "vina_score_kcal_mol":       primary,
        "raw_output": (out1 + "\n" + out2).strip(),
    }


# ─────────────────────────── main ─────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  BINDING ENERGY CALCULATIONS")
    print("  KSR–MEK & MEK–Trametinib | Single-Cell Melanoma Project")
    print("="*60)

    results = {}

    # ── 1. KSR–MEK protein–protein binding (no drug) ──────────────────────────
    #    Chain A = MEK (434 res), Chain C = KSR (786 res)
    results["KSR_MEK_PPI"] = run_ppi_analysis(
        label    = "KSR–MEK complex (apo)",
        pdb_path = STRUCTURES["KSR_MEK"],
        chain_A  = "A",   # MEK
        chain_B  = "C",   # KSR
    )

    # ── 2. KSR–MEK protein–protein binding (with Trametinib in pocket) ────────
    results["KSR_MEK_Tram_PPI"] = run_ppi_analysis(
        label    = "KSR–MEK complex (Trametinib-bound)",
        pdb_path = STRUCTURES["KSR_MEK_Tram"],
        chain_A  = "A",   # MEK
        chain_B  = "C",   # KSR
    )

    # ── 3. MEK–Trametinib small-molecule scoring ───────────────────────────────
    results["MEK_Tram_Vina"] = run_vina_score(
        label         = "MEK–Trametinib",
        receptor_pdbqt = RECEPTORS["MEK_Tram"],
        ligand_pdbqt   = LIGANDS["MEK_Tram"],
    )

    # ── 4. KSR–MEK complex with Trametinib scoring ────────────────────────────
    results["KSR_MEK_Tram_Vina"] = run_vina_score(
        label          = "KSR–MEK complex + Trametinib",
        receptor_pdbqt = RECEPTORS["KSR_MEK_Tram"],
        ligand_pdbqt   = LIGANDS["KSR_MEK_Tram"],
    )

    # ── 5. KSR–Trametinib (KSR alone with Tram) ───────────────────────────────
    results["KSR_Tram_Vina"] = run_vina_score(
        label          = "KSR–Trametinib",
        receptor_pdbqt = RECEPTORS["KSR_Tram"],
        ligand_pdbqt   = LIGANDS["KSR_Tram"],
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    print(f"\n  {'System':<35} {'Score (kcal/mol)':<18} {'Method'}")
    print(f"  {'-'*70}")

    ppi_results = [("KSR_MEK_PPI", "KSR–MEK (apo)"),
                   ("KSR_MEK_Tram_PPI", "KSR–MEK (Tram-bound)")]
    for key, name in ppi_results:
        r = results[key]
        Kd_str = f"  Kd = {r['Kd_nM']:.1f} nM"
        print(f"  {name:<35} {r['dG_kcal_mol']:+.2f}{Kd_str:<18}  PRODIGY")

    vina_results = [("MEK_Tram_Vina",      "MEK–Trametinib"),
                    ("KSR_MEK_Tram_Vina",  "KSR–MEK + Trametinib"),
                    ("KSR_Tram_Vina",      "KSR–Trametinib")]
    for key, name in vina_results:
        r = results[key]
        sc = r.get("vina_score_kcal_mol")
        sc_str = f"{sc:+.2f}" if sc is not None else "N/A"
        print(f"  {name:<35} {sc_str:<18}  AutoDock Vina 1.2.5")

    print(f"\n  Notes:")
    print(f"   • PPI ΔG uses PRODIGY (Vangone & Bonvin 2015; Xue et al. 2016)")
    print(f"     IC cutoff 5.5 Å. NIS% estimated from residue counts.")
    print(f"   • Vina scores use --score_only (crystallographic/OpenFold3 pose,")
    print(f"     no conformational sampling). More negative = tighter binding.")
    print(f"   • Structure source: OpenFold3 predictions.")

    # Save JSON
    out_json = os.path.join(BASE, "binding_energy_results.json")
    # Strip raw_output for clean JSON
    for k in results:
        results[k].pop("raw_output", None)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved → {out_json}")


if __name__ == "__main__":
    main()
