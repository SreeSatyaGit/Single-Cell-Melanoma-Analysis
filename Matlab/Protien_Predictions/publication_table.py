"""
Generate a publication-quality binding energy table (PDF + PNG).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# Columns: Interaction, Type, Method, ΔG (kcal/mol), Kd / Score note, ICs
rows = [
    # Section A – Protein–Protein Interactions
    ("A", "KSR–MEK (apo)",           "Protein–protein", "PRODIGY",          "−3.40", "3.23 mM",   "224"),
    ("A", "KSR–MEK (Trametinib-bound)", "Protein–protein", "PRODIGY",       "−3.30", "3.82 mM",   "281"),
    # Section B – Small-Molecule Binding
    ("B", "MEK–Trametinib",          "Small molecule",  "AutoDock Vina 1.2.5", "−4.16", "—",      "—"),
    ("B", "KSR–Trametinib",          "Small molecule",  "AutoDock Vina 1.2.5", "−9.11", "—",      "—"),
    ("B", "KSR–MEK–Trametinib (ternary)", "Small molecule", "AutoDock Vina 1.2.5", "+0.40", "—",  "—"),
]

col_labels = [
    "Interaction",
    "Interaction\nType",
    "Scoring\nMethod",
    "ΔG\n(kcal mol⁻¹)",
    "Kd",
    "Interface\nContacts (ICs)",
]

# ── Layout ────────────────────────────────────────────────────────────────────
fig_w, fig_h = 11, 4.6
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.axis("off")

# Column x-positions and widths (normalised 0–1)
col_x     = [0.00, 0.22, 0.40, 0.57, 0.70, 0.84]
col_w     = [0.22, 0.18, 0.17, 0.13, 0.14, 0.16]
col_align = ["left", "left", "left", "center", "center", "center"]

# Row y-positions
HEADER_Y  = 0.91
ROW_H     = 0.115
SECTION_H = 0.04   # extra gap before section label row
N_DATA    = len(rows)
START_Y   = HEADER_Y - ROW_H      # y of first data row top edge

# Colours
C_HEADER   = "#1a3a5c"   # deep navy
C_SEC_A    = "#2c6e8a"   # teal-blue  (PPI)
C_SEC_B    = "#3d7a4e"   # forest green (small molecule)
C_ROW_ODD  = "#f0f4f8"
C_ROW_EVEN = "#ffffff"
C_TEXT     = "#1c1c1c"
C_HEADER_T = "#ffffff"
C_SEC_T    = "#ffffff"
C_POS      = "#b22222"   # red for positive (unfavourable) ΔG
C_NEG      = "#1a5276"   # navy for negative ΔG

FONT      = "DejaVu Sans"
FS_HDR    = 8.5
FS_SEC    = 8.2
FS_DATA   = 8.0

def draw_cell(ax, x, y, w, h, text, facecolor, textcolor, fontsize,
              halign="left", bold=False, italic=False):
    rect = FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="square,pad=0",
        linewidth=0, facecolor=facecolor, clip_on=False,
        transform=ax.transAxes
    )
    ax.add_patch(rect)
    pad = 0.008 if halign == "left" else 0.0
    tx = x + pad if halign == "left" else x + w / 2
    ty = y - h / 2
    fw = "bold" if bold else "normal"
    fs = "italic" if italic else "normal"
    ax.text(tx, ty, text,
            ha=halign, va="center",
            fontsize=fontsize, fontfamily=FONT,
            fontweight=fw, fontstyle=fs,
            color=textcolor, clip_on=False,
            transform=ax.transAxes)

# ── Header row ────────────────────────────────────────────────────────────────
for i, (label, cx, cw, ca) in enumerate(zip(col_labels, col_x, col_w, col_align)):
    draw_cell(ax, cx, HEADER_Y, cw, ROW_H,
              label, C_HEADER, C_HEADER_T, FS_HDR,
              halign=ca, bold=True)

# ── Section A header ──────────────────────────────────────────────────────────
sec_a_y = HEADER_Y - ROW_H
draw_cell(ax, 0.0, sec_a_y, 1.0, SECTION_H,
          "  A   Protein–Protein Interactions", C_SEC_A, C_SEC_T, FS_SEC,
          halign="left", bold=True, italic=False)

# ── Data rows ─────────────────────────────────────────────────────────────────
cur_y = sec_a_y - SECTION_H
section_b_inserted = False

for ridx, row in enumerate(rows):
    sec, interaction, int_type, method, dg, kd, ics = row

    # Insert section B header before first B row
    if sec == "B" and not section_b_inserted:
        draw_cell(ax, 0.0, cur_y, 1.0, SECTION_H,
                  "  B   Small-Molecule Binding",
                  C_SEC_B, C_SEC_T, FS_SEC,
                  halign="left", bold=True)
        cur_y -= SECTION_H
        section_b_inserted = True

    bg = C_ROW_ODD if ridx % 2 == 0 else C_ROW_EVEN
    cells = [interaction, int_type, method, dg, kd, ics]

    for ci, (text, cx, cw, ca) in enumerate(zip(cells, col_x, col_w, col_align)):
        # Colour ΔG value based on sign
        if ci == 3:
            tc = C_POS if text.startswith("+") else C_NEG
            fw_bold = True
        else:
            tc = C_TEXT
            fw_bold = False
        draw_cell(ax, cx, cur_y, cw, ROW_H,
                  text, bg, tc, FS_DATA,
                  halign=ca, bold=fw_bold)

    cur_y -= ROW_H

# ── Bottom border line ────────────────────────────────────────────────────────
ax.plot([0, 1], [cur_y, cur_y],
        color="#1a3a5c", linewidth=1.2, transform=ax.transAxes, clip_on=False)

# ── Top border line ───────────────────────────────────────────────────────────
ax.plot([0, 1], [HEADER_Y, HEADER_Y],
        color="#1a3a5c", linewidth=1.2, transform=ax.transAxes, clip_on=False)

# ── Column dividers (subtle) ─────────────────────────────────────────────────
for cx in col_x[1:]:
    ax.plot([cx, cx], [cur_y, HEADER_Y],
            color="#cccccc", linewidth=0.5, transform=ax.transAxes, clip_on=False)

# ── Title & footnotes ─────────────────────────────────────────────────────────
ax.text(0.5, HEADER_Y + 0.055,
        "Table 1.  Computed Binding Energies for KSR–MEK and Trametinib Interactions",
        ha="center", va="bottom", fontsize=10, fontfamily=FONT,
        fontweight="bold", color=C_HEADER, transform=ax.transAxes)

footnote = (
    "ΔG, Gibbs free energy of binding.  Kd, equilibrium dissociation constant.  "
    "ICs, interfacial contacts (5.5 Å heavy-atom cutoff).\n"
    "PPI affinities computed with the PRODIGY empirical model (Vangone & Bonvin, eLife 2015; Xue et al., Proteins 2016).\n"
    "Small-molecule scores computed with AutoDock Vina 1.2.5 using local optimisation of OpenFold3-predicted poses "
    "(Trott & Olson, J Comp Chem 2010; Eberhardt et al., J Chem Inf Model 2021).\n"
    "Positive ΔG (red) indicates an energetically unfavourable pose after local minimisation."
)
ax.text(0.0, cur_y - 0.03, footnote,
        ha="left", va="top", fontsize=6.5, fontfamily=FONT,
        color="#444444", transform=ax.transAxes,
        linespacing=1.6)

# ── Colour legend patches ─────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=C_SEC_A, label="Protein–Protein Interactions (PRODIGY)"),
    mpatches.Patch(facecolor=C_SEC_B, label="Small-Molecule Binding (AutoDock Vina)"),
    mpatches.Patch(facecolor=C_NEG,   label="Favourable ΔG (negative)"),
    mpatches.Patch(facecolor=C_POS,   label="Unfavourable ΔG (positive)"),
]
ax.legend(handles=legend_patches,
          loc="upper right", bbox_to_anchor=(1.0, HEADER_Y - 0.01),
          fontsize=6.5, framealpha=0.0, handlelength=1.2, handleheight=1.0,
          labelcolor="#333333")

fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)

base = "/Users/bharadwajanandivada/Single-Cell-Melanoma-Pathway-Analysis/Matlab/Protien_Predictions"
pdf_path = f"{base}/Table1_BindingEnergies.pdf"
png_path = f"{base}/Table1_BindingEnergies.png"

fig.savefig(pdf_path, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved:\n  {pdf_path}\n  {png_path}")
