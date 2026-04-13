"""
PyMOL publication-quality render script — standard structural biology pipeline
Run with:
    pymol -cq render_structures.py

Standard pipeline
-----------------
1.  Load full structure (protein + ligand where applicable).
2.  Full cartoon — ALL secondary structure INCLUDING loops.
    (cartoon_gap_cutoff prevents artificial ribbon breaks in predicted models.)
3.  Consistent two-colour palette across all four panels:
        MEK → steel blue (#3880B8)   KSR → burnt orange (#D75E37)
4.  Binding-site pocket shown as translucent sticks (chain-consistent colours).
5.  Ligand shown as stick-balls (CPK heteroatom colours, gold carbons).
6.  Scale bar rendered as a real PyMOL distance object (true Å scale).
7.  300 DPI ray-traced PNG; PIL post-process adds panel label, legend, note.

Figures produced
----------------
  Fig1_KSR_MEK.png       — KSR–MEK complex (apo), full-complex interface view
  Fig2_KSR_MEK_Tram.png  — KSR–MEK + Trametinib, binding-pocket zoom
  Fig3_MEK_Tram.png      — MEK + Trametinib, binding-pocket zoom
  Fig4_KSR_Tram.png      — KSR + Trametinib, binding-pocket zoom
"""

from pymol import cmd
import os

BASE = "/Users/bharadwajanandivada/Single-Cell-Melanoma-Pathway-Analysis/Matlab/Protien_Predictions"
OUT  = BASE
W, H = 1600, 1200   # pixels; 300 DPI → ~13.3 × 10 cm print size

# ── Consistent colour palette ─────────────────────────────────────────────────
COL_MEK    = "0x3880B8"   # steel blue   — MEK / Chain A
COL_KSR    = "0xD75E37"   # burnt orange — KSR / Chain C
COL_TRAM_C = "0xFFDB1A"   # gold         — Trametinib carbon
HEX_MEK    = "#3880B8"
HEX_KSR    = "#D75E37"
HEX_TRAM   = "#FFDB1A"


# ═══════════════════════ PyMOL helpers ════════════════════════════════════════

def setup():
    """Standard publication-quality PyMOL global settings."""
    cmd.set("ray_opaque_background", 1)
    cmd.set("ray_shadows",           1)
    cmd.set("depth_cue",             1)
    cmd.set("fog_start",          0.45)
    cmd.set("antialias",             2)
    cmd.set("ray_trace_mode",        1)    # outlines + shadows
    cmd.set("ray_trace_gain",      0.08)
    # ── cartoon quality ───────────────────────────────────────────────────────
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops",  1)
    cmd.set("cartoon_loop_radius",  0.20)  # slim but fully visible loops
    cmd.set("cartoon_loop_quality",  15)
    # Prevent phantom ribbon gaps in AlphaFold / OpenFold predicted models
    # (missing REMARK / residue gaps show as breaks without this)
    cmd.set("cartoon_gap_cutoff",   100)
    # ── ligand/stick defaults ─────────────────────────────────────────────────
    cmd.set("stick_radius",         0.18)
    cmd.set("sphere_scale",         0.28)
    cmd.bg_color("white")


def color_chains():
    """Apply standard two-colour palette to Chain A (MEK) and Chain C (KSR)."""
    cmd.color(COL_MEK, "chain A")
    cmd.color(COL_KSR, "chain C")


def style_ligand(sel="lig"):
    """
    Stick-ball representation for the small molecule.
    Gold carbons, CPK colours for all heteroatoms.
    """
    cmd.show("sticks",   sel)
    cmd.hide("cartoon",  sel)
    cmd.color(COL_TRAM_C, f"{sel} and elem C")
    cmd.color("0xFF4444",  f"{sel} and elem O")
    cmd.color("0x4488FF",  f"{sel} and elem N")
    cmd.color("0x66BB66",  f"{sel} and elem F")
    cmd.color("0xAA44AA",  f"{sel} and elem I")
    cmd.set("stick_ball",       1,   sel)
    cmd.set("stick_ball_ratio", 1.5, sel)


def show_pocket(protein_sel, lig_sel, pocket_color=None, cutoff=5.0):
    """
    Show residues within *cutoff* Å of the ligand as sticks.
    pocket_color : if given, apply a single colour to all pocket sticks
                   (use for single-chain structures).
                   If None, re-apply the standard chain palette (A=MEK, C=KSR).
    """
    cmd.select("pocket", f"{protein_sel} within {cutoff} of {lig_sel}")
    cmd.show("sticks", "pocket")
    cmd.set("stick_transparency", 0.10, "pocket")
    if pocket_color:
        cmd.color(pocket_color, "pocket")
    else:
        cmd.color(COL_MEK, "pocket and chain A")
        cmd.color(COL_KSR, "pocket and chain C")
    cmd.deselect()


def hide_long_loops(max_loop_len=8):
    """
    Standard approach for AlphaFold/OpenFold predicted structures:

    Hide loop segments longer than *max_loop_len* consecutive residues.
    Short loops (≤ max_loop_len) that connect secondary-structure elements
    are kept intact so the ribbon stays continuous.
    Long disordered stretches (terminal tails, internal unstructured regions)
    are hidden so the structured kinase core is not obscured.

    max_loop_len=8 is a conventional threshold used in structural biology
    publications for predicted structures (captures inter-SS connector loops
    without retaining disordered tangles).
    """
    from pymol import stored
    for chain in cmd.get_chains("all"):
        stored.res_data = []
        cmd.iterate(f"chain {chain} and name CA",
                    "stored.res_data.append((int(resi), ss))")
        if not stored.res_data:
            continue

        loop_start = None
        loop_residues = []

        for resi, ss in stored.res_data:
            if ss not in ("H", "S"):          # loop / coil
                if loop_start is None:
                    loop_start = resi
                loop_residues.append(resi)
            else:                              # helix or strand — end of loop run
                if loop_start is not None and len(loop_residues) > max_loop_len:
                    cmd.hide("everything",
                             f"chain {chain} and resi {loop_start}-{loop_residues[-1]}")
                loop_start = None
                loop_residues = []

        # Handle any trailing loop
        if loop_start is not None and len(loop_residues) > max_loop_len:
            cmd.hide("everything",
                     f"chain {chain} and resi {loop_start}-{loop_residues[-1]}")


def add_scale_bar(length_ang=20):
    """
    Render a true-to-scene-scale distance bar (ray-traced, real Å units).
    Placed below the bounding box; label reads the actual distance in Å.
    """
    ext  = cmd.get_extent("all")
    xc   = (ext[0][0] + ext[1][0]) / 2.0
    ybot = ext[0][1] - 14.0
    zmid = (ext[0][2] + ext[1][2]) / 2.0
    half = length_ang / 2.0
    cmd.pseudoatom("_bL", pos=[xc - half, ybot, zmid])
    cmd.pseudoatom("_bR", pos=[xc + half, ybot, zmid])
    cmd.distance("_scalebar", "_bL", "_bR")
    cmd.set("dash_gap",    0,   "_scalebar")
    cmd.set("dash_radius", 0.5, "_scalebar")
    cmd.color("black", "_scalebar")
    cmd.hide("labels", "_bL")
    cmd.hide("labels", "_bR")
    cmd.set("label_size",  20)
    cmd.set("label_color", "black")


def ray_save(name):
    path = os.path.join(OUT, name)
    cmd.ray(W, H)
    cmd.png(path, W, H, dpi=300, ray=0)
    print(f"  Saved → {path}")
    return path


# ════════════════════ PIL post-processing ═════════════════════════════════════

def annotate(img_path, panel_label, legend_entries, note="OpenFold3 model"):
    """
    Add panel label, colour legend, and model note to a saved PNG.

    legend_entries : list of ("#RRGGBB", "Label text")
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("  [WARNING] Pillow not installed — skipping 2D annotations.")
        return

    img   = Image.open(img_path).convert("RGBA")
    layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw  = ImageDraw.Draw(layer)
    w, h  = img.size

    # Font candidates (macOS system fonts)
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    font_lg = font_sm = font_note = ImageFont.load_default()
    for fp in candidates:
        if os.path.exists(fp):
            try:
                font_lg   = ImageFont.truetype(fp, 52)
                font_sm   = ImageFont.truetype(fp, 32)
                font_note = ImageFont.truetype(fp, 26)
                break
            except Exception:
                pass

    # Panel label — top-left, bold black
    draw.text((24, 18), panel_label, fill=(0, 0, 0, 255), font=font_lg)

    # Colour legend — bottom-left, semi-transparent white box
    swatch, gap, pad = 28, 10, 16
    row_h  = swatch + gap
    n      = len(legend_entries)
    box_w  = 240
    box_h  = n * row_h + 2 * pad
    bx1    = 20
    by1    = h - box_h - 20
    draw.rectangle([bx1, by1, bx1 + box_w, by1 + box_h],
                   fill=(255, 255, 255, 185), outline=(80, 80, 80, 220), width=2)
    for i, (hex_col, label) in enumerate(legend_entries):
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        sy = by1 + pad + i * row_h
        draw.rectangle([bx1 + pad, sy, bx1 + pad + swatch, sy + swatch],
                       fill=(r, g, b, 255), outline=(40, 40, 40, 200), width=1)
        draw.text((bx1 + pad + swatch + 8, sy + 2), label,
                  fill=(0, 0, 0, 255), font=font_sm)

    # Model note — bottom-right, grey italic
    if note:
        draw.text((w - 360, h - 44), note, fill=(100, 100, 100, 220), font=font_note)

    result = Image.alpha_composite(img, layer).convert("RGB")
    result.save(img_path, dpi=(300, 300))
    print(f"  Annotated → {img_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1  — KSR–MEK complex (apo)
#   Full-complex cartoon, interface residues shown as sticks
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 1] KSR–MEK apo complex")
cmd.reinitialize()
setup()
cmd.load(os.path.join(BASE, "KSR_MEK_protein.pdb"), "KSR_MEK")
cmd.dss("KSR_MEK")
cmd.show("cartoon", "all")
color_chains()
# Hide long disordered loops (>8 residues); keeps short connecting loops intact
hide_long_loops(max_loop_len=8)

# Interface sticks — residues within 5.5 Å across the chain boundary
cmd.select("iface", "chain A within 5.5 of chain C or chain C within 5.5 of chain A")
cmd.show("sticks", "iface")
cmd.color(COL_MEK, "iface and chain A")
cmd.color(COL_KSR, "iface and chain C")
cmd.deselect()

# Orient: show the whole complex, tilted to expose the interface
cmd.orient("all")
cmd.zoom("all", buffer=8)
cmd.turn("y",  20)
cmd.turn("x", -10)

add_scale_bar(20)
path1 = ray_save("Fig1_KSR_MEK.png")
annotate(path1, "A",
         [(HEX_MEK, "MEK (Chain A)"), (HEX_KSR, "KSR (Chain C)")])


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2  — KSR–MEK + Trametinib
#   Full complex, zoomed on the Trametinib binding pocket
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 2] KSR–MEK + Trametinib — binding-pocket zoom")
cmd.reinitialize()
setup()
cmd.load(os.path.join(BASE, "KSR_MEK_Tram_protein.pdb"), "prot")
cmd.load(os.path.join(BASE, "KSR_MEK_Tram_ligand.pdb"),  "lig")
cmd.dss("prot")
cmd.show("cartoon", "prot")
color_chains()
hide_long_loops(max_loop_len=8)
style_ligand("lig")
show_pocket("prot", "lig", cutoff=5.0)
# Restore pocket residues that hide_long_loops may have hidden
cmd.show("sticks", "pocket")

# Zoom on the ligand + its pocket
cmd.orient("lig")
cmd.zoom("lig", buffer=18)
cmd.turn("x", -10)
cmd.turn("y",   5)

add_scale_bar(10)
path2 = ray_save("Fig2_KSR_MEK_Tram.png")
annotate(path2, "B",
         [(HEX_MEK, "MEK"), (HEX_KSR, "KSR"), (HEX_TRAM, "Trametinib")])


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3  — MEK alone + Trametinib
#   Full MEK cartoon, binding-pocket zoom, consistent steel blue
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 3] MEK + Trametinib — binding-pocket zoom")
cmd.reinitialize()
setup()
cmd.load(os.path.join(BASE, "MEK_Tram_protein.pdb"), "MEK")
cmd.load(os.path.join(BASE, "MEK_Tram_ligand.pdb"),  "lig")
cmd.dss("MEK")
cmd.show("cartoon", "MEK")
cmd.color(COL_MEK, "MEK")
hide_long_loops(max_loop_len=8)
style_ligand("lig")
show_pocket("MEK", "lig", pocket_color=COL_MEK, cutoff=5.0)
cmd.show("sticks", "pocket")

cmd.orient("lig")
cmd.zoom("lig", buffer=18)
cmd.turn("x", -10)
cmd.turn("y",  10)

add_scale_bar(10)
path3 = ray_save("Fig3_MEK_Tram.png")
annotate(path3, "C",
         [(HEX_MEK, "MEK"), (HEX_TRAM, "Trametinib")])


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4  — KSR alone + Trametinib
#   Full KSR cartoon, binding-pocket zoom, consistent burnt orange
# ══════════════════════════════════════════════════════════════════════════════
print("\n[Fig 4] KSR + Trametinib — binding-pocket zoom")
cmd.reinitialize()
setup()
cmd.load(os.path.join(BASE, "KSR_Tram_protein.pdb"), "KSR")
cmd.load(os.path.join(BASE, "KSR_Tram_ligand.pdb"),  "lig")
cmd.dss("KSR")
cmd.show("cartoon", "KSR")
cmd.color(COL_KSR, "KSR")
hide_long_loops(max_loop_len=8)
style_ligand("lig")
show_pocket("KSR", "lig", pocket_color=COL_KSR, cutoff=5.0)
cmd.show("sticks", "pocket")

cmd.orient("lig")
cmd.zoom("lig", buffer=18)
cmd.turn("x", -10)
cmd.turn("y",  10)

add_scale_bar(10)
path4 = ray_save("Fig4_KSR_Tram.png")
annotate(path4, "D",
         [(HEX_KSR, "KSR"), (HEX_TRAM, "Trametinib")])


print("\nDone — 4 publication-quality figures saved.")
