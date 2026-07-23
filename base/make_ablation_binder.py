#!/usr/bin/env python3
"""Build a flip-through binder PPTX of the 24-task FTN ablation xfer matrices.

Fully config-driven: run directories are discovered under artifacts/ and every
decision (concept vs domain, replay on/off, reactivity, default, per-slide
deltas) is read from the recorded config values -- never from folder names.
"""
import ast
import glob
import os
import re

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ART = "/home/klmckee/Desktop/test_actl/functional_task_networks/base/artifacts"
OUT = os.path.join(ART, "ftn_24task_ablation_binder.pptx")

INK    = RGBColor(0x1a, 0x1a, 0x1a)
GREY   = RGBColor(0x55, 0x55, 0x55)
ACCENT = RGBColor(0xb3, 0x1b, 0x1b)

DOMAIN_KEYS = {
    "DOMAIN_SHIFT", "DOMAIN_DENSE_PROJECTION", "DOMAIN_PER_TASK_PROJECTION",
    "DOMAIN_SHIFT_FACTOR", "DOMAIN_SHIFT_STRIDE",
}

# =============================================================================
# Load configs
# =============================================================================


def parse_config(path):
    cfg = {}
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        try:
            cfg[k.strip()] = ast.literal_eval(v.strip())
        except Exception:
            cfg[k.strip()] = v.strip()
    return cfg


def config_header(path):
    with open(path) as f:
        first = f.readline().strip()
    return first.split("config:", 1)[1].strip() if "config:" in first else ""


runs = {}  # name -> {"cfg":dict, "header":str, "xfer":path}
for d in sorted(os.listdir(ART)):
    cfgs = glob.glob(os.path.join(ART, d, "config_*.txt"))
    if not cfgs:
        continue
    # Prefer the canonical single-epoch matrix (xfer_<name>.png); for a
    # multi-epoch run (xfer_<name>_ep<e>.png) fall back to the last epoch.
    name = os.path.basename(cfgs[0])[len("config_"):-len(".txt")]
    canonical = os.path.join(ART, d, f"xfer_{name}.png")
    if os.path.exists(canonical):
        xfer = canonical
    else:
        eps = glob.glob(os.path.join(ART, d, f"xfer_{name}_ep*.png"))
        eps.sort(key=lambda p: int(re.search(r"_ep(\d+)\.png$", p).group(1)))
        xfer = eps[-1] if eps else None
    if not xfer:
        print(f"  skip {d}: no xfer matrix yet (incomplete run)")
        continue
    runs[d] = {
        "cfg": parse_config(cfgs[0]),
        "header": config_header(cfgs[0]),
        "xfer": xfer,
    }


def pair_key(cfg):
    """Everything except the domain-shift axis -> pairs concept with domain."""
    return tuple(sorted((k, repr(v)) for k, v in cfg.items()
                        if k not in DOMAIN_KEYS))


groups = {}  # pair_key -> {"concept":name, "domain":name}
for name, r in runs.items():
    g = groups.setdefault(pair_key(r["cfg"]), {})
    g["domain" if r["cfg"].get("DOMAIN_SHIFT") else "concept"] = name

# =============================================================================
# Classify groups from config VALUES
# =============================================================================


def rep(cfg):
    """Representative concept-side attributes read straight from values."""
    return dict(
        reactive=bool(cfg.get("RECONFIG_REACTIVE")),
        sd=float(cfg.get("RECONFIG_REACTIVE_SD", 0.0)),
        every=int(cfg.get("RECONFIG_EVERY", 0)),
        replay=int(cfg.get("REPLAY_BATCH_SIZE", 0)) > 0,
    )


def group_cfg(g):
    name = g.get("concept") or g.get("domain")
    return runs[name]["cfg"]


DEFAULT_SD = max(rep(group_cfg(g))["sd"] for g in groups.values())  # low-reactivity


def is_default(g):
    a = rep(group_cfg(g))
    return a["replay"] and a["reactive"] and a["every"] == 0 and a["sd"] == DEFAULT_SD


default_key = next(k for k, g in groups.items() if is_default(g))
default_cfg = group_cfg(groups[default_key])


def deltas(cfg):
    """Config-driven diff of a group's config vs the default (non-domain keys)."""
    out = []
    for k in sorted(set(cfg) | set(default_cfg)):
        if k in DOMAIN_KEYS or k == "FTN_PARS":
            continue
        if cfg.get(k) != default_cfg.get(k):
            out.append((k, default_cfg.get(k), cfg.get(k)))
    return out


def friendly_title(cfg):
    a, d = rep(cfg), rep(default_cfg)
    parts = []
    if a["reactive"] and a["sd"] < d["sd"]:
        parts.append("Reactive_High")
    if a["reactive"] and a["sd"] > d["sd"]:
        parts.append("Reactive_Low")
    if not a["reactive"]:
        parts.append(f"Reconfig every {a['every']} step"
                     + ("s" if a["every"] != 1 else ""))
    if a["replay"] and not d["replay"]:
        parts.append("Replay")
    if not a["replay"] and d["replay"]:
        parts.append("No replay")
    return " + ".join(parts) if parts else "Default"


# Order: default first, then replay-on before replay-off, reactive before not,
# lower SD (higher reactivity) later.
def sort_key(item):
    k, g = item
    a = rep(group_cfg(g))
    return (k != default_key, not a["replay"], not a["reactive"], -a["sd"], a["every"])


ordered = sorted(groups.items(), key=sort_key)

# =============================================================================
# Slide construction
# =============================================================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW = prs.slide_width
blank = prs.slide_layouts[6]

# Matrices fill the vertical space, right-aligned, with only small margins.
# Two squares stacked: [top margin][label][gap][matrix][mid margin][label]
# [gap][matrix][bottom margin] = slide height.
MARGIN  = Inches(0.12)   # matrix-to-slide-edge margin (top/bottom/right)
LBL_H   = Inches(0.26)   # label band height
LBL_GAP = Inches(0.02)   # very small margin between label and its matrix
MID     = Inches(0.14)   # small margin between the two matrices
IMG     = Inches(3.28)   # square side; fills height given the margins above
IMG_X   = SW - MARGIN - IMG
CLBL_Y  = MARGIN
CIMG_Y  = CLBL_Y + LBL_H + LBL_GAP
DLBL_Y  = CIMG_Y + IMG + MID
DIMG_Y  = DLBL_Y + LBL_H + LBL_GAP
LEFT_X  = Inches(0.42)
LEFT_W  = IMG_X - LEFT_X - Inches(0.22)   # left column, clear of the matrices


def add_title(slide, text, sub=None):
    tb = slide.shapes.add_textbox(LEFT_X, Inches(0.18), LEFT_W, Inches(0.72))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = text
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = INK
    if sub:
        r2 = tf.add_paragraph().add_run(); r2.text = sub
        r2.font.size = Pt(13); r2.font.italic = True; r2.font.color.rgb = GREY


def _label(slide, text, lbl_y, color):
    lb = slide.shapes.add_textbox(IMG_X, lbl_y, IMG, LBL_H)
    lf = lb.text_frame
    lf.margin_top = 0; lf.margin_bottom = 0
    lf.vertical_anchor = MSO_ANCHOR.BOTTOM   # sit tight just above the matrix
    p = lf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = text
    r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = color


def add_matrix(slide, name, label, lbl_y, img_y):
    _label(slide, label, lbl_y, ACCENT)
    if name and runs[name]["xfer"]:
        slide.shapes.add_picture(runs[name]["xfer"], IMG_X, img_y, height=IMG)


def add_missing(slide, lbl_y, img_y):
    _label(slide, "DOMAIN SHIFT", lbl_y, GREY)
    box = slide.shapes.add_textbox(IMG_X, img_y, IMG, IMG)
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = "(no domain-shift\nrun for this config)"
    r.font.size = Pt(12); r.font.italic = True; r.font.color.rgb = GREY


def add_left(slide, blocks):
    tb = slide.shapes.add_textbox(LEFT_X, Inches(0.98), LEFT_W, Inches(6.4))
    tf = tb.text_frame; tf.word_wrap = True
    first = True
    for header, lines in blocks:
        if header is not None:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            if not first:
                p.space_before = Pt(8)
            r = p.add_run(); r.text = header
            r.font.size = Pt(13); r.font.bold = True; r.font.color.rgb = ACCENT
            first = False
        for ln in lines:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            indent = ln.startswith("  ")
            r = p.add_run(); r.text = ln.strip()
            r.font.size = Pt(10.5) if indent else Pt(11)
            r.font.color.rgb = GREY if indent else INK
            r.font.name = "Consolas"


def fmt(v):
    return repr(v) if isinstance(v, str) else str(v)


def add_config_slide(spec_key, g, is_first):
    cfg = group_cfg(g)
    slide = prs.slides.add_slide(blank)
    header = runs[g.get("concept") or g["domain"]]["header"]

    if is_first:
        add_title(slide, "Default configuration", f"{header}  —  reference values")
        c = default_cfg
        blocks = [
            ("DEFAULT (held constant)", [
                "Reactive_Low" if c["RECONFIG_REACTIVE"] else "Non-reactive",
                f"  RECONFIG_REACTIVE = {c['RECONFIG_REACTIVE']}",
                f"  RECONFIG_REACTIVE_SD = {c['RECONFIG_REACTIVE_SD']}",
                f"  RECONFIG_REACTIVE_EMA = {c['RECONFIG_REACTIVE_EMA']}",
                f"  RECONFIG_EVERY = {c['RECONFIG_EVERY']}",
                "Replay: ON" if rep(c)["replay"] else "Replay: OFF",
                f"  N_SAMPLE_PER_TASK = {c['N_SAMPLE_PER_TASK']}",
                f"  REPLAY_BATCH_SIZE = {c['REPLAY_BATCH_SIZE']}",
                f"  REPLAY_MAXLEN = {c['REPLAY_MAXLEN']}",
                f"Model: dim{c['SQUARE_DIM']}, {c['LAYERS']} layers, k{c['K']}",
                f"  N_TASK={c['N_TASK']}, DUR={c['TRAIN_DURATION']}, "
                f"LR={c['TRAIN_LR']}, SEED={c['SEED']}",
            ]),
            ("MATRIX ROWS", [
                "Top = concept shift (DOMAIN_SHIFT=False)",
                "Bottom = domain shift (DOMAIN_SHIFT=True)",
                "axes: after-train task x eval task",
            ]),
            ("LATER SLIDES", [
                "each lists only the config keys that",
                "differ from this default.",
            ]),
        ]
    else:
        add_title(slide, friendly_title(cfg), header)
        lines = []
        for k, old, new in deltas(cfg):
            lines.append(f"{k}:")
            lines.append(f"  {fmt(old)} -> {fmt(new)}")
        blocks = [("Δ FROM DEFAULT", lines or ["(none)"])]

    add_matrix(slide, g.get("concept"), "CONCEPT SHIFT", CLBL_Y, CIMG_Y)
    if g.get("domain"):
        add_matrix(slide, g["domain"], "DOMAIN SHIFT", DLBL_Y, DIMG_Y)
    else:
        add_missing(slide, DLBL_Y, DIMG_Y)
    add_left(slide, blocks)


for i, (k, g) in enumerate(ordered):
    add_config_slide(k, g, is_first=(i == 0))

prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
for i, (k, g) in enumerate(ordered):
    print(f"  slide {i+1}: {friendly_title(group_cfg(g)) if i else 'Default'} "
          f"| concept={g.get('concept')} domain={g.get('domain')}")
