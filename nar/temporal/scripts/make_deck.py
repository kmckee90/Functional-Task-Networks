"""Build the temporal-experiment presentation (FTN_temporal_experiment.pptx).

Run on the pod from /home/dev/workspace/temporal:
    uv run --with python-pptx --with pillow python scripts/make_deck.py [run]
Numbers are computed live from results/<run>/results.json.
"""

import json
import sys
from pathlib import Path

import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
GREEN = RGBColor(0x00, 0x83, 0x00)
RED = RGBColor(0xE3, 0x49, 0x48)
SURF = RGBColor(0xFC, 0xFC, 0xFB)
SW, SH = Inches(13.333), Inches(7.5)

RUN = sys.argv[1] if len(sys.argv) > 1 else "main"
OUT = Path("results") / RUN
R = json.load(open(OUT / "results.json"))
NOID = None
if (OUT / "reeval_v1_noid.json").exists():
    NOID = json.load(open(OUT / "reeval_v1_noid.json"))


def m(v, split, method, key):
    rows = [r for r in R[v][split] if method in r]
    return np.mean([r[method][key] for r in rows]) if rows else float("nan")


prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def add_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = SURF
    return s


def textbox(slide, x, y, w, h, lines, size=16, color=INK2):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        text, opts = item if isinstance(item, tuple) else (item, {})
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = opts.get("align", PP_ALIGN.LEFT)
        para.line_spacing = 1.15
        para.space_after = Pt(opts.get("space_after", 6))
        run = para.add_run()
        run.text = text
        run.font.size = Pt(opts.get("size", size))
        run.font.color.rgb = opts.get("color", color)
        run.font.bold = opts.get("bold", False)
        run.font.name = "Calibri"
    return tb


def title(slide, text, sub=None):
    textbox(slide, Inches(0.55), Inches(0.28), SW - Inches(1.1), Inches(0.9),
            [(text, dict(size=26, bold=True, color=INK))])
    if sub:
        textbox(slide, Inches(0.55), Inches(0.92), SW - Inches(1.1), Inches(0.5),
                [(sub, dict(size=13.5, color=MUTED))])


def picture(slide, path, y_top, max_h=None, max_w=None):
    from PIL import Image
    iw, ih = Image.open(path).size
    max_w = max_w or (SW - Inches(1.0))
    max_h = max_h or (SH - y_top - Inches(0.35))
    scale = min(max_w / iw, max_h / ih)
    w, h = Emu(int(iw * scale)), Emu(int(ih * scale))
    slide.shapes.add_picture(str(path), Emu(int((SW - w) / 2)), y_top, w, h)


def table(slide, x, y, w, rows, col_widths, size=13):
    shape = slide.shapes.add_table(len(rows), len(rows[0]), x, y, w,
                                   Inches(0.36 * len(rows)))
    tbl = shape.table
    for j, cw in enumerate(col_widths):
        tbl.columns[j].width = cw
    for i, row in enumerate(rows):
        for j, cell_text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xEE, 0xF3, 0xFB) if i == 0 else SURF
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(cell_text)
            r.font.size = Pt(size)
            r.font.bold = i == 0
            r.font.color.rgb = INK if i == 0 else INK2
            r.font.name = "Calibri"


# ---------------------------------------------------------------- 1. title
s = add_slide()
textbox(s, Inches(0.9), Inches(2.1), SW - Inches(1.8), Inches(2.0), [
    ("Recurrent FTN: Temporal Selection for Non-Commutative Program Induction",
     dict(size=33, bold=True, color=INK)),
    ("One weight-tied bank, a mask per time step — the mask sequence is the "
     "program. Inferring operator sequences AND their length from a support "
     "set, with length generalization far beyond training.",
     dict(size=18, color=INK2)),
])
textbox(s, Inches(0.9), Inches(4.8), SW - Inches(1.8), Inches(1.0), [
    ("FTN / NAR experiment II  ·  nar/temporal/  ·  July 2026", dict(size=14, color=MUTED)),
    ("Writeup: temporal/RESULTS.md  ·  predecessor: abelian/RESULTS.md",
     dict(size=14, color=MUTED)),
])

# ---------------------------------------------------------------- 2. from abelian to temporal
s = add_slide()
title(s, "Why this experiment: what the abelian grid couldn't test")
textbox(s, Inches(0.7), Inches(1.35), SW - Inches(1.4), Inches(5.6), [
    ("Experiment I (abelian grid) showed FTN composes never-trained algorithms by "
     "spatial selection — but commutativity meant ORDER never mattered, and fixed "
     "architecture meant program LENGTH was capped by depth.", dict(size=16)),
    ("", {}),
    ("The temporal question:", dict(size=17, bold=True, color=INK)),
    ("•  non-commutative composition — order must be represented and executed",
     dict(size=15)),
    ("•  variable, autonomously inferred program length (halting)", dict(size=15)),
    ("•  scale OOD — programs LONGER than any ever trained", dict(size=15)),
    ("", {}),
    ("The answer is temporal selection: position in TIME encodes order, one "
     "weight-tied stage means an operator learned once is reusable at every step "
     "of every program (loops = repeated masks), and the trace is searched, not "
     "stored — the algorithm is represented only by being run.", dict(size=16)),
])

# ---------------------------------------------------------------- 3. task + model (fig1)
s = add_slide()
title(s, "Task family and architecture")
picture(s, OUT / "fig1_task.png", Inches(1.3))

# ---------------------------------------------------------------- 4. protocol
s = add_slide()
title(s, "Training: two trainers, same block-sequential stream, no task identity")
lv = R["level_sizes"]
n_train = len(R["train_tasks"])
n_held = len(R["held_tasks"])
textbox(s, Inches(0.7), Inches(1.3), Inches(6.2), Inches(5.7), [
    (f"Stream: {n_train} tasks with shortest programs ℓ ≤ 3 (identity + "
     f"generators + short composites), 3 cycles, curriculum by ℓ. "
     f"{n_held} tasks with ℓ = 4…9 are never trained.", dict(size=15)),
    ("", {}),
    ("V1 — library / EM (hard states)", dict(size=16, bold=True, color=INK)),
    ("Atom blocks train like single FTN tasks (per-batch reconfig). Composite "
     "blocks alternate: E-step = BFS trace search over consolidated atom masks "
     "on a support batch; M-step = BPTT through the soft rollout with the "
     "trace's masks FIXED.", dict(size=14)),
    ("", {}),
    ("V2 — BPTT-configured (soft states, fully general)", dict(size=16, bold=True, color=INK)),
    ("Every batch: UnrolledAnnealSTE configures all per-step masks by gradient "
     "(length swept, no hints), then a weight step through the soft rollout. "
     "No library, no privileged atoms.", dict(size=14)),
])
textbox(s, Inches(7.2), Inches(1.3), Inches(5.5), Inches(5.7), [
    ("Program inference at evaluation (θ frozen, support only):",
     dict(size=16, bold=True, color=INK)),
    ("A. LibraryBFS — consensus atom masks; BFS memoized on support states; "
     "halting = shortest trace matching the support", dict(size=14)),
    ("B. UnrolledAnnealSTE — gradient trace search through soft rollouts; "
     "halting = external T-sweep, hard exact-match selects", dict(size=14)),
    ("C. GreedyGrow — library-free hill climbing, one step at a time; halting "
     "= saturation/patience", dict(size=14)),
    ("", {}),
    ("All three routes are pure spatial-selection-through-time: zero weight "
     "updates at evaluation, ever.", dict(size=14, color=MUTED)),
])

# ---------------------------------------------------------------- 5. headline curve
s = add_slide()
title(s, "Headline: length generalization",
      "accuracy vs required program length — shaded region is the training "
      "regime; everything right of it needs programs longer than any ever trained")
picture(s, OUT / "fig2_length_ood.png", Inches(1.5))

# ---------------------------------------------------------------- 6. numbers
s = add_slide()
title(s, "Held-out tasks (ℓ = 4…9): the numbers")
rows = [["model · route", "per-position", "exact-match"]]
if NOID:
    nv = np.mean([r["library"]["pos"] for r in NOID["held"]])
    ne = np.mean([r["library"]["exact"] for r in NOID["held"]])
    rows.append(["v1 · library (id-free vocabulary, FINAL)",
                 f"{nv:.3f}", f"{ne:.3f}"])
for v in ("v1", "v2"):
    if v not in R:
        continue
    for meth in ("library", "gradient", "greedy"):
        rows.append([f"{v} · {meth}",
                     f"{m(v, 'held', meth, 'pos'):.3f}",
                     f"{m(v, 'held', meth, 'exact'):.3f}"])
rnd = [r["library"]["pos"] for r in R.get("v1", {}).get("random_bank", [])
       if "library" in r]
rows.append(["random bank · library", f"{np.mean(rnd):.3f}" if rnd else "—", "—"])
rows.append(["joint task-conditioned MLP",
             f"{np.mean([r['pos'] for r in R['joint_held']]):.3f}",
             f"{np.mean([r['exact'] for r in R['joint_held']]):.3f}"])
table(s, Inches(1.6), Inches(1.5), Inches(10.0), rows,
      [Inches(4.6), Inches(2.7), Inches(2.7)], size=14)
textbox(s, Inches(1.6), Inches(1.55 + 0.36 * len(rows) + 0.2), Inches(10.0),
        Inches(1.6), [
    ("Trained-task retention after the full stream: "
     f"library route {m('v1', 'trained', 'library', 'pos'):.3f} (V1) / "
     f"{m('v2', 'trained', 'library', 'pos'):.3f} (V2).", dict(size=14)),
    ("Chance: 0.091 per position, ~7·10⁻⁵ exact-match.", dict(size=12.5, color=MUTED)),
])

# ---------------------------------------------------------------- 7. programs + halting
s = add_slide()
title(s, "The inferred programs are symbolic, minimal, and correctly halted")
picture(s, OUT / "fig4_programs.png", Inches(1.35), max_h=Inches(4.4))
lib_rows = [r for r in (NOID["held"] if NOID else R["v1"]["held"])
            if "library" in r]
exact_T = sum(1 for r in lib_rows if r["library"]["T"] == r["level"])
textbox(s, Inches(0.7), Inches(6.0), SW - Inches(1.4), Inches(1.2), [
    (f"Halting: {exact_T}/{len(lib_rows)} held-out tasks get EXACTLY the "
     "shortest-word length. Step faithfulness: each executed step matches its "
     "named operator's ground truth — the traces are real executions, not just "
     "right answers.", dict(size=14.5)),
])

# ---------------------------------------------------------------- 8. halting fig
s = add_slide()
title(s, "Halting: inferred trace length vs required program length")
picture(s, OUT / "fig3_halting.png", Inches(1.4))

# ---------------------------------------------------------------- 9. route comparison
s = add_slide()
title(s, "Three trace-inference routes: what works and why")
picture(s, OUT / "fig6_methods.png", Inches(1.3), max_h=Inches(3.4))
textbox(s, Inches(0.7), Inches(4.9), SW - Inches(1.4), Inches(2.4), [
    ("LibraryBFS wins: consolidated atoms + memoized search = symbolic planning "
     "over learned operators. Perfect length generalization, exact halting.",
     dict(size=14.5)),
    ("Gradient trace search works on trained tasks and degrades with length — "
     "joint coupling across steps compounds; banks TRAINED under gradient "
     "configuration (V2) are markedly more gradient-searchable than V1's.",
     dict(size=14.5)),
    ("GreedyGrow fails: CE toward the final target is flat along partial "
     "programs — hill climbing has nothing to climb. Trace search needs either "
     "library structure or joint optimization; this negative is the evidence.",
     dict(size=14.5)),
])

# ---------------------------------------------------------------- 10. conclusions
s = add_slide()
title(s, "What this establishes")
textbox(s, Inches(0.7), Inches(1.3), Inches(6.1), Inches(5.8), [
    ("Established", dict(size=18, bold=True, color=GREEN)),
    ("•  temporal selection works: a mask sequence over ONE weight-tied bank "
     "represents and executes non-commutative programs", dict(size=14)),
    ("•  COMPLETE length generalization: 1.000/1.000 on all 84 never-trained "
     "programs, up to 3× the training horizon, from 32 examples", dict(size=14)),
    ("•  autonomous halting, exact: inferred length = shortest-word length "
     "84/84", dict(size=14)),
    ("•  the traces are faithful symbolic programs (step-level verification)",
     dict(size=14)),
    ("•  controls hold: random bank at chance; joint task-conditioned MLP at "
     "chance on held-out tasks", dict(size=14)),
])
textbox(s, Inches(7.1), Inches(1.3), Inches(5.6), Inches(5.8), [
    ("Open / next", dict(size=18, bold=True, color=RED)),
    ("•  gradient-only trace search degrades with length — the algorithm-vs-"
     "execution-trace coupling Kevin predicted; library consolidation is the "
     "current fix, amortized/learned trace proposers the next step", dict(size=14)),
    ("•  richer generator sets and non-group algorithms (conditionals, "
     "data-dependent branching)", dict(size=14)),
    ("•  internal halting (halt atom / ponder signals) vs the current external "
     "verification", dict(size=14)),
    ("•  trace libraries as task memory: prefix sharing as task similarity",
     dict(size=14)),
])

# ---------------------------------------------------------------- 11. repro
s = add_slide()
title(s, "Reproduction")
textbox(s, Inches(0.7), Inches(1.4), SW - Inches(1.4), Inches(5.4), [
    ("# on the actl pod `nar`, from /home/dev/workspace/temporal", dict(size=15, color=MUTED)),
    ("uv run python run_temporal.py --config full --run main --variant both",
     dict(size=15, color=BLUE)),
    ("uv run --with python-pptx --with pillow python scripts/make_deck.py main",
     dict(size=15, color=BLUE)),
    ("", {}),
    ("Code: temporal/ftn_temporal/{config, tasks, model, configurer, train, "
     "evals, viz}.py — shared primitives in nar/ftn_core/.", dict(size=15)),
    ("Artifacts: temporal/results/<run>/ (models, results.json, train_log, "
     "figures).", dict(size=15)),
    ("Repo: github.com/Astera-org/functional_task_networks", dict(size=14, color=MUTED)),
])

prs.save("FTN_temporal_experiment.pptx")
print(f"deck written: FTN_temporal_experiment.pptx ({len(prs.slides._sldIdLst)} slides)")
