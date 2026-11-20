"""Final presentation, built on the proposal deck's template (its master, theme, fonts and layouts).
Usage: python build_deck.py  ->  ../mech_interp_final_presentation.pptx"""
import copy
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "mech_interp_proposal_presentation.pptx"
OUT = HERE.parent / "mech_interp_final_presentation.pptx"
FIGS = HERE / "figures" / "slides"
NAVY, INK, MUTED, GOLD, SAGE, BRICK = "1E2B42", "1A2B42", "5B6472", "C39D53", "7F9A85", "AE5449"
T_GOLD, T_SAGE, T_BRICK, T_BLUE, LINE, WHITE = "F4EDDB", "E5ECE4", "F3E3E0", "E7EDF4", "E2DDD1", "FFFFFF"
FOOTER = "RELATION OR TRIGGER?  ·  FINAL PRESENTATION"

prs = Presentation(str(TEMPLATE))
L_TITLE, L_CONTENT, L_CLOSE = prs.slide_layouts[1], prs.slide_layouts[2], prs.slide_layouts[3]

# ---- drop the proposal's slides
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)

# ---- content layout: no rule under the title, no category label, new footer
for sh in list(L_CONTENT.shapes):
    if sh.name == "Shape 0" or (sh.is_placeholder and sh.placeholder_format.idx == 103):
        sh._element.getparent().remove(sh._element)
    elif sh.has_text_frame and "MECHANISTIC" in sh.text_frame.text:
        sh.text_frame.paragraphs[0].runs[0].text = FOOTER


def rgb(h):
    return RGBColor.from_string(h)


def text(slide, x, y, w, h, runs, size=15, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         spacing=None, italic=False):
    """runs: str, or list of paragraphs; a paragraph is a str or a list of (text, {opts}) runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def shadow(shape):
    sp = shape._element.spPr
    eff = etree.SubElement(sp, qn("a:effectLst"))
    sh = etree.SubElement(eff, qn("a:outerShdw"), blurRad="101600", dist="25400", dir="5400000", algn="bl", rotWithShape="0")
    c = etree.SubElement(sh, qn("a:srgbClr"), val="000000")
    etree.SubElement(c, qn("a:alpha"), val="12000")


def card(slide, x, y, w, h, fill=WHITE, line=LINE, shade=True, radius=0.03):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    if line:
        s.line.color.rgb = rgb(line)
        s.line.width = Pt(0.75)
    else:
        s.line.fill.background()
    if shade:
        shadow(s)
    s.text_frame.text = ""
    return s


def pill(slide, x, y, w, h, label, fill, color=INK, size=11):
    card(slide, x, y, w, h, fill=fill, line=None, shade=False, radius=0.5)
    text(slide, x, y, w, h, label, size=size, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


def image(slide, name, x, y, w=None, h=None):
    path = FIGS / f"{name}.png"
    iw, ih = Image.open(path).size
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


N = [0]


def content(title, notes):
    s = prs.slides.add_slide(L_CONTENT)
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx == 102:
            ph.text_frame.text = title
        else:
            ph._element.getparent().remove(ph._element)
    N[0] += 1
    text(s, 12.15, 7.0, 0.5, 0.25, str(N[0] + 1), size=9, color="8A8F98", align=PP_ALIGN.RIGHT)
    s.notes_slide.notes_text_frame.text = notes
    return s


def dark(layout, notes):
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        ph._element.getparent().remove(ph._element)
    s.notes_slide.notes_text_frame.text = notes
    return s


# =========================================================================================== 1 title
s = dark(L_TITLE, "Last time we proposed a question: when a fallacy detector gets the label right, is it comparing the "
                  "claims, or reacting to a familiar phrase? Today we answer it with probes, minimal pairs, activation "
                  "patching, erasure and head ablation.")
text(s, 1.15, 1.05, 9, 0.3, "MECHANISTIC INTERPRETABILITY  ·  FINAL PRESENTATION", size=10, bold=True, color=GOLD)
text(s, 1.15, 1.75, 11, 1.6, ["Relation or Trigger?", "Inside a Cross-Turn Fallacy Detector"], size=40, bold=True, color=WHITE)
text(s, 1.15, 3.75, 10.5, 0.9, "What a DeBERTa-v3 encoder actually uses to tell motte-and-bailey and strawman apart "
                                "from fair replies", size=18, color="C9CFD8")
text(s, 1.15, 6.35, 9, 0.3, "Aaditya Sharma  ·  2019113009", size=11, color="A9B1BF")

# =========================================================================================== 2 the question
s = content("Same opening, opposite labels",
            "Both replies are real ChangeMyView comments, shown with the turn before them and both open with 'I didn't say'. In the first the speaker "
            "retreats from a strong claim to a weaker one: motte-and-bailey. In the second the speaker correctly "
            "points out that the challenger added the word 'more': a fair clarification. Every example we mined "
            "contains a phrase like this, because that is how candidates were found. So a detector can score well "
            "by reading the phrase. The question is which signal the model actually uses.")
text(s, 0.65, 1.55, 4.9, 1.2, "A detector can get the label right by comparing the claims, or by spotting the phrase",
     size=22, bold=True)
text(s, 0.65, 3.05, 4.7, 1.6, "Every example we mined from ChangeMyView carries a denial or attribution phrase, "
                              "fallacy or not. Accuracy cannot tell the two routes apart. The model's internals can.",
     size=15, color=MUTED)
pill(s, 0.65, 4.85, 2.25, 0.5, "Route 1: the relation", T_BLUE)
pill(s, 3.05, 4.85, 2.25, 0.5, "Route 2: the phrase", T_BRICK)
for i, (lab, fill, a, c, b, col) in enumerate([
        ("MOTTE-AND-BAILEY", T_GOLD,
         "Sweatshops out-competing artisans is a separate issue from governments killing union leaders.",
         "Corrupt governments and businesses ARE linked, and weak international law means little risk for such killings.",
         [("I didn't say", {"bold": True, "color": BRICK}), (" there's no link at all. What I said was: sweatshops are "
                                                             "not NECESSARILY products of corruption.", {})], GOLD),
        ("NONE  ·  FAIR CLARIFICATION", T_SAGE,
         "A child will grow into an adult human; the child deserves the respect owed to a human.",
         "Why are humans entitled to MORE respect than dogs?",
         [("I didn't say", {"bold": True, "color": BRICK}), (" more. Humans deserve respect appropriate to humans, "
                                                             "dogs appropriate to dogs.", {})], SAGE)]):
    y = 1.35 + i * 2.7
    card(s, 6.1, y, 6.55, 2.55)
    pill(s, 6.4, y + 0.2, 2.9 if i else 2.2, 0.36, lab, fill, size=9.5)
    for j, (name, body) in enumerate([("Claim A", a), ("Challenge", c), ("Claim B", [b])]):
        text(s, 6.4, y + 0.72 + j * 0.6, 1.0, 0.3, name, size=10.5, bold=True, color=MUTED)
        text(s, 7.45, y + 0.72 + j * 0.6, 4.95, 0.58, body, size=11.5, color=INK)

# =========================================================================================== 3 what we built
s = content("What we built",
            "The dataset is 683 labelled exchanges from the earlier project: 353 motte-and-bailey exchanges with "
            "three turns (claim A, the challenge, claim B) and 330 strawman exchanges with two (claim A, claim B). "
            "188 are hand-labelled ChangeMyView threads, 20 are hand-written, and the rest are variants of both. On top of that we wrote 416 minimal pairs. We study three "
            "encoders with the same architecture: DeBERTa-v3-large trained on NLI, plain DeBERTa-v3-large, and "
            "random weights as a floor. A linear readout on the frozen NLI encoder already reaches AUC 0.72 and 0.74, "
            "on par with full fine-tuning, so we interpret the encoder itself, at every layer and every input segment. "
            "TransformerLens does not support DeBERTa, so all hooks are our own PyTorch code.")
stats = [("683", "labelled exchanges:\n353 motte-and-bailey,\n330 strawman"), ("416", "minimal-pair edits\nof claim B"),
         ("3", "encoders: NLI, plain,\nrandom init"), ("124k", "linear probes, each\nwith a shuffled-label control"),
         ("41k", "patched forward\npasses")]
for i, (v, l) in enumerate(stats):
    x = 0.65 + i * 2.43
    card(s, x, 1.45, 2.23, 1.75)
    text(s, x + 0.25, 1.6, 1.9, 0.75, v, size=34, bold=True, color=NAVY)
    text(s, x + 0.25, 2.4, 1.9, 0.7, l, size=11.5, color=MUTED)
steps = [("1", "Probe", "Where is the label, the phrase and entailment decodable? 25 layers × 7 token groups.", T_BLUE),
         ("2", "Minimal pairs", "Which edit moves the representation: the relation or the phrase?", T_GOLD),
         ("3", "Patch", "Which positions and layers carry the decision, causally?", T_SAGE),
         ("4", "Erase and ablate", "Remove the phrase or entailment direction (LEACE); knock out heads.", T_BRICK)]
for i, (n, h, d, fill) in enumerate(steps):
    x = 0.65 + i * 3.04
    card(s, x, 3.55, 2.84, 1.9)
    card(s, x + 0.25, 3.8, 0.5, 0.5, fill=fill, line=None, shade=False, radius=0.5)
    text(s, x + 0.25, 3.8, 0.5, 0.5, n, size=14, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.9, 3.85, 1.85, 0.4, h, size=15, bold=True)
    text(s, x + 0.25, 4.5, 2.4, 1.2, d, size=12.5, color=MUTED)
text(s, 0.65, 5.85, 12, 0.5, [[("Frozen DeBERTa-v3-large + linear readout: ", {"bold": True}),
                              ("AUC 0.72 (motte-and-bailey) and 0.74 (strawman), on par with full fine-tuning. "
                               "We interpret the encoder directly, at every layer.", {"color": MUTED})]], size=14)

# =========================================================================================== 4 minimal pairs
s = content("Minimal pairs pull the two routes apart",
            "For each of the 208 original examples we wrote two edits of claim B, leaving A and C untouched. The "
            "relation edit flips the label but keeps the phrase word for word. The phrase edit removes the phrase but "
            "keeps the meaning and the label. Because A and C are identical, every position before B lines up between "
            "the two inputs, which is what activation patching needs. Every edit passed automatic checks: relation "
            "edits keep the trigger, phrase edits toggle it.")
cols = [("ORIGINAL", "STRAWMAN", T_BRICK,
         [("I'm just trying to clarify -- ", {}), ("are you saying", {"bold": True, "color": BRICK}),
          (" people who are good at math/physics are NOT good at history?", {})]),
        ("RELATION EDIT", "NONE", T_SAGE,
         [("I'm just trying to clarify -- ", {}), ("are you saying", {"bold": True, "color": BRICK}),
          (" math and history ability can vary independently, so someone can be good at one and not the other?", {})]),
        ("PHRASE EDIT", "STRAWMAN", T_BRICK,
         [("I'm just trying to clarify -- people who are good at math/physics are NOT good at history?", {})])]
card(s, 0.65, 1.45, 12.0, 0.85, fill=T_BLUE, line=None, shade=False)
text(s, 0.95, 1.45, 1.2, 0.85, "Claim A", size=12, bold=True, color=MUTED, anchor=MSO_ANCHOR.MIDDLE)
text(s, 2.1, 1.45, 10.3, 0.85, "Aptitude in math/physics and in history varies independently: some people are strong "
                               "in one and weak in the other.", size=14, anchor=MSO_ANCHOR.MIDDLE)
for i, (h, lab, fill, runs) in enumerate(cols):
    x = 0.65 + i * 4.07
    card(s, x, 2.6, 3.86, 2.15)
    text(s, x + 0.3, 2.85, 2.2, 0.3, h, size=10, bold=True, color=MUTED)
    pill(s, x + 2.35, 2.8, 1.25, 0.36, lab, fill, size=9.5)
    text(s, x + 0.3, 3.4, 3.3, 1.7, [runs], size=14)
for i, (v, l) in enumerate([("208", "relation edits: label flips, phrase kept"),
                            ("208", "phrase edits: label kept, phrase removed or added"),
                            ("A, C fixed", "so token positions align for patching")]):
    x = 0.65 + i * 4.07
    text(s, x, 5.2, 3.86, 0.5, v, size=24, bold=True, color=NAVY)
    text(s, x, 5.75, 3.86, 0.4, l, size=12.5, color=MUTED)

# =========================================================================================== 5 Exp 1 phrase vs fallacy
s = content("The phrase is visible at once; the fallacy builds up",
            "One linear probe per layer, on the claim-B tokens, grouped cross-validation, three seeds. The grey band "
            "is the same probe on shuffled labels. The trigger phrase is almost perfectly decodable from the very "
            "first layer, AUC 0.94 to 0.98. The fallacy label is not: for motte-and-bailey it starts at chance and "
            "climbs to 0.73 by the last layers. So the phrase is an easy, early signal. The fallacy is something the "
            "network has to compute. The two directions are also nearly orthogonal, absolute cosine below 0.12 at every layer.")
image(s, "s_trigger_vs_label", 0.65, 1.45, w=8.6)
for i, (v, l, c) in enumerate([("0.94+", "phrase AUC on claim B,\nalready at layer 0", BRICK),
                               ("0.50 → 0.73", "motte-and-bailey label,\nlayer 0 → layer 24", NAVY),
                               ("< 0.12", "cosine between fallacy\nand phrase directions", MUTED)]):
    y = 1.5 + i * 1.6
    card(s, 9.65, y, 3.0, 1.4)
    text(s, 9.9, y + 0.15, 2.6, 0.6, v, size=26, bold=True, color=c)
    text(s, 9.9, y + 0.78, 2.6, 0.6, l, size=11.5, color=MUTED)

# =========================================================================================== 6 context
s = content("Motte-and-bailey needs the earlier turns",
            "Same probe, three inputs. Navy is the full exchange, A, C and B. Gold encodes claim B on its own. Grey "
            "is the random-weights encoder. For motte-and-bailey, B alone plateaus around 0.60 to 0.64, while B read "
            "in context climbs to 0.73: the extra signal only exists when the model can see what was claimed before. "
            "That is the retreat being detected. For strawman, B carries most of the signal by itself, which fits the "
            "task: a distorted paraphrase is already visible in how B is worded. Random weights stay at chance.")
image(s, "s_context", 0.65, 1.45, w=8.6)
card(s, 9.65, 1.5, 3.0, 3.6)
text(s, 9.95, 1.75, 2.45, 0.8, "+0.12 AUC", size=28, bold=True, color=NAVY)
text(s, 9.95, 2.55, 2.45, 1.0, "from adding A and C, on the motte-and-bailey B tokens at layer 24", size=13, color=MUTED)
text(s, 9.95, 3.75, 2.45, 2.2, "The retreat is only visible against the original claim. The encoder picks this up in its "
                               "last third.", size=14, color=INK)

# =========================================================================================== 7 geometry
s = content("Relation edits move the model; phrase edits barely do",
            "For every held-out pair we measure how far the edit moves the representation along the fallacy "
            "direction, in standard deviations, with probes fitted without that conversation. Navy: relation edits, "
            "signed toward the new label. Red: phrase edits, absolute. For strawman, relation edits move the model "
            "1.3 sd by the last layer, phrase edits 0.3: four times more. For motte-and-bailey the relation edit "
            "peaks at 0.73 at layer 20, about 2.4 times the phrase edit. The two curves are the same in the early "
            "layers and separate from layer 14.")
image(s, "s_geometry", 0.65, 1.45, w=8.6)
card(s, 9.65, 1.5, 3.0, 2.15)
text(s, 9.95, 1.65, 2.5, 0.8, "4×", size=40, bold=True, color=NAVY)
text(s, 9.95, 2.55, 2.5, 1.0, "strawman: relation edits\n1.30 sd vs phrase edits 0.32", size=12.5, color=MUTED)
card(s, 9.65, 3.9, 3.0, 2.15)
text(s, 9.95, 4.05, 2.5, 0.8, "2.4×", size=40, bold=True, color=NAVY)
text(s, 9.95, 4.95, 2.5, 1.0, "motte-and-bailey at layer 20:\n0.73 sd vs 0.30", size=12.5, color=MUTED)

# =========================================================================================== 8 patching
s = content("The decision forms in B, then flows back into A",
            "Activation patching. We run the edited input, and at one layer and one segment we paste in the "
            "activations from the original. The colour is the share of the original decision that comes back. Up to "
            "layer 16 only B matters: patching B restores everything, patching A restores nothing. From layer 17 the "
            "decision starts to live in A's tokens as well, even though A's words are identical in both inputs. That "
            "information can only have come from B through attention: the model writes the comparison back into "
            "the original claim. Relation edits also change the readout three to four times more than phrase edits.")
image(s, "s_patching", 0.55, 1.45, w=7.35)
x0 = 8.2
for i, (v, l) in enumerate([("Layers 0-16", "B carries over 90% of the decision"),
                            ("Layers 17-23", "A and C take over up to 46%"),
                            ("1.4 vs 0.3 sd", "readout change: relation edit vs phrase edit (strawman)")]):
    y = 1.5 + i * 1.55
    card(s, x0, y, 4.45, 1.35)
    text(s, x0 + 0.3, y + 0.17, 3.9, 0.5, v, size=20, bold=True, color=NAVY)
    text(s, x0 + 0.3, y + 0.7, 3.9, 0.55, l, size=12.5, color=MUTED)
pill(s, x0, 6.2, 4.45, 0.45, "Clean run → paste one segment at one layer → corrupted run", T_BLUE, size=10.5)

# =========================================================================================== 9 erasure
s = content("Erasing the phrase changes nothing; erasing entailment does",
            "LEACE removes a concept's linear direction exactly. We erase it from every token at one layer, let the "
            "network run on, and score held-out examples with the final readout. Erasing the trigger direction does "
            "nothing at any layer: the AUC stays on the dashed baseline. Erasing the entailment direction, A entails "
            "B, hurts in the last layers: strawman drops from 0.71 to 0.66. Erasing a random direction does nothing, "
            "and erasing the label itself drops the probe to chance. The phrase is present but "
            "unused; entailment is used.")
image(s, "s_erasure", 0.65, 1.45, w=8.6)
for i, (v, l, c) in enumerate([("±0.01", "AUC change after erasing\nthe phrase, at any layer", BRICK),
                               ("−0.05", "strawman AUC after erasing\nentailment at layer 24", NAVY)]):
    y = 1.6 + i * 2.1
    card(s, 9.65, y, 3.0, 1.8)
    text(s, 9.9, y + 0.25, 2.6, 0.7, v, size=32, bold=True, color=c)
    text(s, 9.9, y + 1.05, 2.6, 0.6, l, size=12.5, color=MUTED)

# =========================================================================================== 10 heads
s = content("Ten late attention heads carry the strawman decision",
            "We ranked all 384 heads and 24 MLPs with attribution patching, one forward and backward pass per pair, "
            "and confirmed the top ones with real patching; the two rankings agree. The components that write the "
            "B-dependent signal back into A are late: MLPs in layers 21 to 23 and heads in layers 20 to 23. Removing "
            "the top ten heads on held-out real examples drops strawman AUC from 0.71 to 0.44, below chance. Removing "
            "ten random heads from the same layers barely matters, 0.69.")
image(s, "s_heads", 0.65, 1.45, w=7.0)
card(s, 8.0, 1.5, 4.65, 4.65)
text(s, 8.3, 1.75, 4.0, 0.3, "WHAT THE TOP COMPONENTS ARE", size=10, bold=True, color=MUTED)
for i, (h, d) in enumerate([("L23H11  ·  L23H5", "strongest heads, last layer"),
                            ("L20H3  ·  L21H2  ·  L22H12", "write the comparison into A"),
                            ("MLP 21-23", "largest single contributors"),
                            ("384 heads → 10", "ranked by attribution patching, verified one by one")]):
    y = 2.2 + i * 0.95
    text(s, 8.3, y, 4.1, 0.4, h, size=15, bold=True, color=NAVY)
    text(s, 8.3, y + 0.4, 4.1, 0.4, d, size=12, color=MUTED)

# =========================================================================================== 11 NLI vs plain
s = content("NLI pre-training sharpens the relational signal",
            "RQ4. Same architecture, same probes, with and without NLI pre-training. The NLI encoder separates the "
            "minimal pairs more on every measure: relation edits move it 1.30 sd against 0.92, the patched readout "
            "gap is about 1.4 times larger, and its late heads are causally important, while removing the same "
            "number of heads in the plain encoder costs almost nothing. Entailment erasure also hurts the NLI readout "
            "more. Both encoders can decode the label about equally well; NLI pre-training changes how the decision "
            "is computed, not just how well.")
rows = [("Relation-edit shift, strawman (sd)", "1.30", "0.92"),
        ("Readout gap on relation edits, strawman (sd)", "1.36", "1.04"),
        ("Readout gap on relation edits, motte-and-bailey (sd)", "1.02", "0.71"),
        ("AUC lost when the top-10 heads are removed, strawman", "0.27", "0.02"),
        ("AUC lost when entailment is erased at layer 24, strawman", "0.05", "0.03")]
card(s, 0.65, 1.45, 12.0, 4.85)
text(s, 0.95, 1.7, 6.5, 0.35, "MEASURE", size=10, bold=True, color=MUTED)
pill(s, 8.15, 1.62, 2.0, 0.45, "DeBERTa-NLI", NAVY, color=WHITE, size=11)
pill(s, 10.4, 1.62, 2.0, 0.45, "DeBERTa plain", T_GOLD, size=11)
for i, (m, a, b) in enumerate(rows):
    y = 2.35 + i * 0.75
    if i:
        ln = s.shapes.add_connector(1, Inches(0.95), Inches(y - 0.12), Inches(12.35), Inches(y - 0.12))
        ln.line.color.rgb = rgb(LINE)
        ln.line.width = Pt(0.75)
    text(s, 0.95, y, 7.0, 0.5, m, size=15, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 8.15, y, 2.0, 0.5, a, size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, 10.4, y, 2.0, 0.5, b, size=20, bold=True, color=GOLD, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# =========================================================================================== 12 closing
s = dark(L_CLOSE, "To sum up. The phrase is the easiest thing in the input to read, and the model reads it, but it "
                  "does not decide with it. The fallacy is computed from B in context, written back into A in the last "
                  "layers by a small set of late heads and MLPs, and NLI pre-training makes that relational computation "
                  "sharper. Thank you; happy to take questions.")
text(s, 0.65, 0.9, 8, 0.3, "ANSWER", size=10, bold=True, color=GOLD)
text(s, 0.65, 1.3, 12, 0.8, "Relation, not trigger", size=36, bold=True, color=WHITE)
ans = [("RQ1  WHERE", "The phrase is decodable from layer 0; the fallacy builds up and peaks in layers 18-24."),
       ("RQ2  WHAT", "Relation edits move the model up to 4× more than phrase edits; erasing the phrase changes nothing."),
       ("RQ3  CAUSAL", "Decided in B, then written into A in layers 17-23 by late MLPs and ten heads."),
       ("RQ4  NLI", "NLI pre-training sharpens the relational computation on every causal measure.")]
for i, (h, d) in enumerate(ans):
    x = 0.65 + i * 3.04
    c = card(s, x, 2.6, 2.84, 2.4, fill="263A57", line=None, shade=False)
    text(s, x + 0.3, 2.85, 2.3, 0.3, h, size=11, bold=True, color=GOLD)
    text(s, x + 0.3, 3.3, 2.3, 2.2, d, size=15, color=WHITE)
text(s, 0.65, 5.45, 12, 0.5, "The model reads the phrase, but it decides on the relation.", size=16, color="C9CFD8")

prs.save(str(OUT))
print("wrote", OUT, len(prs.slides), "slides")
