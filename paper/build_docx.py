"""
Build the IEEE conference paper as a Word document (IEEE two-column template layout).

    python make_figures.py          # (repo root) figures, tables and numbers.tex from results/
    python paper/make_docx_assets.py
    python paper/build_docx.py      # -> paper/GeoColor_IEEE_Conference_Paper.docx

Every number is read from results/metrics.json (through paper/tables/numbers.tex), so the Word file and the
LaTeX file report identical values.
"""
import json
import os
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

from omml import latex_to_omml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "GeoColor_IEEE_Conference_Paper_final.docx")

R = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
S, PC, D, W = R["summary"], R["per_class"], R["despeckling"], R["wilcoxon_vs_ours"]
MAIN = R["main"]

# ------------------------------------------------------------------ numbers from numbers.tex
N = {}
for name, val in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", open(os.path.join(HERE, "tables", "numbers.tex")).read(), re.M):
    m = re.match(r"\\ensuremath\{<10\^\{(-?\d+)\}\}", val)
    N[name] = f"< 10^{{{m.group(1).replace('-', '−')}}}" if m else val


# ------------------------------------------------------------------ citations (numbered by first use)
REFS = {
    "torres2012s1": "R. Torres et al., “GMES Sentinel-1 mission,” Remote Sens. Environ., vol. 120, pp. 9–24, 2012.",
    "goodman1976speckle": "J. W. Goodman, “Some fundamental properties of speckle,” J. Opt. Soc. Amer., vol. 66, no. 11, pp. 1145–1150, Nov. 1976.",
    "isola2017pix2pix": "P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 1125–1134.",
    "zhu2017cyclegan": "J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 2223–2232.",
    "wang2019sar2opt": "L. Wang et al., “SAR-to-optical image translation using supervised cycle-consistent adversarial networks,” IEEE Access, vol. 7, pp. 129136–129149, 2019.",
    "fuentes2019sar2opt": "M. Fuentes Reyes, S. Auer, N. Merkle, C. Henry, and M. Schmitt, “SAR-to-optical image translation based on conditional generative adversarial networks—Optimization, opportunities and limits,” Remote Sens., vol. 11, no. 17, p. 2067, 2019.",
    "zhang2016colorful": "R. Zhang, P. Isola, and A. A. Efros, “Colorful image colorization,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2016, pp. 649–666.",
    "lee1980filter": "J.-S. Lee, “Digital image enhancement and noise filtering by use of local statistics,” IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-2, no. 2, pp. 165–168, Mar. 1980.",
    "rudin1992tv": "L. I. Rudin, S. Osher, and E. Fatemi, “Nonlinear total variation based noise removal algorithms,” Physica D, vol. 60, no. 1–4, pp. 259–268, 1992.",
    "chierchia2017sarcnn": "G. Chierchia, D. Cozzolino, G. Poggi, and L. Verdoliva, “SAR image despeckling through convolutional neural networks,” in Proc. IEEE Int. Geosci. Remote Sens. Symp. (IGARSS), 2017, pp. 5438–5441.",
    "zhang2017dncnn": "K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising,” IEEE Trans. Image Process., vol. 26, no. 7, pp. 3142–3155, Jul. 2017.",
    "gal2016dropout": "Y. Gal and Z. Ghahramani, “Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,” in Proc. Int. Conf. Mach. Learn. (ICML), 2016, pp. 1050–1059.",
    "kendall2017uncertainty": "A. Kendall and Y. Gal, “What uncertainties do we need in Bayesian deep learning for computer vision?,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 5574–5584.",
    "ilg2018uncertainty": "E. Ilg et al., “Uncertainty estimates and multi-hypotheses networks for optical flow,” in Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 652–667.",
    "ronneberger2015unet": "O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. Med. Image Comput. Comput.-Assist. Interv. (MICCAI), 2015, pp. 234–241.",
    "perez2018film": "E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, “FiLM: Visual reasoning with a general conditioning layer,” in Proc. AAAI Conf. Artif. Intell., 2018, pp. 3942–3951.",
    "vivone2015pansharp": "G. Vivone et al., “A critical comparison among pansharpening algorithms,” IEEE Trans. Geosci. Remote Sens., vol. 53, no. 5, pp. 2565–2586, May 2015.",
    "schmitt2019sen12ms": "M. Schmitt, L. H. Hughes, C. Qiu, and X. X. Zhu, “SEN12MS—A curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion,” ISPRS Ann. Photogramm. Remote Sens. Spatial Inf. Sci., vol. IV-2/W7, pp. 153–160, 2019.",
    "drusch2012s2": "M. Drusch et al., “Sentinel-2: ESA’s optical high-resolution mission for GMES operational services,” Remote Sens. Environ., vol. 120, pp. 25–36, 2012.",
    "wang2004ssim": "Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Trans. Image Process., vol. 13, no. 4, pp. 600–612, Apr. 2004.",
    "kruse1993sam": "F. A. Kruse et al., “The spectral image processing system (SIPS)—Interactive visualization and analysis of imaging spectrometer data,” Remote Sens. Environ., vol. 44, no. 2–3, pp. 145–163, 1993.",
    "xue2014gmsd": "W. Xue, L. Zhang, X. Mou, and A. C. Bovik, “Gradient magnitude similarity deviation: A highly efficient perceptual image quality index,” IEEE Trans. Image Process., vol. 23, no. 2, pp. 684–695, Feb. 2014.",
    "sattar1997epi": "F. Sattar, L. Floreby, G. Salomonsson, and B. Lövström, “Image enhancement based on a nonlinear multiscale method,” IEEE Trans. Image Process., vol. 6, no. 6, pp. 888–895, Jun. 1997.",
    "blau2018perception": "Y. Blau and T. Michaeli, “The perception-distortion tradeoff,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 6228–6237.",
    "rombach2022ldm": "R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, “High-resolution image synthesis with latent diffusion models,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 10684–10695.",
    "liu2021swin": "Z. Liu et al., “Swin Transformer: Hierarchical vision transformer using shifted windows,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2021, pp. 10012–10022.",
}
ORDER = []


def cite(*keys):
    nums = []
    for k in keys:
        if k not in REFS:
            raise KeyError(k)
        if k not in ORDER:
            ORDER.append(k)
        nums.append(ORDER.index(k) + 1)
    return ", ".join(f"[{n}]" for n in nums)


# ------------------------------------------------------------------ low-level formatting helpers
doc = Document()
FONT = "Times New Roman"


def set_font(run, size=10, bold=None, italic=None, smallcaps=False, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if smallcaps:
        run.font.small_caps = True
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def para(align=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.2, before=0, after=0, keep=False):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.alignment = align
    pf.first_line_indent = Inches(indent) if indent else None
    pf.space_before, pf.space_after = Pt(before), Pt(after)
    pf.line_spacing = 1.0
    if keep:
        pf.keep_with_next = True
    return p


TOKEN = re.compile(r"(\*[^*]+\*|\^\{[^}]*\}|_\{[^}]*\})")


def add_rich(p, text, size=10, bold=None, italic=False):
    """Mini markup: *italic*, ^{superscript}, _{subscript}."""
    for part in TOKEN.split(text):
        if not part:
            continue
        r_italic, sup, sub = italic, False, False
        if part.startswith("*") and part.endswith("*"):
            part, r_italic = part[1:-1], not italic
        elif part.startswith("^{"):
            part, sup = part[2:-1], True
        elif part.startswith("_{"):
            part, sub = part[2:-1], True
        r = p.add_run(part)
        set_font(r, size, bold=bold, italic=r_italic)
        # set only the flag that applies: assigning None to the other one clears w:vertAlign entirely
        if sup:
            r.font.superscript = True
        elif sub:
            r.font.subscript = True
    return p


def body(text):
    return add_rich(para(), text)


def set_cols(section, num, space_in=0.24):
    sectPr = section._sectPr
    cols = sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sectPr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(space_in * 1440)))


def page_setup(section):
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(0.75), Inches(1.0)
    section.left_margin = section.right_margin = Inches(0.625)


def new_section(cols):
    s = doc.add_section(WD_SECTION.CONTINUOUS)
    page_setup(s)
    set_cols(s, cols)
    return s


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
H1, H2 = [0], [0]


def heading1(text, numbered=True):
    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=8, after=4, keep=True)
    label = f"{ROMAN[H1[0]]}.\u2002" if numbered else ""
    if numbered:
        H1[0] += 1
    H2[0] = 0
    set_font(p.add_run(label + text), 10, smallcaps=True)


def heading2(text):
    p = para(WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=6, after=3, keep=True)
    set_font(p.add_run(f"{chr(ord('A') + H2[0])}.\u2002{text}"), 10, italic=True)
    H2[0] += 1


EQN = [0]


EQ_LATEX = {
    "eq1": r"Y = X \cdot F, \quad \mathbb{E}[F] = 1",
    "eq2": r"\log Y = \log X + \log F",
    "eq3": r"\mathcal{L}_{rad} = \left\| \mathcal{A}_{w}(\hat{X}) - \mathcal{A}_{w}(Y) \right\|_{1}",
    "eq4": r"\mathcal{L}_{tv} = \left\| \nabla \log(\hat{X} + \epsilon) \right\|_{1}",
    "eq5": r"h \mapsto h (1 + \gamma_{c}) + \beta_{c}",
    # long equations are given as lines; the number goes on the last line
    "eq6": [r"\mathcal{L} = \lambda_{1}\mathcal{L}_{1} + \lambda_{s}(1 - \mathrm{SSIM}) + \lambda_{g}\mathcal{L}_{grad}",
            r"+ \lambda_{a}\mathcal{L}_{GAN} + \lambda_{r}\mathcal{L}_{rad} + \lambda_{tv}\mathcal{L}_{tv}"],
    "eq7": r"L^{*} = \alpha S + (1 - \alpha) L^{*}_{pred}",
    "eq8": r"L^{*} = L^{*}_{pred} + \beta \frac{\sigma(L^{*}_{pred})}{\sigma(S)} (S - G_{\sigma} * S)",
    "eq9": [r"\mu = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_{t},",
            r"\sigma^{2} = \frac{1}{T-1} \sum_{t=1}^{T} (\hat{y}_{t} - \mu)^{2}"],
}


def equation(name, col_width=3.5):
    """Native, editable Word equation (OMML) centred in the column, number right-aligned, IEEE style."""
    EQN[0] += 1
    lines = EQ_LATEX[name] if isinstance(EQ_LATEX[name], list) else [EQ_LATEX[name]]
    for k, tex in enumerate(lines):
        last = k == len(lines) - 1
        p = para(WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=3 if k == 0 else 0, after=3 if last else 0, keep=not last)
        ts = p.paragraph_format.tab_stops
        ts.add_tab_stop(Inches(col_width / 2), WD_TAB_ALIGNMENT.CENTER)
        ts.add_tab_stop(Inches(col_width), WD_TAB_ALIGNMENT.RIGHT)
        set_font(p.add_run("\t"), 10)
        p._p.append(latex_to_omml(tex))
        if last:
            set_font(p.add_run(f"\t({EQN[0]})"), 10)
    return EQN[0]


FIGN, TABN = [0], [0]


def figure(path, width_in, caption):
    FIGN[0] += 1
    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=4, after=2, keep=True)
    p.add_run().add_picture(path, width=Inches(width_in))
    c = para(WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0, before=0, after=6)
    set_font(c.add_run(f"Fig. {FIGN[0]}.\u2002"), 8)
    add_rich(c, caption, size=8)
    return FIGN[0]


def table_caption(caption):
    TABN[0] += 1
    p = para(WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=6, after=1, keep=True)
    set_font(p.add_run(f"TABLE {ROMAN[TABN[0] - 1]}"), 8, smallcaps=True)
    p2 = para(WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=0, after=3, keep=True)
    add_rich(p2, caption, size=8)
    return ROMAN[TABN[0] - 1]


def cell_border(cell, **edges):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = tcPr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge, sz in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single" if sz else "nil")
        if sz:
            el.set(qn("w:sz"), str(sz)); el.set(qn("w:color"), "000000")
        borders.append(el)


def fix_grid(t, widths):
    """Word ignores cell widths unless the table grid and a fixed layout are set explicitly."""
    tbl = t._tbl
    tblPr = tbl.tblPr
    for tag in ("w:tblW", "w:tblLayout"):
        old = tblPr.find(qn(tag))
        if old is not None:
            tblPr.remove(old)
    tw = OxmlElement("w:tblW")
    tw.set(qn("w:w"), str(int(sum(widths) * 1440)))
    tw.set(qn("w:type"), "dxa")
    lay = OxmlElement("w:tblLayout")
    lay.set(qn("w:type"), "fixed")
    tblPr.append(tw)
    tblPr.append(lay)
    for gc, wdt in zip(tbl.tblGrid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(wdt * 1440)))
    for row in t.rows:
        for c, wdt in zip(row.cells, widths):
            c.width = Inches(wdt)
    # compact cell margins (0.03 in) so short numeric cells do not wrap
    mar = tblPr.find(qn("w:tblCellMar"))
    if mar is None:
        mar = OxmlElement("w:tblCellMar")
        tblPr.append(mar)
    for side in ("left", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:w"), "43")
        el.set(qn("w:type"), "dxa")
        mar.append(el)


def keep_table_together(t):
    """Rows never split and every row stays with the next, so a table is never broken across columns."""
    for r_i, row in enumerate(t.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        trPr.append(cs)
        if r_i < len(t.rows) - 1:
            for c in row.cells:
                for p in c.paragraphs:
                    p.paragraph_format.keep_with_next = True


def data_table(header, rows, widths, bold_rows=(), size=7.5, align_first=WD_ALIGN_PARAGRAPH.LEFT):
    """Fully ruled table: horizontal and vertical lines on every cell, centred numbers, left-aligned names."""
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.width = Inches(widths[j])
            p = c.paragraphs[0]
            p.paragraph_format.alignment = align_first if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(0.5)
            add_rich(p, str(val), size=size, bold=True if (i == 0 or (i - 1) in bold_rows) else None)
            c.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell_border(c, top=4, bottom=4, left=4, right=4)
            if i == 0:  # light shading on the header row
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7E6E6")
                c._tc.get_or_add_tcPr().append(shd)
    fix_grid(t, widths)
    keep_table_together(t)
    spacer = para(indent=0, after=4)
    set_font(spacer.add_run(""), 4)
    return t


def fmt(v, k):
    return f"{v:.3f}" if k in ("ssim", "gmsd") else f"{v:.2f}"


def nsig(m, k):
    v = W.get(m, {}).get(k)
    return v is not None and v >= 0.05


# ------------------------------------------------------------------ document
sec0 = doc.sections[0]
page_setup(sec0)
set_cols(sec0, 1)
normal = doc.styles["Normal"]
normal.font.name, normal.font.size = FONT, Pt(10)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

t = para(WD_ALIGN_PARAGRAPH.CENTER, indent=0, after=10)
set_font(t.add_run("Physics-Guided SAR-to-Optical Colorization with Radiometry-Preserving Despeckling: "
                   "A Leakage-Free Benchmark on Sentinel-1/2"), 24)

AUTHORS = [
    ("SELVANAYAKI S", "M.Tech., Assistant Professor", "Department of Artificial Intelligence and Data Science", "selvanayakis@saveetha.ac.in"),
    ("Dr. K. MICHEAL MAHESH", "M.E., Ph.D., MIE, Professor", "Department of Electronics and Communication Engineering", "michealmaheshk@saveetha.ac.in"),
    ("KOLLI AMOS DANIEL", "M.Tech., (Ph.D), Assistant Professor", "Department of Artificial Intelligence and Machine Learning", "kollihimantharao@saveetha.ac.in"),
    ("GEDIPUDI DARSHANI", None, "Department of Artificial Intelligence and Data Science", "gedipudidarshani@gmail.com"),
    ("YENUGANTI PRATHYUSHA", None, "Department of Artificial Intelligence and Machine Learning", "yenugantiprathyusharavi@gmail.com"),
    ("P KEERTHANA", None, "Department of Artificial Intelligence and Machine Learning", "keerthanap816@gmail.com"),
]
at = doc.add_table(rows=2, cols=3)
at.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, (name, desig, dept, mail) in enumerate(AUTHORS):
    c = at.cell(i // 3, i % 3)
    c.width = Inches(2.4)
    lines = [(name, 10.5, False)] + ([(desig, 9, False)] if desig else []) + \
            [(dept, 9, True), ("Saveetha Engineering College", 9, False), ("Chennai, India", 9, False), (mail, 9, False)]
    for k, (txt, sz, it) in enumerate(lines):
        p = c.paragraphs[0] if k == 0 else c.add_paragraph()
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(6 if k == len(lines) - 1 and i < 3 else 0)
        set_font(p.add_run(txt), sz, italic=it)
    cell_border(c, top=0, bottom=0, left=0, right=0)
fix_grid(at, [2.4, 2.4, 2.4])
para(indent=0, after=6)

new_section(2)

# Abstract + keywords (IEEE: 9 pt bold, no symbols)
p = para(indent=0.2, after=4)
set_font(p.add_run("Abstract—"), 9, bold=True, italic=True)
add_rich(p, (
    "Synthetic Aperture Radar (SAR) images the Earth through cloud and at night, but its single-channel, speckled "
    "backscatter is hard for non-experts to interpret. We study SAR-to-optical colorization with a two-stage, "
    "physics-guided network. Stage 1 is a log-domain (homomorphic) residual despeckler trained without clean references; "
    "it is supervised by a prior derived from the multiplicative speckle model (local radiometric preservation plus "
    "log-domain total variation). Stage 2 is a U-Net colorizer with an ImageNet-pretrained ResNet-18 encoder and "
    "terrain-conditioned FiLM layers. On 16,000 Sentinel-1/Sentinel-2 pairs with a scene-level split (no test scene is "
    "seen in training) we train nine models under an identical budget. On "
    f"{N['nTest']} test tiles the final model reaches {N['psnrOursRBICondNoadv']} dB PSNR and {N['ssimOursRBICondNoadv']} "
    f"SSIM, {N['dFinalVsPix']} dB above Pix2Pix and {N['dFinalVsUnet']} dB above a U-Net regressor, a difference that is "
    "statistically significant under a paired Wilcoxon test. The physics prior makes Stage 1 radiometrically unbiased "
    f"(mean of ratio {N['morOurs']} versus {N['morOursNophys']} without it). We also report three negative results: the "
    f"adversarial loss costs {N['dGanCost']} dB on unseen scenes, terrain conditioning does not transfer across scenes, "
    f"and Monte Carlo dropout variance does not rank pixel errors (Spearman correlation {N['uncRho']}). The absolute "
    "scores show that colorizing unseen scenes from single-polarization SAR remains far from solved."), size=9, bold=True)
p = para(indent=0.2, after=6)
set_font(p.add_run("Keywords—synthetic aperture radar, image colorization, despeckling, Sentinel-1, Sentinel-2, "
                   "image-to-image translation, uncertainty"), 9, bold=True, italic=True)

# I. Introduction
heading1("Introduction")
body("Optical satellites such as Sentinel-2 cannot image the ground through cloud, smoke or at night, which is often "
     "exactly when imagery is needed for disaster response. Synthetic Aperture Radar (SAR) such as Sentinel-1 is an active "
     f"microwave sensor that works in all weather, day and night {cite('torres2012s1')}. Its output, however, is a "
     f"single-channel backscatter image corrupted by multiplicative speckle {cite('goodman1976speckle')}, and it is hard to "
     "read for anyone who is not a radar specialist. Translating SAR into an optical-like colour image is therefore "
     "attractive and is the subject of the ISRO Smart India Hackathon problem SIH1733, “SAR Image Colorization for "
     "Comprehensive Insight using Deep Learning Model”.")
body("Most published approaches treat the task as generic image-to-image translation with conditional or cycle-consistent "
     f"GANs {cite('isola2017pix2pix', 'zhu2017cyclegan', 'wang2019sar2opt', 'fuentes2019sar2opt')}. One network then has "
     "to suppress speckle and infer colour at the same time, with no model of how speckle is formed. The translation is "
     "also fundamentally ambiguous: the same backscatter can correspond to many optical appearances. In addition, "
     "evaluations often draw training and test patches from the same scenes; overlapping patches then leak between the "
     "splits and the scores come out optimistic.")
body("Our contributions are:")
for b in [
    "A two-stage generator whose first stage is a homomorphic despeckler trained *without clean SAR references*. It is "
    "supervised by a physics prior derived from the speckle model, and we show that this prior removes a large "
    "radiometric bias that the stage otherwise develops.",
    "A leakage-free, scene-level benchmark on 16,000 Sentinel-1/2 pairs. Nine models (three baselines and six variants of "
    "ours) plus two trivial floors are compared, all trained under an identical budget and compared with paired "
    "significance tests.",
    "Controlled evidence that, on unseen scenes, the adversarial loss lowers PSNR by about 2 dB and the non-adversarial "
    "physics-guided models perform best, significantly ahead of Pix2Pix and a U-Net regressor.",
    "Three negative results that we report rather than omit: terrain conditioning, Monte Carlo dropout uncertainty and "
    "SAR luminance fusion do not deliver the gains commonly assumed.",
]:
    p = para(indent=0, after=1)
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.15)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.25))
    add_rich(p, "•\t" + b)

# II. Related work
heading1("Related Work")
heading2("SAR-to-Optical Translation")
body(f"Pix2Pix {cite('isola2017pix2pix')} couples an L1 loss with a PatchGAN discriminator, and CycleGAN "
     f"{cite('zhu2017cyclegan')} removes the need for paired data. Both have been adapted to SAR "
     f"{cite('wang2019sar2opt', 'fuentes2019sar2opt')}. Fuentes Reyes *et al.* {cite('fuentes2019sar2opt')} observe that "
     "such models produce plausible textures but unreliable colours on unseen scenes. Natural-image colorization "
     f"{cite('zhang2016colorful')} relies on semantic priors that do not transfer to radar.")
heading2("Despeckling")
body(f"The Lee filter {cite('lee1980filter')} uses local statistics, and total-variation methods {cite('rudin1992tv')} "
     f"favour piecewise-smooth solutions. CNN despecklers {cite('chierchia2017sarcnn')} use residual learning "
     f"{cite('zhang2017dncnn')} in the log domain but normally need clean or multi-temporal references. Our Stage 1 "
     "instead learns from a physics prior, jointly with the colorization objective.")
heading2("Uncertainty")
body(f"MC dropout {cite('gal2016dropout')} approximates Bayesian inference by sampling dropout masks at test time to "
     f"estimate epistemic uncertainty. Sparsification analysis "
     f"{cite('ilg2018uncertainty')} tests whether that uncertainty actually ranks pixels by their error.")

# III. Proposed methodology: architecture diagram (same flowchart as the GitHub README) inside this column
heading1("Proposed Methodology")
body("Fig. 1 shows the proposed pipeline, which is also documented in the project repository.")
f_arch = figure(os.path.join(FIG, "fig_architecture_column.png"), 3.1,
                "Proposed architecture. Stage 1 removes speckle in the log domain and is supervised by the physics prior "
                "*L*_{rad} + *L*_{tv}. Stage 2 predicts RGB from the despeckled image, modulated by the terrain class "
                "(FiLM). Stage 3 optionally injects the high-pass SAR detail. Repeated stochastic passes through the decoder "
                "dropout layers give a predictive mean and variance.")
heading2("Speckle Model and Stage 1")
body("The observed intensity is modelled as")
e1 = equation("eq1")
body(f"where X is the reflectivity and F is speckle {cite('goodman1976speckle')}. In the log domain this becomes additive,")
e2 = equation("eq2")
body("Stage 1 is a residual CNN (a 3×3 convolution, two residual blocks of 64 channels and a tanh-bounded output) that "
     "predicts the log-speckle and returns X̂ = exp(log Y − log F̂). No clean reflectivity is available, so Stage 1 is "
     "supervised by two properties of the speckle model:")
e3 = equation("eq3")
e4 = equation("eq4")
body(f"where *A*_{{w}} is a w×w moving average (w = 7). Because E[Y] = X, the local multi-look mean of Y is an unbiased "
     f"estimate of X, so ({e3}) forbids the despeckler from changing the radiometry. Term ({e4}) is a log-domain "
     f"total-variation prior {cite('rudin1992tv')} that penalises the high-frequency, additive log F.")
heading2("Stage 2: Colorization")
body(f"The colorizer is a U-Net {cite('ronneberger2015unet')}. Its encoder is the stem and first three stages of an "
     "ImageNet-pretrained ResNet-18; the despeckled SAR is replicated to three channels and normalised with ImageNet "
     "statistics. A 16×16 bottleneck feeds a decoder with bilinear upsampling (to avoid checkerboard artefacts) and "
     "spatial dropout with p = 0.15, 0.10 and 0.05. The output layer also receives X̂ through a skip connection.")
body(f"*Terrain FiLM.* A learned embedding of the terrain class c ∈ {{agriculture, barren, grassland, urban}} produces a "
     f"per-channel scale and shift (γ_{{c}}, β_{{c}}) {cite('perez2018film')} that modulates the bottleneck and three "
     "decoder levels as")
equation("eq5")
body("The embedding is zero-initialised, so training starts from the unconditioned network. A fifth, “unknown” embedding "
     "is trained by replacing the label with probability 0.2, so the model also runs without a label.")
body("*Loss.* The generator minimises")
equation("eq6")
body("where *L*_{grad} is the L1 distance between the image gradients of the prediction and the target, and *L*_{GAN} uses a "
     f"PatchGAN discriminator conditioned on the SAR input {cite('isola2017pix2pix')}. We use λ_{{1}} = 100, "
     "λ_{s} = 10, λ_{g} = 10, λ_{r} = 10 and λ_{tv} = 0.5. Our first version (v1) used λ_{a} = 1; the final model uses "
     "λ_{a} = 0, a choice made on the validation scenes (Section V-B).")
heading2("Stage 3: Optional Structure-Guided Fusion")
body("The prediction is converted to CIELAB. *Luminance replacement* sets")
e7 = equation("eq7")
body(f"where S is the percentile-stretched X̂. *Detail injection*, in the spirit of component-substitution pansharpening "
     f"{cite('vivone2015pansharp')}, adds only the high-pass part of the SAR:")
equation("eq8")
body("In both cases the chroma (a*, b*) is kept as predicted. The strengths α and β are chosen on the validation scenes.")
heading2("Epistemic Uncertainty")
body("At inference the batch-normalisation layers use their running statistics and only the decoder dropout is sampled, "
     f"over T = {N['mcT']} passes:")
equation("eq9")
body("with the variance averaged over the RGB channels.")

# IV. Experimental setup
heading1("Experimental Setup")
heading2("Data and Leakage-Free Split")
body("We use 16,000 co-registered Sentinel-1/Sentinel-2 pairs of 256×256 pixels derived from SEN12MS "
     f"{cite('schmitt2019sen12ms')}, covering four terrain classes and 30 scenes. Sentinel-1 is given as 8-bit backscatter "
     f"and Sentinel-2 as 8-bit RGB {cite('drusch2012s2')}; both are scaled to [0, 1]. Patches of one scene overlap and "
     "share appearance, so we split by *scene*, with stratification by class: 11,014 training tiles (18 scenes), 1,139 "
     "validation tiles (4 scenes) and 3,847 test tiles (8 scenes, all four classes). No scene appears in two splits, which "
     f"an assertion in the split script enforces. We evaluate a class-balanced subset of {N['nTest']} test tiles.")
heading2("Matched Training Budget")
body("Every model is trained with the same settings: 4,000 iterations, batch 8, random 128×128 crops with flips and 90° "
     "rotations, and Adam (2×10^{−4}, β_{1} = 0.5) with cosine decay, in fp32 with seed 42 on one 2 GB NVIDIA MX550 laptop "
     "GPU. Each model’s checkpoint and the final configuration are selected by validation L1. The test set is never used "
     "for any selection, including the fusion strengths and whether to supply the terrain label. The baselines are a "
     f"U-Net trained with L1 only, Pix2Pix {cite('isola2017pix2pix')} and a Lee filter {cite('lee1980filter')} (7×7, L = 40, "
     "the median ENL of the training scenes) followed by Pix2Pix. Two trivial floors are also reported: the SAR replicated "
     "as grey and the mean training colour. We did not retrain CycleGAN, which targets unpaired data, or transformer and "
     "diffusion models, which exceed our compute budget, and we do not copy their numbers from papers that use other data.")
heading2("Metrics")
body(f"We report PSNR, SSIM {cite('wang2004ssim')}, SAM {cite('kruse1993sam')} and GMSD {cite('xue2014gmsd')} (edge "
     "fidelity) on full 256×256 tiles. Despeckling is assessed without a reference by ENL, computed on the same "
     f"homogeneous blocks of the raw and filtered image (median over tiles), EPI {cite('sattar1997epi')} relative to the "
     "raw SAR, and the mean of the ratio image (MoR, ideal 1). Uncertainty is assessed with AUSE "
     f"{cite('ilg2018uncertainty')} and the Spearman correlation between error and uncertainty. Every comparison with our "
     "final model uses the paired Wilcoxon signed-rank test.")

# V. Results
heading1("Results")
heading2("Comparison with Baselines")
body(f"Table I and Fig. 2 summarise the test results. The final model reaches {N['psnrOursRBICondNoadv']} dB PSNR and "
     f"{N['ssimOursRBICondNoadv']} SSIM, which is {N['dFinalVsPix']} dB above Pix2Pix (p {N['pFinalPixPsnr']}) and "
     f"{N['dFinalVsUnet']} dB above the U-Net regressor (p = {N['pFinalUnetPsnr']}); SSIM is also higher (by "
     f"{N['dSsimFinalVsUnet']}, p {N['pFinalUnetSsim']}). The SAM difference from the U-Net is not significant "
     f"(p = {N['pFinalUnetSam']}). Absolute scores are low for every method: the final model is only {N['dFinalVsMean']} dB "
     "above predicting the mean training colour. This is the central practical finding: colours in unseen scenes depend "
     "on season, soil and vegetation state, and single-polarization backscatter alone does not determine them.")
body(f"SAM must be read together with PSNR and SSIM. The trivial grey and mean-colour predictions obtain the *lowest* SAM "
     f"({N['samSarGray']}° and {N['samMeanColor']}°), because Sentinel-2 RGB in these scenes is close to neutral and SAM "
     "rewards hue-neutral outputs regardless of their brightness.")

NAMES = {"sar_gray": "SAR as grey (floor)", "mean_color": "Mean colour (floor)", "unet_l1": "U-Net (L1)",
         "pix2pix": f"Pix2Pix {cite('isola2017pix2pix')}", "lee_pix2pix": "Lee + Pix2Pix", "ours": "Ours v1 (GAN)",
         "ours_noadv": "Ours w/o GAN", MAIN: "Ours (final)", MAIN + "@nolabel": "Ours (final), no label",
         "ours+lum": "Ours + lum. repl.", "ours+detail": "Ours + detail inj."}
table_caption(f"Test Results on {N['nTest']} Tiles from Scenes Unseen in Training. "
              "† = Not Significantly Different from Ours (Final), Wilcoxon p ≥ 0.05")
rows = []
for m in ["sar_gray", "mean_color", "unet_l1", "pix2pix", "lee_pix2pix", "ours", "ours_noadv", MAIN, MAIN + "@nolabel",
          "ours+lum", "ours+detail"]:
    s = S[m]
    rows.append([NAMES[m]] + [fmt(s[k], k) + ("^{†}" if m != MAIN and nsig(m, k) else "")
                               for k in ["psnr", "ssim", "sam", "gmsd"]])
data_table(["Method", "PSNR (dB)↑", "SSIM↑", "SAM (°)↓", "GMSD↓"], rows, [1.25, 0.6, 0.5, 0.55, 0.5],
           bold_rows=(7,))

new_section(1)
figure(os.path.join(FIG, "fig_comparison.png"), 5.5,
       "Test PSNR, SSIM and SAM for all methods. Blue: final model and its fused outputs; dark grey: baselines and "
       "ablations; light grey: trivial floors.")
ck, no = "x", "–"
abl = [("unet_l1", no, no, no, no, no, no), ("pix2pix", no, no, no, ck, no, no), ("ours_nodesp", no, no, ck, ck, no, no),
       ("ours_nophys", ck, no, ck, ck, no, no), ("ours", ck, ck, ck, ck, no, no), ("ours_cond", ck, ck, ck, ck, no, ck),
       ("ours_noadv", ck, ck, ck, no, no, no), (MAIN, ck, ck, ck, no, ck, ck)]
ABL_N = {"unet_l1": "U-Net (L1)", "pix2pix": "Pix2Pix", "ours_nodesp": "Ours v1 w/o despeckling stage",
         "ours_nophys": "Ours v1 w/o physics prior", "ours": "Ours v1", "ours_cond": "Ours v1 + terrain FiLM",
         "ours_noadv": "Ours w/o GAN", MAIN: "Ours (final)"}
table_caption("Ablation on the Test Set. S1: Despeckling Stage; *L*_{phys}: Physics Prior; *L*_{s,g}: SSIM and Gradient "
              "Losses; R18: Pretrained ResNet-18 Encoder; FiLM: Terrain Conditioning; MoR: Mean of Ratio of Stage 1 (Ideal 1)")
arows = []
for m, *f in abl:
    s = S[m]
    mor = f"{D[m]['mor']:.3f}" if m in D else "–"
    arows.append([ABL_N[m]] + f + [f"{s['psnr']:.2f}", f"{s['ssim']:.3f}", f"{s['sam']:.2f}", mor])
data_table(["Variant", "S1", "*L*_{phys}", "*L*_{s,g}", "GAN", "R18", "FiLM", "PSNR↑", "SSIM↑", "SAM↓", "MoR"], arows,
           [1.85, 0.4, 0.45, 0.45, 0.4, 0.4, 0.45, 0.55, 0.55, 0.55, 0.55], bold_rows=(7,))
new_section(2)

heading2("Ablation")
body(f"Table II isolates each component. *Adversarial loss.* Removing the GAN term from our model raises PSNR by "
     f"{N['dGanCost']} dB, and Pix2Pix trails the identical U-Net trained with L1 alone by {N['dPixVsUnet']} dB. On unseen "
     "scenes the discriminator pushes the generator towards vivid but wrong colours and creates artefacts (Fig. 3). This "
     f"is the perception–distortion trade-off {cite('blau2018perception')}, and it dominates every other design choice.")
body("*Physics prior.* Without *L*_{phys}, Stage 1 is shaped only by the colorization loss and develops a strong radiometric "
     f"bias (MoR {N['morOursNophys']}); with the prior the bias disappears (MoR {N['morOurs']}, Table III).")
body("*Pretrained encoder and FiLM.* Adding both to the non-adversarial model gives the best validation L1 and the highest "
     f"test SSIM, but its test PSNR ({N['psnrOursRBICondNoadv']} dB) is slightly below that of the non-adversarial model "
     f"without them ({N['psnrOursNoadv']} dB): the two additions mainly speed up and stabilise training (Fig. 6) rather "
     f"than raise accuracy. Moreover, supplying the true terrain label at test time is {N['dNoLabelVsFinal']} dB *worse* "
     f"than using the “unknown” embedding (p {N['pNoLabelPsnr']}), even though the label helped on the validation scenes. "
     "The class-specific palettes learned from 18 training scenes therefore do not transfer reliably to new scenes. We "
     "keep the validation-selected configuration (with label) as the final model and report the label-free result for "
     "transparency.")

heading2("Despeckling Quality")
body("Table III compares the Stage 1 outputs with the raw SAR and the Lee filter on identical homogeneous blocks. The Lee "
     f"filter smooths most strongly (ENL {N['enlLee']}). Our Stage 1 is a mild filter (ENL {N['enlOursRBICondNoadv']} "
     f"versus {N['enlRaw']} raw) that keeps edges (EPI {N['epiOursRBICondNoadv']}) and radiometry. The stage trained "
     f"without the prior *lowers* ENL to {N['enlOursNophys']}, i.e. it adds texture rather than removing speckle.")
table_caption("No-Reference Despeckling Metrics (Median over Test Tiles)")
DN = {"raw": "Raw SAR (no filter)", "lee": "Lee filter (7×7, L = 40)", "ours_nophys": "Stage 1, no physics prior",
      "ours": "Stage 1 + prior (v1)", "ours_noadv": "Stage 1 + prior, no GAN", MAIN: "Stage 1 + prior (final)"}
data_table(["Filter", "ENL↑", "EPI↑", "MoR (ideal 1)"],
           [[DN[k], f"{D[k]['enl']:.1f}", f"{D[k]['epi']:.3f}", f"{D[k]['mor']:.3f}"] for k in DN],
           [1.55, 0.55, 0.55, 0.75])

# full-width qualitative figure here, so the per-terrain table below is never split by column balancing
new_section(1)
figure(os.path.join(FIG, "fig_qualitative.png"), 6.2,
       "Qualitative test results, one tile per terrain class (all scenes unseen in training). “Ours stage 1” is the "
       "despeckled input of the final model. GAN-based methods hallucinate saturated colours and blob artefacts; the "
       "non-adversarial final model produces muted but terrain-plausible palettes and misses the lake in the grassland tile.")
new_section(2)

heading2("Per-Terrain Results")
CL = ["agri", "barrenland", "grassland", "urban"]
table_caption("Per-Terrain Test PSNR (dB) / SAM (°)")
TN = {"mean_color": "Mean colour", "unet_l1": "U-Net (L1)", "pix2pix": "Pix2Pix", "lee_pix2pix": "Lee + Pix2Pix",
      MAIN: "Ours (final)"}
data_table(["Method", "Agri.", "Barren", "Grass.", "Urban"],
           [[TN[m]] + [f"{PC[m][c]['psnr']:.2f} / {PC[m][c]['sam']:.1f}" for c in CL] for m in TN],
           [1.0, 0.62, 0.62, 0.62, 0.62], bold_rows=(4,))
body("Table IV breaks the results down by class. The final model is best on barren, grassland and urban PSNR, and the "
     "U-Net is best on agriculture. Grassland has the largest SAM for every method because one test scene contains a "
     "lake: SAR shows it as dark, smooth backscatter, but no model renders it as water (Fig. 3, third row).")

heading2("Uncertainty")
U = R["uncertainty"]
figure(os.path.join(FIG, "fig_sparsification.png"), 2.7,
       "Sparsification curves of the final model. A useful uncertainty would follow the oracle; MC-dropout variance does not.")
body(f"Fig. 4 shows sparsification curves for the final model (T = {N['mcT']}, {N['nMC']} tiles). Removing the most "
     "uncertain pixels does *not* lower the error of the remaining pixels: the model curve stays close to random removal "
     f"(AUSE {N['ause']}, Spearman ρ = {N['uncRho']}), and averaging the MC passes changes PSNR by only {N['mcGain']} dB. "
     "MC-dropout variance here reflects decoder stochasticity, not colour ambiguity, so it is not a hallucination guardrail.")

heading2("Structure-Guided Fusion")
body("Fig. 5 sweeps the fusion strength. Luminance replacement at the prototype setting α = 0.7 lowers PSNR by about 2.6 dB, "
     "because SAR brightness is not optical brightness. Detail injection improves edge fidelity (GMSD) at a much smaller "
     f"PSNR cost. Validation selects small strengths (α = {N['fusionAlpha']}, β = {N['fusionBeta']}), and neither fusion "
     "yields a meaningful fidelity gain over the raw network output (Table I). We therefore offer fusion only as a display option.")
figure(os.path.join(FIG, "fig_fusion.png"), 2.55, "Test PSNR and GMSD versus fusion strength (0 = no fusion).")

heading2("Training Dynamics and Cost")
body("Validation curves (Fig. 6) show that the GAN-based models fluctuate strongly between checkpoints, whereas the "
     "pretrained-encoder model is stable from the first 500 iterations. All generators have 8.3–9.6 M parameters; Stage 1 "
     "raises inference time from 15.0 to 31.5 ms per tile on an MX550.")
figure(os.path.join(FIG, "fig_training.png"), 2.55, "Validation PSNR during training (identical budget for all models).")

# VI. Limitations
heading1("Limitations")
body("(i) Sentinel-1 and Sentinel-2 acquisitions are not simultaneous, so pixel metrics are bounded by real scene change "
     "between them. (ii) The 8-bit PNG tiles are not calibrated σ^{0}, so the speckle model is applied to quantised "
     "backscatter. (iii) Only single-polarization input is used. (iv) The budget is small (4,000 iterations on a 2 GB "
     "GPU); absolute numbers would likely rise with longer training, and the ranking might change. (v) The test set has 8 "
     "scenes, so conclusions about transfer across scenes, in particular the effect of terrain conditioning, need more "
     "scenes. (vi) Colours are a learned estimate, not a measurement, and should not be used for quantitative spectral analysis.")

# VII. Conclusion
heading1("Conclusion and Future Work")
body("We presented a physics-guided two-stage SAR colorization network and evaluated it on a leakage-free, scene-level "
     "Sentinel-1/2 benchmark with matched-budget baselines. A despeckling stage supervised only by the speckle model "
     "preserves radiometry. A non-adversarial objective gives the best PSNR and SSIM and significantly outperforms Pix2Pix "
     "and a U-Net regressor, while a pretrained encoder mainly makes training faster and more stable. The absolute gains "
     "over trivial predictors are small, and terrain conditioning, MC-dropout uncertainty and SAR luminance fusion did not "
     "deliver the commonly assumed benefits; we consider these negative results as useful as the positive ones. Future "
     "work will add dual-polarization (VV+VH) and multi-temporal inputs, train longer on larger scene sets, test "
     "calibrated uncertainty methods (deep ensembles, conformal prediction), and "
     f"investigate conditional diffusion models {cite('rombach2022ldm')}, Swin Transformer encoders {cite('liu2021swin')} "
     "and perceptual metrics such as LPIPS.")

heading1("Acknowledgment", numbered=False)
body("The authors thank the Department of Artificial Intelligence and Data Science, Saveetha Engineering College, for "
     "guidance and support. Sentinel-1 and Sentinel-2 data are provided by the European Space Agency through the "
     "Copernicus programme.")

heading1("References", numbered=False)
for i, k in enumerate(ORDER, 1):
    p = para(WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0, after=0)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.3))
    set_font(p.add_run(f"[{i}]\t{REFS[k]}"), 8)

unused = sorted(set(REFS) - set(ORDER))
doc.core_properties.title = "Physics-Guided SAR-to-Optical Colorization with Radiometry-Preserving Despeckling"
doc.core_properties.author = "Selvanayaki S; K. Micheal Mahesh; Kolli Amos Daniel; Gedipudi Darshani; Yenuganti Prathyusha; P Keerthana"
doc.save(OUT)
print(f"saved {OUT}\n  {len(ORDER)} references cited, unused: {unused}\n  figures {FIGN[0]}, tables {TABN[0]}, equations {EQN[0]}")
