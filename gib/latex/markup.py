"""A tiny document model rendered to both LaTeX and Word.

Blocks are tuples:
  ("h1"|"h2"|"h3", text)       ("p", text)          ("bullets", [text, ...])
  ("eq", latex, unicode)       ("pending", text)    ("table", Table)
  ("figure", filename, caption, label)   ("pagebreak",)

Inline markup inside text:
  **bold**  *italic*  [@key] (parenthetical cite)  [@@key] (textual cite)
  {{name}} (a value from the report, or a PENDING marker)
  <<latex|unicode>> (inline maths: LaTeX for the paper, Unicode for Word)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PENDING = "PENDING"
LABELS: Dict[str, int] = {}   # label -> number, filled by number_labels() before rendering Word


SCALE = {"fig_gallery.png": 0.72, "fig_attention.png": 0.78}  # keeps the body within the 8-page limit


def number_labels(blocks: Sequence[tuple]) -> None:
    LABELS.clear()
    nt = nf = 0
    for b in blocks:
        if b[0] == "table" and b[1].label:
            nt += 1
            LABELS[b[1].label] = nt
        elif b[0] == "table":
            nt += 1
        elif b[0] == "figure":
            nf += 1
            LABELS[b[3]] = nf

UNNUMBERED = {"Limitations", "Ethics Statement"}

_INLINE = re.compile(
    r"\[#(?P<ref>[\w:\-]+)\]|\*\*(?P<b>.+?)\*\*|\*(?P<i>[^*]+?)\*|\[@(?P<tc>@?)(?P<cite>[\w,:\-]+)\]|\{\{(?P<val>[\w.\-]+)\}\}|<<(?P<mtex>.+?)\|(?P<muni>.+?)>>")


@dataclass
class Table:
    caption: str
    header: List[str]
    rows: List[List[str]]
    label: str = ""
    note: str = ""
    wide: bool = False
    widths: Optional[List[float]] = None   # column widths in inches (Word)


@dataclass
class Bib:
    key: str
    authors: str       # "Rohrbach, Anna and Hendricks, Lisa Anne and ..."
    year: str
    title: str
    venue: str
    kind: str = "inproceedings"   # or article / book / misc
    extra: Dict[str, str] = field(default_factory=dict)

    def short(self) -> str:
        names = [a.split(",")[0].strip() for a in self.authors.split(" and ")]
        if len(names) == 1:
            who = names[0]
        elif len(names) == 2:
            who = f"{names[0]} and {names[1]}"
        else:
            who = f"{names[0]} et al."
        return f"{who}, {self.year}"

    def textual(self) -> str:
        who, year = self.short().rsplit(", ", 1)
        return f"{who} ({year})"

    def reference(self) -> str:
        names = [a.strip() for a in self.authors.split(" and ")]
        def fmt(n: str) -> str:
            if "," in n:
                last, first = [x.strip() for x in n.split(",", 1)]
                return f"{first} {last}"
            return n
        people = [fmt(n) for n in names]
        who = people[0] if len(people) == 1 else ", ".join(people[:-1]) + " and " + people[-1]
        return f"{who}. {self.year}. {self.title}. {self.venue}."

    def bibtex(self) -> str:
        venue_field = {"inproceedings": "booktitle", "article": "journal", "book": "publisher",
                       "misc": "howpublished"}[self.kind]
        fields = {"author": self.authors, "title": "{" + self.title + "}", venue_field: self.venue,
                  "year": self.year, **self.extra}
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
        return f"@{self.kind}{{{self.key},\n{body}\n}}"


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
_TEX_UNI = {"σ": r"$\sigma$", "×": r"$\times$", "≥": r"$\geq$", "≤": r"$\leq$", "−": r"$-$", "→": r"$\rightarrow$",
            "≈": r"$\approx$", "χ": r"$\chi$", "²": r"$^2$", "κ": r"$\kappa$", "η": r"$\eta$", "±": r"$\pm$",
            "∅": r"$\varnothing$", "⟨": r"$\langle$", "⟩": r"$\rangle$", "≠": r"$\neq$", "·": r"$\cdot$",
            "Δ": r"$\Delta$", "∈": r"$\in$", "↔": r"$\leftrightarrow$", "∝": r"$\propto$", "†": r"$^\dagger$",
            "‡": r"$^\ddagger$"}


def tex_escape(s: str) -> str:
    out = []
    for ch in s:
        out.append(_TEX_ESC.get(ch) or _TEX_UNI.get(ch) or ch)
    return "".join(out)


def tex_inline(text: str, values: Dict[str, Any]) -> str:
    out, pos = [], 0
    for m in _INLINE.finditer(text):
        out.append(tex_escape(text[pos:m.start()]))
        if m.group("ref") is not None:
            out.append(r"\ref{" + m.group("ref") + "}")
        elif m.group("b") is not None:
            out.append(r"\textbf{" + tex_inline(m.group("b"), values) + "}")
        elif m.group("i") is not None:
            out.append(r"\emph{" + tex_inline(m.group("i"), values) + "}")
        elif m.group("cite") is not None:
            out.append((r"\citet{" if m.group("tc") else r"\citep{") + m.group("cite") + "}")
        elif m.group("val") is not None:
            v = values.get(m.group("val"))
            k = m.group("val")
            if v is not None and k.startswith("gib"):
                out.append("\\" + k + "{}")  # a macro from the audited macros.tex
            else:
                out.append(tex_escape(str(v)) if v is not None else r"\pending{" + tex_escape(k) + "}")
        else:
            out.append("$" + m.group("mtex") + "$")
        pos = m.end()
    out.append(tex_escape(text[pos:]))
    return "".join(out)


def render_tex(blocks: Sequence[tuple], values: Dict[str, Any]) -> str:
    L: List[str] = []
    for b in blocks:
        kind = b[0]
        if kind in ("h1", "h2", "h3"):
            cmd = {"h1": "section", "h2": "subsection", "h3": "paragraph"}[kind]
            star = "*" if b[1] in UNNUMBERED else ""
            L.append(f"\\{cmd}{star}{{{tex_inline(b[1], values)}}}")
        elif kind == "p":
            L.append(tex_inline(b[1], values) + "\n")
        elif kind == "bullets":
            L.append("\\begin{itemize}[leftmargin=*,nosep]")
            L += [f"  \\item {tex_inline(t, values)}" for t in b[1]]
            L.append("\\end{itemize}\n")
        elif kind == "eq":
            L.append("\\begin{equation}\n" + b[1] + "\n\\end{equation}\n")
        elif kind == "pending":
            L.append("\\pendingbox{" + tex_inline(b[1], values) + "}\n")
        elif kind == "table":
            t: Table = b[1]
            env = "table*" if t.wide else "table"
            cols = "l" + "c" * (len(t.header) - 1)
            L.append(f"\\begin{{{env}}}[t]\n\\centering\\small\n\\resizebox{{\\{'textwidth' if t.wide else 'columnwidth'}}}{{!}}{{%")
            L.append(f"\\begin{{tabular}}{{{cols}}}\n\\toprule")
            L.append(" & ".join(tex_inline(h, values) for h in t.header) + r" \\ \midrule")
            for r in t.rows:
                if len(r) == 1:
                    L.append(f"\\multicolumn{{{len(t.header)}}}{{c}}{{\\pending{{{tex_inline(r[0], values)}}}}} \\\\")
                else:
                    L.append(" & ".join(tex_inline(c, values) for c in r) + r" \\")
            L.append("\\bottomrule\n\\end{tabular}}")
            cap = tex_inline(t.caption, values) + ((" " + tex_inline(t.note, values)) if t.note else "")
            L.append(f"\\caption{{{cap}}}" + (f"\\label{{{t.label}}}" if t.label else ""))
            L.append(f"\\end{{{env}}}\n")
        elif kind == "figure":
            _, fname, cap, label = b[:4]
            wide = len(b) > 4 and b[4]
            env = "figure*" if wide else "figure"
            L.append(f"\\begin{{{env}}}[t]\n\\centering")
            L.append(f"\\IfFileExists{{figures/{fname}}}{{\\includegraphics[width={SCALE.get(fname, 1.0)}\\{'textwidth' if wide else 'columnwidth'}]{{figures/{fname}}}}}"
                     f"{{\\pendingbox{{Figure pending: run \\texttt{{gib report}}}}}}")
            L.append(f"\\caption{{{tex_inline(cap, values)}}}\\label{{{label}}}\n\\end{{{env}}}\n")
        elif kind == "pagebreak":
            L.append("\\clearpage")
        elif kind == "appendix":
            L.append("\\appendix")
        elif kind == "references":
            L.append("\\bibliographystyle{ieeenat_fullname}\n\\bibliography{references}\n")
        elif kind == "raw_tex":
            L.append(b[1])
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Word
# --------------------------------------------------------------------------- #
def docx_inline(par, text: str, values: Dict[str, Any], bib: Dict[str, Bib], bold=False, italic=False, size=None) -> None:
    from docx.enum.text import WD_COLOR_INDEX

    def run(t: str, b=False, i=False, pending=False):
        r = par.add_run(t)
        r.bold = bold or b
        r.italic = italic or i
        if size:
            from docx.shared import Pt
            r.font.size = Pt(size)
        if pending:
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            r.bold = True
        return r

    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            run(text[pos:m.start()])
        if m.group("ref") is not None:
            run(str(LABELS.get(m.group("ref"), "?")))
        elif m.group("b") is not None:
            docx_inline(par, m.group("b"), values, bib, bold=True, italic=italic, size=size)
        elif m.group("i") is not None:
            docx_inline(par, m.group("i"), values, bib, bold=bold, italic=True, size=size)
        elif m.group("cite") is not None:
            keys = m.group("cite").split(",")
            if m.group("tc"):
                run("; ".join(bib[k].textual() for k in keys))
            else:
                run("(" + "; ".join(bib[k].short() for k in keys) + ")")
        elif m.group("val") is not None:
            v = values.get(m.group("val"))
            if v is None:
                run(f"[{PENDING}: {m.group('val')}]", pending=True)
            else:
                run(str(v))
        else:
            run(m.group("muni"), i=True)
        pos = m.end()
    if pos < len(text):
        run(text[pos:])


def render_docx(doc, blocks: Sequence[tuple], values: Dict[str, Any], bib: Dict[str, Bib],
                figure_dir: Optional[Path] = None) -> None:
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor

    number_labels(blocks)
    eq_no = 0
    tab_no = 0
    fig_no = 0
    sec = sub = 0
    appendix = False
    for b in blocks:
        kind = b[0]
        if kind in ("h1", "h2", "h3"):
            level = {"h1": 1, "h2": 2, "h3": 3}[kind]
            prefix = ""
            if kind == "h1" and b[1] not in UNNUMBERED:
                sec += 1
                sub = 0
                prefix = (chr(ord("A") + sec - 1) if appendix else str(sec)) + "  "
            elif kind == "h2":
                sub += 1
                prefix = f"{chr(ord('A') + sec - 1) if appendix else sec}.{sub}  "
            h = doc.add_heading(level=level)
            docx_inline(h, prefix + b[1], values, bib)
        elif kind == "p":
            docx_inline(doc.add_paragraph(), b[1], values, bib)
        elif kind == "bullets":
            for t in b[1]:
                docx_inline(doc.add_paragraph(style="List Bullet"), t, values, bib)
        elif kind == "eq":
            eq_no += 1
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(b[2])
            r.italic = True
            p.add_run(f"\t({eq_no})")
        elif kind == "pending":
            p = doc.add_paragraph()
            r = p.add_run(f"[{PENDING}] ")
            r.bold = True
            r.font.color.rgb = RGBColor(0xB0, 0x2A, 0x37)
            docx_inline(p, b[1], values, bib, italic=True)
            shd = OxmlElement("w:shd")
            shd.set(qn("w:val"), "clear")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:fill"), "FFF3CD")
            p.paragraph_format.element.get_or_add_pPr().append(shd)
        elif kind == "table":
            t: Table = b[1]
            tab_no += 1
            cap = doc.add_paragraph()
            r = cap.add_run(f"Table {tab_no}: ")
            r.bold = True
            docx_inline(cap, t.caption + ((" " + t.note) if t.note else ""), values, bib, size=9)
            table = doc.add_table(rows=1, cols=len(t.header))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            if t.widths:
                table.autofit = False
            for c, h in zip(table.rows[0].cells, t.header):
                c.text = ""
                docx_inline(c.paragraphs[0], h, values, bib, bold=True, size=8)
            for row in t.rows:
                cells = table.add_row().cells
                if len(row) == 1:
                    merged = cells[0].merge(cells[-1])
                    merged.text = ""
                    docx_inline(merged.paragraphs[0], f"[{PENDING}] " + row[0], values, bib, italic=True, size=8)
                else:
                    for c, v in zip(cells, row):
                        c.text = ""
                        docx_inline(c.paragraphs[0], v, values, bib, size=8)
            if t.widths:
                for r_ in table.rows:
                    for c, w in zip(r_.cells, t.widths):
                        c.width = Inches(w)
            doc.add_paragraph()
        elif kind == "figure":
            _, fname, cap, _label = b[:4]
            fig_no += 1
            path = (figure_dir / fname) if figure_dir else None
            if path is not None and path.exists():
                doc.add_picture(str(path), width=Inches(6.0 if (len(b) > 4 and b[4]) else 4.2))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                p = doc.add_paragraph()
                r = p.add_run(f"[{PENDING}] figure {fname} is created by `gib report` from your real runs.")
                r.italic = True
                r.font.highlight_color = 7  # yellow
            c = doc.add_paragraph()
            r = c.add_run(f"Figure {fig_no}: ")
            r.bold = True
            docx_inline(c, cap, values, bib, size=9)
        elif kind == "pagebreak":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif kind == "appendix":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
            appendix, sec = True, 0
        elif kind == "references":
            doc.add_heading("References", level=1)
            for key in sorted(bib, key=lambda k: (bib[k].authors.split(",")[0], bib[k].year)):
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.first_line_indent = Inches(-0.3)
                r = p.add_run(bib[key].reference())
                r.font.size = Pt(9)
