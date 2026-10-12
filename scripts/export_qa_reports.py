"""Export the QA review and corrected-answer draft to readable PDFs.

Install with: python -m pip install -e '.[report]'
Run from the repository root: python scripts/export_qa_reports.py
"""
from html import escape
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle

ROOT = Path(__file__).resolve().parents[1]
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle("QAParagraph", parent=STYLES["BodyText"], fontSize=9, leading=12, spaceAfter=6))
STYLES.add(ParagraphStyle("QATable", parent=STYLES["BodyText"], fontSize=7.5, leading=9.5))


def inline(text):
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', text)
    return text


def footer(canvas, doc):
    canvas.setFont("Helvetica", 8)
    canvas.drawString(36, 22, "IDA117V | QA review draft | 6 October 2026")
    canvas.drawRightString(A4[0] - 36, 22, str(doc.page))


def export(source, destination):
    lines = source.read_text(encoding="utf-8").splitlines()
    story = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            columns = len(rows[0])
            weights = [max(4, min(40, max(len(row[c]) for row in rows) ** 0.6)) for c in range(columns)]
            widths = [(A4[0] - 72) * w / sum(weights) for w in weights]
            cells = [[Paragraph(inline(c), STYLES["QATable"]) for c in row] for row in rows]
            table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce8f2")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.extend([table, Spacer(1, 10)])
            continue
        if line.startswith("#"):
            count = len(line) - len(line.lstrip("#"))
            style = STYLES["Title"] if count == 1 else STYLES[f"Heading{min(count-1, 3)}"]
            story.append(Paragraph(inline(line.lstrip("# ")), style))
        else:
            story.append(Paragraph(inline(line), STYLES["QAParagraph"]))
        i += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=36, rightMargin=36,
                      topMargin=36, bottomMargin=36).build(story, onFirstPage=footer, onLaterPages=footer)
    print(destination.relative_to(ROOT))


if __name__ == "__main__":
    for name in ["QA_REVIEW", "QA_CORRECTED_ANSWERS"]:
        export(ROOT / "docs" / (name + ".md"), ROOT / "reports" / "qa" / (name + ".pdf"))
