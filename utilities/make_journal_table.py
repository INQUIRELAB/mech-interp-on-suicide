#!/usr/bin/env python
"""Build research_docs/target_journals.xlsx from the Q1 journal shortlist."""
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

OUT = Path(__file__).parent.parent / "research_docs" / "target_journals.xlsx"

rows = [
    # Journal, Publisher, IF, Quartile, Category, Fit, Acceptance realism, Best framing
    ["Patterns", "Cell Press", 7.4, "Q1", "Data science",
     "Interdisciplinary; interpretability + ethics + applied", "Good (recommended primary)", "Findings"],
    ["Nature Machine Intelligence", "Nature", 18.4, "Q1", "AI / CS",
     "Broad AI + societal significance", "Reach (very selective)", "Findings"],
    ["TACL", "MIT Press / ACL", 4.0, "Q1", "Computational linguistics",
     "NLP interpretability/probing core; no length cap", "Good (MI fallback)", "Mechanistic"],
    ["Computational Linguistics", "MIT Press", 5.0, "Q1", "Computational linguistics",
     "Mechanistic interp published directly", "Good; slower", "Mechanistic"],
    ["npj Digital Medicine", "Nature", 15.1, "Q1", "Digital health",
     "High-impact digital health", "Reach (clinical)", "Clinical"],
    ["JAMIA", "Oxford / AMIA", 4.8, "Q1", "Medical informatics",
     "NLP for suicide/health; auditing LLMs", "Core clinical fit", "Clinical"],
    ["JMIR Mental Health", "JMIR", 5.8, "Q1", "Psychiatry / digital MH",
     "Digital mental health", "Realistic clinical home", "Clinical"],
    ["TMLR (note)", "OpenReview", None, "no JCR quartile", "ML",
     "MI community prestige; NOT Q1-indexed", "Excluded per Q1 requirement", "Mechanistic"],
]
headers = ["Journal", "Publisher", "Impact Factor", "Quartile", "Category",
           "Fit", "Acceptance realism", "Best-fit framing"]

wb = Workbook()
ws = wb.active
ws.title = "Q1 journal shortlist"

hdr_fill = PatternFill("solid", fgColor="1F4E78")
hdr_font = Font(bold=True, color="FFFFFF", size=11)
rec_fill = PatternFill("solid", fgColor="C6EFCE")   # green for recommended primary
thin = Side(style="thin", color="BFBFBF")
border = Border(left=thin, right=thin, top=thin, bottom=thin)

for c, h in enumerate(headers, 1):
    cell = ws.cell(row=1, column=c, value=h)
    cell.fill = hdr_fill; cell.font = hdr_font
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = border

for r, row in enumerate(rows, 2):
    for c, val in enumerate(row, 1):
        cell = ws.cell(row=r, column=c, value=val)
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        cell.border = border
    if "recommended primary" in str(row[6]):
        for c in range(1, len(headers) + 1):
            ws.cell(row=r, column=c).fill = rec_fill

widths = [26, 16, 13, 16, 22, 40, 26, 16]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[chr(64 + i)].width = w
ws.freeze_panes = "A2"
ws.auto_filter.ref = f"A1:H{len(rows) + 1}"

# --- notes sheet ---
ws2 = wb.create_sheet("Recommendation & notes")
notes = [
    ["Recommendation by framing", ""],
    ["Findings / methodological-caution (current draft)", "1) Patterns  2) Nature Machine Intelligence (reach)  3) TACL"],
    ["Mechanistic-interpretability method", "1) TACL  2) Computational Linguistics  3) Patterns"],
    ["Clinical / safety application", "1) JAMIA  2) npj Digital Medicine (reach)  3) JMIR Mental Health"],
    ["", ""],
    ["Cross-cutting submission notes", ""],
    ["Suicide-specific", "Safe-messaging required (we classify only, no means/methods); crisis-resource statement added"],
    ["IRB / ethics", "Hard gate at all clinical venues (JAMIA, npj DM, JMIR); 2 ethics blanks must be filled first"],
    ["Length", "TACL/CL no hard page cap (fits our appendices); NMI/Patterns short main text + supplementary"],
    ["House style", "NMI/Patterns/npj=Nature/Cell; JAMIA=Oxford; TACL/CL=MIT Press LaTeX (closest to current)"],
    ["Default pick", "Patterns (primary), TACL (mechanistic fallback), NMI (reach)"],
    ["Metrics as of", "2024/2025 JCR/Scopus, retrieved 2026-05-25"],
]
for r, (a, b) in enumerate(notes, 1):
    ca = ws2.cell(row=r, column=1, value=a); cb = ws2.cell(row=r, column=2, value=b)
    cb.alignment = Alignment(wrap_text=True, vertical="top")
    if b == "" and a:
        ca.font = Font(bold=True, size=11)
ws2.column_dimensions["A"].width = 42
ws2.column_dimensions["B"].width = 70

wb.save(OUT)
print(f"Saved {OUT}")
