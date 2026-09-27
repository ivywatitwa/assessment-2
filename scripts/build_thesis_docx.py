#!/usr/bin/env python3
"""Build a formatted DOCX from the verified Markdown dissertation draft.

The project environment deliberately avoids a hard dependency on a GUI editor.
This writer uses the DOCX Open XML package directly, so the result is reproducible
with Python's standard library. It supports the Markdown subset used by the draft:
headings, paragraphs, bullets, numbered lists, fenced code, pipe tables and SVG
figure assets.
"""
from __future__ import annotations

import argparse
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PR = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
DC = "http://purl.org/dc/elements/1.1/"
DCTERMS = "http://purl.org/dc/terms/"
CP = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"

ET.register_namespace("w", W)
ET.register_namespace("r", R)
ET.register_namespace("pr", PR)
ET.register_namespace("ct", CT)
ET.register_namespace("dc", DC)
ET.register_namespace("dcterms", DCTERMS)
ET.register_namespace("cp", CP)
ET.register_namespace("a", A)
ET.register_namespace("wp", WP)
ET.register_namespace("pic", PIC)


def tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def element(namespace: str, name: str, **attrs: str) -> ET.Element:
    return ET.Element(tag(namespace, name), {tag(W, key): value for key, value in attrs.items()})


def child(parent: ET.Element, namespace: str, name: str, **attrs: str) -> ET.Element:
    node = element(namespace, name, **attrs)
    parent.append(node)
    return node


def foreign_child(parent: ET.Element, namespace: str, name: str, attrs: dict[str, str] | None = None) -> ET.Element:
    """Create a drawing element whose attributes are not in the Word namespace."""
    node = ET.SubElement(parent, tag(namespace, name), attrs or {})
    return node


def add_text_run(paragraph: ET.Element, text: str, *, bold: bool = False, italic: bool = False,
                 font: str | None = None, size: int | None = None) -> None:
    if not text:
        return
    run = child(paragraph, W, "r")
    props = child(run, W, "rPr")
    if bold:
        child(props, W, "b")
    if italic:
        child(props, W, "i")
    if font:
        child(props, W, "rFonts", ascii=font, hAnsi=font, eastAsia=font)
    if size:
        child(props, W, "sz", val=str(size))
        child(props, W, "szCs", val=str(size))
    text_node = child(run, W, "t")
    if text[:1].isspace() or text[-1:].isspace():
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text


INLINE_RE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_inline(paragraph: ET.Element, text: str) -> None:
    """Render a small, deterministic subset of Markdown inline syntax."""
    cursor = 0
    for match in INLINE_RE.finditer(text):
        add_text_run(paragraph, text[cursor:match.start()])
        token = match.group(0)
        if token.startswith("**"):
            add_text_run(paragraph, token[2:-2], bold=True)
        elif token.startswith("*"):
            add_text_run(paragraph, token[1:-1], italic=True)
        else:
            add_text_run(paragraph, token[1:-1], font="Courier New", size=20)
        cursor = match.end()
    add_text_run(paragraph, text[cursor:])


def paragraph(document: ET.Element, text: str = "", style: str = "Normal") -> ET.Element:
    p = child(document, W, "p")
    props = child(p, W, "pPr")
    child(props, W, "pStyle", val=style)
    add_inline(p, text)
    return p


def centered_paragraph(document: ET.Element, text: str, style: str) -> ET.Element:
    p = paragraph(document, text, style)
    child(p.find(tag(W, "pPr")), W, "jc", val="center")
    return p


def page_break(document: ET.Element) -> None:
    p = child(document, W, "p")
    run = child(p, W, "r")
    child(run, W, "br", type="page")


def add_svg_image(document: ET.Element, image_path: Path, rel_id: str, image_id: int) -> None:
    """Add an SVG as a centered Word drawing using a package relationship."""
    svg = image_path.read_text(encoding="utf-8")
    width_match = re.search(r'\bwidth="([0-9.]+)"', svg)
    height_match = re.search(r'\bheight="([0-9.]+)"', svg)
    svg_width = float(width_match.group(1)) if width_match else 1200.0
    svg_height = float(height_match.group(1)) if height_match else 700.0
    max_width = 5943600  # 6.5 inches, within the dissertation page margins.
    width_emu = max_width
    height_emu = int(max_width * svg_height / svg_width)

    paragraph_node = paragraph(document, style="FigureImage")
    p_props = paragraph_node.find(tag(W, "pPr"))
    child(p_props, W, "jc", val="center")
    run = child(paragraph_node, W, "r")
    drawing = child(run, W, "drawing")
    inline = foreign_child(
        drawing,
        WP,
        "inline",
        {"distT": "0", "distB": "0", "distL": "0", "distR": "0"},
    )
    foreign_child(inline, WP, "extent", {"cx": str(width_emu), "cy": str(height_emu)})
    foreign_child(inline, WP, "effectExtent", {"l": "0", "t": "0", "r": "0", "b": "0"})
    foreign_child(inline, WP, "docPr", {"id": str(image_id), "name": image_path.name})
    foreign_child(inline, WP, "cNvGraphicFramePr")
    graphic = foreign_child(inline, A, "graphic")
    graphic_data = foreign_child(
        graphic,
        A,
        "graphicData",
        {"uri": "http://schemas.openxmlformats.org/drawingml/2006/picture"},
    )
    picture = foreign_child(graphic_data, PIC, "pic")
    nv_picture = foreign_child(picture, PIC, "nvPicPr")
    foreign_child(nv_picture, PIC, "cNvPr", {"id": str(image_id), "name": image_path.name})
    foreign_child(nv_picture, PIC, "cNvPicPr")
    blip_fill = foreign_child(picture, PIC, "blipFill")
    blip = foreign_child(blip_fill, A, "blip", {tag(R, "embed"): rel_id})
    foreign_child(blip, A, "extLst")
    stretch = foreign_child(blip_fill, A, "stretch")
    foreign_child(stretch, A, "fillRect")
    shape_properties = foreign_child(picture, PIC, "spPr")
    transform = foreign_child(shape_properties, A, "xfrm")
    foreign_child(transform, A, "off", {"x": "0", "y": "0"})
    foreign_child(transform, A, "ext", {"cx": str(width_emu), "cy": str(height_emu)})
    preset = foreign_child(shape_properties, A, "prstGeom", {"prst": "rect"})
    foreign_child(preset, A, "avLst")


def field(paragraph_node: ET.Element, instruction: str, fallback: str) -> None:
    simple = child(paragraph_node, W, "fldSimple", instr=instruction)
    run = child(simple, W, "r")
    text = child(run, W, "t")
    text.text = fallback


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        rows.append(cells)
    width = max((len(row) for row in rows), default=1)
    return [row + [""] * (width - len(row)) for row in rows]


def add_table(document: ET.Element, rows: list[list[str]], number: int) -> None:
    if not rows:
        return
    captions = {
        1: "Table 1. Controlled model configurations.",
        2: "Table 2. Source and exclusion audit.",
        3: "Table 3. Retrieval smoke result.",
        4: "Table 4. Public T1 model validation and test results.",
        5: "Table 5. Evidence and reproducibility map.",
    }
    paragraph(document, captions.get(number, f"Table {number}. Results table."), "Caption")
    table = child(document, W, "tbl")
    props = child(table, W, "tblPr")
    child(props, W, "tblStyle", val="TableGrid")
    child(props, W, "tblW", w="9000", type="dxa")
    child(props, W, "tblLayout", type="fixed")
    grid = child(table, W, "tblGrid")
    widths = {
        1: (2900, 2900, 3200),
        2: (2600, 1800, 4600),
        3: (950, 2800, 1700, 1950, 1600),
        4: (1200, 850, 750, 1050, 1050, 1900, 2200),
        5: (3150, 5850),
    }.get(number, tuple([9000 // len(rows[0])] * len(rows[0])))
    for width in widths:
        child(grid, W, "gridCol", w=str(width))
    for row_index, row in enumerate(rows):
        tr = child(table, W, "tr")
        tr_props = child(tr, W, "trPr")
        child(tr_props, W, "cantSplit")
        if row_index == 0:
            child(tr_props, W, "tblHeader", val="true")
        for column, cell_text in enumerate(row):
            tc = child(tr, W, "tc")
            tc_props = child(tc, W, "tcPr")
            child(tc_props, W, "tcW", w=str(widths[column]), type="dxa")
            child(tc_props, W, "shd", fill="E8F0F7" if row_index == 0 else ("F5F8FC" if row_index % 2 == 0 else "FFFFFF"))
            child(tc_props, W, "vAlign", val="center")
            p = child(tc, W, "p")
            p_props = child(p, W, "pPr")
            child(p_props, W, "pStyle", val="TableText")
            if row_index == 0:
                add_text_run(p, cell_text, bold=True, size=20)
            else:
                add_inline(p, cell_text)


def add_numbered_or_bullet(document: ET.Element, text: str, style: str) -> None:
    p = paragraph(document, text, style)
    props = p.find(tag(W, "pPr"))
    num_props = child(props, W, "numPr")
    child(num_props, W, "ilvl", val="0")
    child(num_props, W, "numId", val="1" if style == "List Bullet" else "2")


def build_document(markdown: str, asset_root: Path) -> tuple[ET.Element, int, int, list[Path]]:
    body = ET.Element(tag(W, "body"))
    lines = markdown.splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    table_number = 0
    figure_number = 0
    first_heading = True
    inserted_preliminaries = False
    in_references = False
    title_metadata = True
    paragraph_lines: list[str] = []
    image_paths: list[Path] = []

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            text = " ".join(line.strip() for line in paragraph_lines)
            if title_metadata and text.startswith("**"):
                centered_paragraph(body, text, "TitleMeta")
            else:
                paragraph(body, text, "Reference" if in_references else "Normal")
            paragraph_lines = []

    while index < len(lines):
        line = lines[index]
        if line.strip().startswith("```"):
            flush_paragraph()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                for code_line in code_lines:
                    paragraph(body, code_line, "Code")
                in_code = False
                code_lines = []
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if line.startswith(":::figure "):
            flush_paragraph()
            figure_number += 1
            caption = line[len(":::figure "):]
            if not caption.startswith(f"Figure {figure_number}. "):
                raise ValueError(f"Figure caption must start with Figure {figure_number}. ")
            index += 1
            while index < len(lines) and lines[index] != ":::endfigure":
                if lines[index].startswith(":::image "):
                    image_path = (asset_root / lines[index][len(":::image "):].strip()).resolve()
                    if not image_path.exists() or image_path.suffix.lower() != ".svg":
                        raise ValueError(f"Figure image does not exist or is not SVG: {image_path}")
                    image_paths.append(image_path)
                    add_svg_image(body, image_path, f"rId{len(image_paths) + 2}", len(image_paths))
                elif lines[index].strip():
                    paragraph(body, lines[index], "FigureText")
                index += 1
            if index == len(lines):
                raise ValueError(f"Unclosed Figure {figure_number}")
            paragraph(body, caption, "FigureCaption")
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2)
            if first_heading:
                centered_paragraph(body, text, "Title")
                first_heading = False
            else:
                if title_metadata and text.startswith("A Provenance-Aware"):
                    centered_paragraph(body, text, "Subtitle")
                    index += 1
                    continue
                if text == "Declaration of scope" and not inserted_preliminaries:
                    page_break(body)
                    paragraph(body, "Contents", "Heading 1")
                    for entry in (
                        "Declaration of scope", "Acknowledgements", "Abstract",
                        "1. Introduction", "2. Literature Review", "3. Methodology",
                        "4. Implementation", "5. Findings", "6. Discussion",
                        "7. Conclusion", "References", "Appendix A: Evidence and reproducibility map",
                        "Appendix B: Interpretation limits",
                    ):
                        paragraph(body, entry, "ContentsEntry")
                    paragraph(body, "List of tables and figures", "Heading 2")
                    for caption in (
                        "Table 1. Controlled model configurations (Section 3.7)",
                        "Table 2. Source and exclusion audit (Section 5.2)",
                        "Table 3. Retrieval smoke result (Section 5.4)",
                        "Table 4. Public T1 model validation and test results (Section 5.5)",
                        "Table 5. Evidence and reproducibility map (Appendix A)",
                        "Figure 1. Implemented pipeline and evidence boundaries (Section 3.1)",
                        "Figure 2. Public Tryp image flow and grouped split (Section 3.5)",
                        "Figure 3. MedGemma held-out test confusion matrix (Section 5.5)",
                    ):
                        paragraph(body, caption, "ContentsEntry")
                    paragraph(body, "List of abbreviations", "Heading 2")
                    paragraph(body, "AI — artificial intelligence; C&S — culture and sensitivity; ETL — extract, transform, load; KB — knowledge base; MRR — mean reciprocal rank; QLoRA — quantised low-rank adaptation; RAG — retrieval-augmented generation; SHAP — Shapley additive explanations; T1 — microscopy classification task; T2 — treatment recommendation task; XAI — explainable artificial intelligence.")
                    page_break(body)
                    inserted_preliminaries = True
                    title_metadata = False
                if text == "References" or re.match(r"^[1-7]\. ", text) or text.startswith("Appendix "):
                    page_break(body)
                if text == "References":
                    in_references = True
                elif text.startswith("Appendix "):
                    in_references = False
                style = {1: "Heading 1", 2: "Heading 2", 3: "Heading 3"}.get(level, "Heading 3")
                paragraph(body, text, style)
            index += 1
            continue

        if line.startswith("|"):
            flush_paragraph()
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            table_number += 1
            add_table(body, parse_table(table_lines), table_number)
            continue

        if line.startswith("> "):
            flush_paragraph()
            paragraph(body, line[2:], "Quote")
            index += 1
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            add_numbered_or_bullet(body, (bullet or numbered).group(1), "List Bullet" if bullet else "List Number")
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
        else:
            paragraph_lines.append(line)
        index += 1

    flush_paragraph()
    sect = child(body, W, "sectPr")
    child(sect, W, "pgSz", w="11906", h="16838")
    child(sect, W, "pgMar", top="1440", right="1440", bottom="1440", left="1440", header="720", footer="720", gutter="0")
    footer_reference = child(sect, W, "footerReference", type="default")
    footer_reference.set(tag(R, "id"), "rId2")
    child(sect, W, "titlePg")
    return body, table_number, figure_number, image_paths


def styles_xml() -> bytes:
    styles = ET.Element(tag(W, "styles"))
    def style(style_id: str, name: str, based_on: str = "Normal", size: str = "24", bold: bool = False,
              italic: bool = False, color: str | None = None, space_after: str = "160",
              keep_next: bool = False) -> ET.Element:
        node = child(styles, W, "style", type="paragraph", styleId=style_id)
        child(node, W, "name", val=name)
        child(node, W, "basedOn", val=based_on)
        ppr = child(node, W, "pPr")
        child(ppr, W, "spacing", after=space_after, line="360", lineRule="auto")
        if keep_next:
            child(ppr, W, "keepNext")
        rpr = child(node, W, "rPr")
        child(rpr, W, "rFonts", ascii="Times New Roman", hAnsi="Times New Roman", eastAsia="Times New Roman")
        child(rpr, W, "sz", val=size)
        child(rpr, W, "szCs", val=size)
        if bold:
            child(rpr, W, "b")
        if italic:
            child(rpr, W, "i")
        if color:
            child(rpr, W, "color", val=color)
        return node
    style("Normal", "Normal", size="24", space_after="160")
    style("Title", "Title", size="36", bold=True, color="17365D", space_after="300", keep_next=True)
    style("Subtitle", "Subtitle", size="28", italic=True, space_after="540")
    style("TitleMeta", "Title Metadata", size="24", space_after="260")
    style("ContentsEntry", "Contents Entry", size="23", space_after="60")
    style("Heading 1", "Heading 1", size="30", bold=True, color="1F4E79", space_after="180", keep_next=True)
    style("Heading 2", "Heading 2", size="26", bold=True, color="2F5597", space_after="140", keep_next=True)
    style("Heading 3", "Heading 3", size="24", bold=True, color="365F91", space_after="100", keep_next=True)
    style("Caption", "Caption", size="20", italic=True, color="365F91", space_after="80", keep_next=True)
    style("FigureCaption", "Figure Caption", size="20", italic=True, color="365F91", space_after="180")
    figure_style = style("FigureText", "Figure Text", size="19", space_after="0", keep_next=True)
    child(figure_style.find(tag(W, "pPr")), W, "shd", fill="F0F5FA")
    child(figure_style.find(tag(W, "pPr")), W, "ind", left="320", right="320")
    fonts = figure_style.find(f"{tag(W, 'rPr')}/{tag(W, 'rFonts')}")
    for face in ("ascii", "hAnsi", "eastAsia"):
        fonts.set(tag(W, face), "Courier New")
    style("FigureImage", "Figure Image", size="20", space_after="80", keep_next=True)
    style("Code", "Code", size="19", space_after="20")
    style("TableText", "Table Text", size="20", space_after="20")
    quote = style("Quote", "Quote", size="24", italic=True, color="17365D", space_after="160")
    child(quote.find(tag(W, "pPr")), W, "ind", left="600", right="600")
    reference = style("Reference", "Reference", size="23", space_after="120")
    child(reference.find(tag(W, "pPr")), W, "ind", left="720", hanging="720")
    style("List Bullet", "List Bullet", size="24", space_after="60")
    style("List Number", "List Number", size="24", space_after="60")
    return ET.tostring(styles, encoding="utf-8", xml_declaration=True)


def numbering_xml() -> bytes:
    numbering = ET.Element(tag(W, "numbering"))
    for num_id, fmt, text in (("1", "bullet", "•"), ("2", "decimal", "%1.")):
        abstract = child(numbering, W, "abstractNum", abstractNumId=num_id)
        lvl = child(abstract, W, "lvl", ilvl="0")
        child(lvl, W, "start", val="1")
        child(lvl, W, "numFmt", val=fmt)
        child(lvl, W, "lvlText", val=text)
        child(lvl, W, "lvlJc", val="left")
        ppr = child(lvl, W, "pPr")
        child(ppr, W, "ind", left="720", hanging="360")
        num = child(numbering, W, "num", numId=num_id)
        child(num, W, "abstractNumId", val=num_id)
    return ET.tostring(numbering, encoding="utf-8", xml_declaration=True)


def footer_xml() -> bytes:
    footer = ET.Element(tag(W, "ftr"))
    p = child(footer, W, "p")
    ppr = child(p, W, "pPr")
    child(ppr, W, "jc", val="center")
    add_text_run(p, "UEL-CN-7000 Dissertation  |  ")
    run = child(p, W, "r")
    begin = child(run, W, "fldChar", fldCharType="begin")
    _ = begin
    instr_run = child(p, W, "r")
    instr = child(instr_run, W, "instrText")
    instr.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    instr.text = " PAGE "
    end_run = child(p, W, "r")
    child(end_run, W, "fldChar", fldCharType="end")
    return ET.tostring(footer, encoding="utf-8", xml_declaration=True)


def settings_xml() -> bytes:
    settings = ET.Element(tag(W, "settings"))
    child(settings, W, "updateFields", val="true")
    child(settings, W, "defaultTabStop", val="720")
    return ET.tostring(settings, encoding="utf-8", xml_declaration=True)


def document_rels(image_paths: list[Path]) -> bytes:
    rels = ET.Element(tag(PR, "Relationships"))
    ET.SubElement(rels, tag(PR, "Relationship"), {
        "Id": "rId2", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer",
        "Target": "word/footer1.xml",
    })
    for index, image_path in enumerate(image_paths, 3):
        ET.SubElement(rels, tag(PR, "Relationship"), {
            "Id": f"rId{index}",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            "Target": f"media/{image_path.name}",
        })
    return ET.tostring(rels, encoding="utf-8", xml_declaration=True)


def content_types() -> bytes:
    types = ET.Element(tag(CT, "Types"))
    ET.SubElement(types, tag(CT, "Default"), {"Extension": "rels", "ContentType": "application/vnd.openxmlformats-package.relationships+xml"})
    ET.SubElement(types, tag(CT, "Default"), {"Extension": "xml", "ContentType": "application/xml"})
    ET.SubElement(types, tag(CT, "Default"), {"Extension": "svg", "ContentType": "image/svg+xml"})
    overrides = {
        "/word/document.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        "/word/styles.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml",
        "/word/numbering.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml",
        "/word/settings.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml",
        "/word/footer1.xml": "application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml",
        "/docProps/core.xml": "application/vnd.openxmlformats-package.core-properties+xml",
    }
    for part, content_type in overrides.items():
        ET.SubElement(types, tag(CT, "Override"), {"PartName": part, "ContentType": content_type})
    return ET.tostring(types, encoding="utf-8", xml_declaration=True)


def core_properties() -> bytes:
    core = ET.Element(tag(CP, "coreProperties"))
    title = child(core, DC, "title")
    title.text = "Fine-Tuning MedGemma for Explainable Veterinary Decision Support"
    creator = child(core, DC, "creator")
    creator.text = "Ivy Watitwa"
    subject = child(core, DC, "subject")
    subject.text = "UEL-CN-7000 Assessment 2 Dissertation"
    created = child(core, DCTERMS, "created")
    created.set(tag(DCTERMS, "W3CDTF"), "2026-09-20T00:00:00Z")
    modified = child(core, DCTERMS, "modified")
    modified.set(tag(DCTERMS, "W3CDTF"), datetime.now(timezone.utc).isoformat())
    return ET.tostring(core, encoding="utf-8", xml_declaration=True)


def root_rels() -> bytes:
    rels = ET.Element(tag(PR, "Relationships"))
    ET.SubElement(rels, tag(PR, "Relationship"), {
        "Id": "rId1", "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
        "Target": "word/document.xml",
    })
    ET.SubElement(rels, tag(PR, "Relationship"), {
        "Id": "rId2", "Type": "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
        "Target": "docProps/core.xml",
    })
    return ET.tostring(rels, encoding="utf-8", xml_declaration=True)


def build_docx(source: Path, destination: Path) -> dict[str, int | str]:
    markdown = source.read_text(encoding="utf-8")
    body, table_count, figure_count, image_paths = build_document(markdown, source.parent)
    document = ET.Element(tag(W, "document"))
    document.append(body)
    document_xml = ET.tostring(document, encoding="utf-8", xml_declaration=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types())
        archive.writestr("_rels/.rels", root_rels())
        archive.writestr("word/_rels/document.xml.rels", document_rels(image_paths))
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/numbering.xml", numbering_xml())
        archive.writestr("word/footer1.xml", footer_xml())
        archive.writestr("word/settings.xml", settings_xml())
        archive.writestr("docProps/core.xml", core_properties())
        for image_path in image_paths:
            archive.write(image_path, f"word/media/{image_path.name}")
    words = len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", markdown))
    return {"output": str(destination), "word_count_source": words, "tables": table_count, "figures": figure_count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/THESIS_DRAFT.md"))
    parser.add_argument("--output", type=Path, default=Path("thesis.docx"))
    args = parser.parse_args()
    result = build_docx(args.input, args.output)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
