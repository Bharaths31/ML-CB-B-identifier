import io
import time

def generate_odt_report(results, mode="single"):
    """Generate an ODT report from prediction results.

    Args:
        results: single result dict or list of result dicts (batch).
        mode: "single" or "batch".

    Returns:
        bytes: ODT file content.
    """
    from odf.opendocument import OpenDocumentText
    from odf.style import Style, TextProperties, ParagraphProperties, TableColumnProperties, TableCellProperties
    from odf.text import P
    from odf.table import Table, TableColumn, TableRow, TableCell

    doc = OpenDocumentText()

    # --- Define styles ---
    title_style = Style(name="Title", family="paragraph")
    title_style.addElement(TextProperties(attributes={
        "fontsize": "20pt", "fontweight": "bold", "color": "#1a1a2e"
    }))
    title_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.3cm", "margintop": "0.3cm"
    }))
    doc.styles.addElement(title_style)

    heading_style = Style(name="Heading", family="paragraph")
    heading_style.addElement(TextProperties(attributes={
        "fontsize": "14pt", "fontweight": "bold", "color": "#16213e"
    }))
    heading_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.2cm", "margintop": "0.4cm"
    }))
    doc.styles.addElement(heading_style)

    subheading_style = Style(name="SubHeading", family="paragraph")
    subheading_style.addElement(TextProperties(attributes={
        "fontsize": "12pt", "fontweight": "bold", "color": "#0f3460"
    }))
    subheading_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.15cm", "margintop": "0.3cm"
    }))
    doc.styles.addElement(subheading_style)

    body_style = Style(name="Body", family="paragraph")
    body_style.addElement(TextProperties(attributes={
        "fontsize": "11pt", "color": "#333333"
    }))
    body_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.1cm"
    }))
    doc.styles.addElement(body_style)

    meta_style = Style(name="Meta", family="paragraph")
    meta_style.addElement(TextProperties(attributes={
        "fontsize": "9pt", "fontstyle": "italic", "color": "#666666"
    }))
    meta_style.addElement(ParagraphProperties(attributes={
        "marginbottom": "0.1cm"
    }))
    doc.styles.addElement(meta_style)

    # Table styles
    table_style = Style(name="TableStyle", family="table")
    doc.automaticstyles.addElement(table_style)

    col_wide = Style(name="ColWide", family="table-column")
    col_wide.addElement(TableColumnProperties(attributes={"columnwidth": "8cm"}))
    doc.automaticstyles.addElement(col_wide)

    col_narrow = Style(name="ColNarrow", family="table-column")
    col_narrow.addElement(TableColumnProperties(attributes={"columnwidth": "4cm"}))
    doc.automaticstyles.addElement(col_narrow)

    header_cell_style = Style(name="HeaderCell", family="table-cell")
    header_cell_style.addElement(TableCellProperties(attributes={
        "backgroundcolor": "#1a1a2e", "padding": "0.15cm",
        "borderbottom": "0.5pt solid #333333"
    }))
    doc.automaticstyles.addElement(header_cell_style)

    header_text_style = Style(name="HeaderText", family="paragraph")
    header_text_style.addElement(TextProperties(attributes={
        "fontsize": "10pt", "fontweight": "bold", "color": "#ffffff"
    }))
    doc.automaticstyles.addElement(header_text_style)

    cell_style = Style(name="DataCell", family="table-cell")
    cell_style.addElement(TableCellProperties(attributes={
        "padding": "0.1cm",
        "borderbottom": "0.5pt solid #cccccc"
    }))
    doc.automaticstyles.addElement(cell_style)

    cell_text_style = Style(name="CellText", family="paragraph")
    cell_text_style.addElement(TextProperties(attributes={
        "fontsize": "10pt", "color": "#333333"
    }))
    doc.automaticstyles.addElement(cell_text_style)

    # --- Document content ---
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    p = P(stylename=title_style, text="🐄 Cattle & Buffalo Breed Classifier — Test Report")
    doc.text.addElement(p)

    p = P(stylename=meta_style, text=f"Generated: {timestamp}")
    doc.text.addElement(p)

    if mode == "single":
        results_list = [results] if isinstance(results, dict) else results
    else:
        results_list = results if isinstance(results, list) else [results]

    model_name = results_list[0].get("model_used", "unknown") if results_list else "unknown"
    p = P(stylename=meta_style, text=f"Model: {model_name}")
    doc.text.addElement(p)
    p = P(stylename=meta_style, text=f"Mode: {'Single Image' if mode == 'single' else 'Batch (' + str(len(results_list)) + ' images)'}")
    doc.text.addElement(p)

    p = P(stylename=body_style, text="─" * 60)
    doc.text.addElement(p)

    for i, result in enumerate(results_list):
        if "error" in result:
            p = P(stylename=body_style, text=f"Error: {result['error']}")
            doc.text.addElement(p)
            continue

        filename = result.get("filename", f"Image {i+1}")

        p = P(stylename=heading_style, text=f"{'Result' if mode == 'single' else f'Image {i+1}'}: {filename}")
        doc.text.addElement(p)

        p = P(stylename=body_style, text=f"Species: {result['species']} ({result['species_confidence']:.1f}%)")
        doc.text.addElement(p)

        p = P(stylename=body_style, text=f"Predicted Breed: {result['top_breed']} ({result['top_breed_confidence']:.1f}%)")
        doc.text.addElement(p)

        p = P(stylename=subheading_style, text="Top 5 Predictions")
        doc.text.addElement(p)

        table = Table(name=f"Top5_{i}", stylename=table_style)
        table.addElement(TableColumn(stylename=col_narrow))
        table.addElement(TableColumn(stylename=col_wide))
        table.addElement(TableColumn(stylename=col_narrow))

        hrow = TableRow()
        for htext in ["Rank", "Breed", "Confidence"]:
            hcell = TableCell(stylename=header_cell_style)
            hcell.addElement(P(stylename=header_text_style, text=htext))
            hrow.addElement(hcell)
        table.addElement(hrow)

        for rank, breed_info in enumerate(result.get("top5_breeds", []), 1):
            row = TableRow()
            for val in [str(rank), breed_info["breed"], f"{breed_info['confidence']:.1f}%"]:
                dcell = TableCell(stylename=cell_style)
                dcell.addElement(P(stylename=cell_text_style, text=val))
                row.addElement(dcell)
            table.addElement(row)

        doc.text.addElement(table)

        if mode == "batch" and i < len(results_list) - 1:
            p = P(stylename=body_style, text="")
            doc.text.addElement(p)
            p = P(stylename=body_style, text="─" * 60)
            doc.text.addElement(p)

    if mode == "batch" and len(results_list) > 1:
        p = P(stylename=body_style, text="")
        doc.text.addElement(p)
        p = P(stylename=heading_style, text="Batch Summary")
        doc.text.addElement(p)

        summary_table = Table(name="Summary", stylename=table_style)
        summary_table.addElement(TableColumn(stylename=col_wide))
        summary_table.addElement(TableColumn(stylename=col_narrow))
        summary_table.addElement(TableColumn(stylename=col_wide))
        summary_table.addElement(TableColumn(stylename=col_narrow))

        hrow = TableRow()
        for htext in ["Filename", "Species", "Breed", "Confidence"]:
            hcell = TableCell(stylename=header_cell_style)
            hcell.addElement(P(stylename=header_text_style, text=htext))
            hrow.addElement(hcell)
        summary_table.addElement(hrow)

        for r in results_list:
            if "error" in r:
                continue
            row = TableRow()
            for val in [
                r.get("filename", "—"),
                r.get("species", "—"),
                r.get("top_breed", "—"),
                f"{r.get('top_breed_confidence', 0):.1f}%"
            ]:
                dcell = TableCell(stylename=cell_style)
                dcell.addElement(P(stylename=cell_text_style, text=val))
                row.addElement(dcell)
            summary_table.addElement(row)

        doc.text.addElement(summary_table)

        valid = [r for r in results_list if "error" not in r]
        cattle_count = sum(1 for r in valid if r.get("species") == "Cattle")
        buffalo_count = sum(1 for r in valid if r.get("species") == "Buffalo")
        avg_conf = sum(r.get("top_breed_confidence", 0) for r in valid) / len(valid) if valid else 0

        p = P(stylename=body_style, text="")
        doc.text.addElement(p)
        p = P(stylename=body_style, text=f"Total Images: {len(results_list)} | Cattle: {cattle_count} | Buffalo: {buffalo_count}")
        doc.text.addElement(p)
        p = P(stylename=body_style, text=f"Average Top Breed Confidence: {avg_conf:.1f}%")
        doc.text.addElement(p)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
