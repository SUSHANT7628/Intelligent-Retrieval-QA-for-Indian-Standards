import fitz
from pathlib import Path
import csv


def extract_tables_basic(pdf_path):
    """Best-effort table extraction using PyMuPDF text blocks.
    Produces page-level text rows when a true table extractor is unavailable."""
    rows = []
    doc = fitz.open(pdf_path)
    for page_no, page in enumerate(doc, 1):
        try:
            tables = page.find_tables()
            for table_no, table in enumerate(tables.tables, 1):
                data = table.extract()
                rows.append({'source': Path(pdf_path).name, 'page': page_no, 'table': table_no, 'rows': data})
        except Exception:
            continue
    return rows


def save_tables_csv(tables, output_dir='data/processed/tables'):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files=[]
    for t in tables:
        name=f"{Path(t['source']).stem}_p{t['page']}_t{t['table']}.csv"
        path=Path(output_dir)/name
        with path.open('w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerows(t['rows'])
        files.append(str(path))
    return files
