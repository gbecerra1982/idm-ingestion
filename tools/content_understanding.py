import io
import json
import logging
from typing import Any, Dict, List, Tuple

from tabulate import tabulate

from .aoai import AzureOpenAIClient


class ContentUnderstandingService:
    """
    Transforms raw OCR table structure into normalized artifacts and semantics.

    Inputs: OCR JSON with cells/rows/cols (e.g., from Pixtral)
    Outputs:
      - normalized grid (list[list[str]]) with merged cells expanded
      - markdown table (str)
      - CSV bytes (io.BytesIO)
      - simple schema (dict)
      - header hierarchy (list[list[str]])
      - semantic summary text (str)
      - quality/confidence estimation (float)
    """

    def __init__(self, document_filename: str = ""):
        self.aoai = AzureOpenAIClient(document_filename=document_filename)

    # ---------------------- Public API ----------------------
    def process(self, ocr: Dict[str, Any]) -> Dict[str, Any]:
        grid, header_rows, header_hierarchy = self._reconstruct_grid_and_headers(ocr)

        markdown = self._grid_to_markdown(grid, header_rows)
        csv_bytes = self._grid_to_csv_bytes(grid)
        schema = self._infer_schema(grid, header_rows)
        quality = float(ocr.get("confidence", 0.0) or 0.0)
        summary = self._summarize(markdown)

        semantic = {
            "header_hierarchy": header_hierarchy,
            "summary": summary,
        }

        return {
            "grid": grid,
            "markdown": markdown,
            "csv_bytes": csv_bytes,
            "schema": schema,
            "semantic": semantic,
            "quality_confidence": quality,
        }

    # ---------------------- Helpers ----------------------
    def _reconstruct_grid_and_headers(
        self, ocr: Dict[str, Any]
    ) -> Tuple[List[List[str]], List[int], List[List[str]]]:
        rows = int(ocr.get("rows") or 0)
        cols = int(ocr.get("cols") or 0)
        cells = ocr.get("cells", [])

        # If rows/cols are not provided, infer from cells
        if rows == 0 or cols == 0:
            max_r = 0
            max_c = 0
            for cell in cells:
                r = int(cell.get("row", 0)) + int(cell.get("rowspan", 1))
                c = int(cell.get("col", 0)) + int(cell.get("colspan", 1))
                max_r = max(max_r, r)
                max_c = max(max_c, c)
            rows = max(rows, max_r)
            cols = max(cols, max_c)

        # Initialize grid
        grid: List[List[str]] = [["" for _ in range(cols)] for _ in range(rows)]
        header_mask: List[List[bool]] = [[False for _ in range(cols)] for _ in range(rows)]

        # Place text into grid, expanding spans
        for cell in cells:
            r0 = int(cell.get("row", 0))
            c0 = int(cell.get("col", 0))
            rs = int(cell.get("rowspan", 1))
            cs = int(cell.get("colspan", 1))
            text = str(cell.get("text", "")).strip()
            role = str(cell.get("role", "")).lower()

            for r in range(r0, min(rows, r0 + rs)):
                for c in range(c0, min(cols, c0 + cs)):
                    grid[r][c] = text
                    if role == "header":
                        header_mask[r][c] = True

        # Determine header rows: any row with majority header cells
        header_rows: List[int] = []
        for r in range(rows):
            header_count = sum(1 for c in range(cols) if header_mask[r][c])
            if cols > 0 and header_count / float(cols) >= 0.5:
                header_rows.append(r)

        if not header_rows and rows > 0:
            # Fallback: use first row as header
            header_rows = [0]

        # Build header hierarchy: for each column, collect header values from header rows
        header_hierarchy: List[List[str]] = []
        for c in range(cols):
            path: List[str] = []
            for r in header_rows:
                val = grid[r][c].strip()
                if val:
                    path.append(val)
            header_hierarchy.append(path)

        return grid, header_rows, header_hierarchy

    def _grid_to_markdown(self, grid: List[List[str]], header_rows: List[int]) -> str:
        if not grid:
            return ""

        # Collapse multi-level headers by joining with " / " into a single header row
        header_row_idx = header_rows[-1] if header_rows else 0
        headers = grid[header_row_idx]

        # Data starts after the last header row
        data_rows = grid[header_row_idx + 1 :] if header_row_idx + 1 < len(grid) else []

        # Build table with tabulate
        try:
            md = tabulate(data_rows, headers=headers, tablefmt="pipe")
        except Exception:
            # Fallback to a naive markdown formatter
            header_line = "| " + " | ".join(headers) + " |"
            sep_line = "|" + "---|" * len(headers)
            body = "\n".join(["| " + " | ".join(row) + " |" for row in data_rows])
            md = "\n".join([header_line, sep_line, body])
        return md

    def _grid_to_csv_bytes(self, grid: List[List[str]]) -> io.BytesIO:
        import csv

        out = io.StringIO()
        writer = csv.writer(out)
        for row in grid:
            writer.writerow(row)
        data = out.getvalue().encode("utf-8")
        return io.BytesIO(data)

    def _infer_schema(self, grid: List[List[str]], header_rows: List[int]) -> Dict[str, Any]:
        if not grid:
            return {"columns": []}
        header_row_idx = header_rows[-1] if header_rows else 0
        headers = grid[header_row_idx]
        # Sample a few rows to guess types
        sample_rows = grid[header_row_idx + 1 : header_row_idx + 6]

        def infer_type(values: List[str]) -> str:
            numeric = 0
            total = 0
            for v in values:
                v = (v or "").strip().replace(",", "")
                if not v:
                    continue
                total += 1
                try:
                    float(v)
                    numeric += 1
                except Exception:
                    pass
            if total > 0 and numeric / float(total) >= 0.8:
                return "number"
            return "string"

        columns = []
        for idx, name in enumerate(headers):
            col_vals = [row[idx] for row in sample_rows if idx < len(row)]
            columns.append({"name": name, "type": infer_type(col_vals)})
        return {"columns": columns}

    def _summarize(self, markdown_table: str) -> str:
        if not markdown_table:
            return ""
        prompt = (
            "Eres un analista de datos. Resume la tabla siguiente, destacando jerarquías de encabezados, "
            "columnas clave, unidades y relaciones implícitas. Devuelve bullets claros y concisos.\n\n"
            f"{markdown_table}"
        )
        try:
            return self.aoai.get_completion(prompt, max_tokens=800)
        except Exception as e:
            logging.error(f"[content-understanding] summarize error: {str(e)}")
            return ""

