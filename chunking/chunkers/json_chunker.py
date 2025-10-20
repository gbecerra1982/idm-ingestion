import json
import logging
import os
from typing import Any, Dict, List, Tuple

from .base_chunker import BaseChunker
from ..exceptions import UnsupportedFormatError


class JSONChunker(BaseChunker):
    """
    Chunker especializado para documentos JSON.

    Estrategia:
    - Si la raíz es un array de objetos: agrupa N elementos por chunk (JSON_MAX_ITEMS_PER_CHUNK).
      - Si los objetos tienen llaves homogéneas (>= JSON_MIN_KEYS_FOR_TABLE), renderiza tabla Markdown.
      - Si no, renderiza lista de pares clave=valor (flatten parcial).
    - Si la raíz es un objeto: detecta el mayor array de objetos en paths top-level y procede como arriba.
      - Si no hay arrays, renderiza objeto plano en un único chunk.

    Metadatos añadidos por chunk:
    - chunk_type = "json"
    - headers = claves principales (si tabla)
    - jsonPath: path origen ("$" para raíz o "$.<key>" para arrays top-level)
    - itemRange: "start-end" para arrays
    - keys: lista de claves usadas
    """

    def __init__(self, data, max_chunk_size=None, minimum_chunk_size=None, token_overlap=None):
        super().__init__(data)
        self.max_chunk_size = max_chunk_size or int(os.getenv("NUM_TOKENS", "2048"))
        self.minimum_chunk_size = minimum_chunk_size or int(os.getenv("MIN_CHUNK_SIZE", "100"))
        self.token_overlap = token_overlap or int(os.getenv("TOKEN_OVERLAP", "100"))

        self.json_max_items = int(os.getenv("JSON_MAX_ITEMS_PER_CHUNK", "50"))
        self.json_flatten_depth = int(os.getenv("JSON_FLATTEN_DEPTH", "3"))
        self.json_min_keys_for_table = int(os.getenv("JSON_MIN_KEYS_FOR_TABLE", "3"))
        exclude = os.getenv("JSON_EXCLUDE_FIELDS", "").strip()
        self.json_exclude_fields = [p.strip() for p in exclude.split(",") if p.strip()] if exclude else []
        self.json_summarize = os.getenv("JSON_SUMMARIZE", "false").lower() == "true"

    def get_chunks(self):
        if self.extension != 'json':
            raise UnsupportedFormatError(f"[json_chunker] {self.extension} format is not supported")

        logging.info(f"[json_chunker][{self.filename}] Running get_chunks.")
        blob_data = self.blob_client.download_blob()
        text = blob_data.decode('utf-8', errors='replace')

        try:
            payload = json.loads(text)
        except Exception as e:
            logging.error(f"[json_chunker][{self.filename}] Invalid JSON: {e}")
            payload = text  # fallback: treat as plain text

        chunks: List[Dict[str, Any]] = []

        if isinstance(payload, list):
            chunks.extend(self._chunks_from_array(payload, json_path="$") )
        elif isinstance(payload, dict):
            # choose largest top-level array of objects
            best_key, best_arr = self._select_best_array(payload)
            if best_arr is not None:
                path = f"$.{best_key}"
                chunks.extend(self._chunks_from_array(best_arr, json_path=path))
            else:
                # single object → one chunk
                md, keys = self._render_object(payload)
                if self.json_summarize:
                    summary = self._summarize(md)
                else:
                    summary = ""
                content = self._ensure_limits(md)
                chunk = self._create_chunk(
                    chunk_id=1,
                    content=content,
                    summary=summary,
                    headers=keys,
                    chunk_type="json"
                )
                chunk["jsonPath"] = "$"
                chunk["keys"] = keys
                chunk["itemRange"] = "0-0"
                chunks.append(chunk)
        else:
            # raw text fallback
            content = self._ensure_limits(str(payload))
            chunk = self._create_chunk(
                chunk_id=1,
                content=content,
                chunk_type="json"
            )
            chunk["jsonPath"] = "$"
            chunk["keys"] = []
            chunk["itemRange"] = "0-0"
            chunks.append(chunk)

        logging.info(f"[json_chunker][{self.filename}] {len(chunks)} chunk(s) created")
        return chunks

    def _chunks_from_array(self, arr: List[Any], json_path: str) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        if not arr:
            return chunks

        # Identify if items are dict-like with somewhat homogeneous keys
        dict_items = [it for it in arr if isinstance(it, dict)]
        common_keys = self._common_keys(dict_items)

        chunk_id = 0
        for start in range(0, len(arr), self.json_max_items):
            end = min(start + self.json_max_items, len(arr))
            batch = arr[start:end]

            if dict_items and len(common_keys) >= self.json_min_keys_for_table:
                md = self._render_table(batch, columns=list(common_keys))
                keys = list(common_keys)
            else:
                # render list of items
                md = self._render_items(batch)
                keys = list(common_keys)

            summary = self._summarize(md) if self.json_summarize else ""
            content = self._ensure_limits(md)
            chunk_id += 1
            chunk = self._create_chunk(
                chunk_id=chunk_id,
                content=content,
                summary=summary,
                headers=keys,
                page=0,
                offset=start,
                chunk_type="json"
            )
            chunk["jsonPath"] = json_path
            chunk["keys"] = keys
            chunk["itemRange"] = f"{start}-{end-1}"
            chunks.append(chunk)

        return chunks

    def _select_best_array(self, obj: Dict[str, Any]) -> Tuple[str, List[Any] | None]:
        best_key = None
        best_arr = None
        best_len = 0
        for k, v in obj.items():
            if isinstance(v, list) and len(v) > best_len:
                best_key, best_arr, best_len = k, v, len(v)
        return best_key, best_arr

    def _common_keys(self, items: List[Dict[str, Any]]) -> set:
        keys = None
        for it in items[:100]:  # sample
            if not isinstance(it, dict):
                continue
            kset = set(it.keys())
            keys = kset if keys is None else keys & kset
            if not keys:
                break
        return keys or set()

    def _render_table(self, items: List[Any], columns: List[str]) -> str:
        # Build markdown table; flatten values shallowly
        header = "| " + " | ".join(columns) + " |\n" + "|" + "---|" * len(columns) + "\n"
        rows = []
        for it in items:
            if isinstance(it, dict):
                row = []
                for col in columns:
                    val = it.get(col, "")
                    sval = self._short_value(val)
                    row.append(sval)
                rows.append("| " + " | ".join(row) + " |")
            else:
                rows.append(f"| {self._short_value(it)} |")
        return header + "\n".join(rows)

    def _render_items(self, items: List[Any]) -> str:
        lines = []
        for idx, it in enumerate(items):
            if isinstance(it, dict):
                flat = self._flatten(it, depth=self.json_flatten_depth)
                filtered = {k: v for k, v in flat.items() if not self._excluded(k)}
                body = "\n".join([f"- {k}: {self._short_value(v)}" for k, v in filtered.items()])
                lines.append(f"### Item {idx}\n{body}")
            else:
                lines.append(f"- {self._short_value(it)}")
        return "\n\n".join(lines)

    def _render_object(self, obj: Dict[str, Any]) -> Tuple[str, List[str]]:
        flat = self._flatten(obj, depth=self.json_flatten_depth)
        filtered = {k: v for k, v in flat.items() if not self._excluded(k)}
        keys = list(filtered.keys())
        md = "\n".join([f"- {k}: {self._short_value(v)}" for k, v in filtered.items()])
        return md, keys

    def _excluded(self, key: str) -> bool:
        for pat in self.json_exclude_fields:
            if key.startswith(pat):
                return True
        return False

    def _short_value(self, v: Any, maxlen: int = 200) -> str:
        try:
            if isinstance(v, (dict, list)):
                s = json.dumps(v, ensure_ascii=False)
            else:
                s = str(v)
            if len(s) > maxlen:
                return s[:maxlen] + "…"
            return s
        except Exception:
            return ""

    def _flatten(self, obj: Dict[str, Any], parent_key: str = "", depth: int = 3) -> Dict[str, Any]:
        items: Dict[str, Any] = {}
        if depth <= 0:
            return {parent_key.rstrip('.'): obj} if parent_key else obj
        for k, v in obj.items():
            new_key = f"{parent_key}{k}" if not parent_key else f"{parent_key}.{k}"
            if isinstance(v, dict):
                items.update(self._flatten(v, new_key, depth-1))
            elif isinstance(v, list):
                # represent lists compactly
                try:
                    s = json.dumps(v[:10], ensure_ascii=False)
                    items[new_key] = s + ("…" if len(v) > 10 else "")
                except Exception:
                    items[new_key] = "[]"
            else:
                items[new_key] = v
        return items

    def _summarize(self, markdown_text: str) -> str:
        try:
            prompt = (
                "Eres un analista de datos. Resume brevemente el contenido JSON a nivel de campos y patrones.\n\n"
                f"{markdown_text[:4000]}"
            )
            return self.aoai_client.get_completion(prompt, max_tokens=400)
        except Exception:
            return ""

    def _ensure_limits(self, text: str) -> str:
        if self.token_estimator.estimate_tokens(text) > self.max_chunk_size:
            logging.info(f"[json_chunker][{self.filename}] truncating chunk to fit within {self.max_chunk_size} tokens")
            return self._truncate_chunk(text)
        return text

