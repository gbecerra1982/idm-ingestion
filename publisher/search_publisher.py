import hashlib
import json
import logging
import os
from typing import Dict, List

from tools.aisearch import AISearchClient


class SearchPublisher:
    def __init__(self):
        self.index_name = os.getenv("SEARCH_INDEX_NAME", "").strip()
        if not self.index_name:
            raise ValueError("SEARCH_INDEX_NAME must be set when ENABLE_DIRECT_INDEXING=true")
        self.client = AISearchClient()

    def _hash_id(self, url: str, chunk_id: int) -> str:
        h = hashlib.sha1(f"{url}|{chunk_id}".encode("utf-8")).hexdigest()
        return h

    def map_chunk_to_doc(self, chunk: Dict) -> Dict:
        doc: Dict = {}
        doc["id"] = self._hash_id(chunk.get("url", ""), chunk.get("chunk_id", 0))
        doc["url"] = chunk.get("url")
        doc["filepath"] = chunk.get("filepath")
        doc["content"] = chunk.get("content")
        doc["summary"] = chunk.get("summary")
        doc["headers"] = chunk.get("headers")
        doc["page"] = chunk.get("page")
        doc["offset"] = chunk.get("offset")
        doc["chunk_type"] = chunk.get("chunk_type")
        # Vectors
        doc["contentVector"] = chunk.get("contentVector")
        # JSON/table-specific metadata
        if "jsonPath" in chunk:
            doc["jsonPath"] = chunk["jsonPath"]
        if "keys" in chunk:
            doc["keys"] = chunk["keys"]
        if "tableIds" in chunk:
            doc["tableIds"] = chunk["tableIds"]
        if "tableHeaderHierarchies" in chunk:
            try:
                doc["tableHeaderHierarchies"] = json.dumps(chunk["tableHeaderHierarchies"], ensure_ascii=False)
            except Exception:
                doc["tableHeaderHierarchies"] = None
        # Related resources
        doc["relatedImages"] = chunk.get("relatedImages")
        doc["relatedFiles"] = chunk.get("relatedFiles")
        return doc

    def publish_chunks(self, chunks: List[Dict]) -> bool:
        docs = [self.map_chunk_to_doc(c) for c in chunks]
        return self.client.index_documents(self.index_name, docs)

