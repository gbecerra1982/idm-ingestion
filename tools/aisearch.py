import logging
import os
from typing import List, Dict

from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient


class AISearchClient:
    """
    Cliente simple para indexación directa en Azure AI Search usando AAD (DefaultAzureCredential).
    """

    def __init__(self):
        self.search_service_name = os.getenv("SEARCH_SERVICE_NAME", "").strip()
        if not self.search_service_name:
            raise ValueError("SEARCH_SERVICE_NAME must be set for direct indexing")
        self.endpoint = f"https://{self.search_service_name}.search.windows.net"
        self.credential = DefaultAzureCredential()
        self._clients: Dict[str, SearchClient] = {}

    def _get_client(self, index_name: str) -> SearchClient:
        if index_name not in self._clients:
            self._clients[index_name] = SearchClient(
                endpoint=self.endpoint,
                index_name=index_name,
                credential=self.credential,
            )
        return self._clients[index_name]

    def index_documents(self, index_name: str, documents: List[Dict]) -> bool:
        client = self._get_client(index_name)
        batch_size = int(os.getenv("SEARCH_BATCH_SIZE", "256"))
        ok = True
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            try:
                result = client.upload_documents(documents=batch)
                # result is a list of IndexingResult
                failed = [r for r in result if not r.succeeded]
                if failed:
                    ok = False
                    logging.error(f"[aisearch] {len(failed)} docs failed in batch {i//batch_size}")
            except Exception as e:
                ok = False
                logging.error(f"[aisearch] exception indexing batch {i//batch_size}: {e}")
        return ok

    def delete_documents(self, index_name: str, key_field: str, keys: List[str]) -> bool:
        client = self._get_client(index_name)
        batch_size = int(os.getenv("SEARCH_BATCH_SIZE", "256"))
        ok = True
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            try:
                client.delete_documents(key_field, batch)
            except Exception as e:
                ok = False
                logging.error(f"[aisearch] exception deleting batch {i//batch_size}: {e}")
        return ok

