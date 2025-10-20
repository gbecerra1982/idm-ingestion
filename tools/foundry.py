import os
import json
import base64
import logging
import time
from typing import Any, Dict, Tuple

import requests


class FoundryBaseClient:
    def __init__(self):
        self.endpoint = os.getenv("FOUNDRY_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("FOUNDRY_API_KEY", "")
        self.timeout = int(os.getenv("FOUNDRY_TIMEOUT_SEC", "60"))
        self.max_retries = int(os.getenv("FOUNDRY_MAX_RETRIES", "8"))
        if not self.endpoint or not self.api_key:
            raise RuntimeError("Foundry is not configured. Set FOUNDRY_ENDPOINT and FOUNDRY_API_KEY.")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.endpoint}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.time()
                resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                resp.raise_for_status()
                elapsed = time.time() - start
                logging.info(f"[foundry] POST {path} {resp.status_code} in {elapsed:.2f}s")
                return resp.json()
            except Exception as e:
                last_error = e
                wait = min(2 ** attempt * 0.2, 8.0)
                logging.warning(f"[foundry] POST {path} attempt {attempt}/{self.max_retries} failed: {e}. Retrying in {wait}s")
                time.sleep(wait)
        raise last_error


class FoundryEmbeddingsClient(FoundryBaseClient):
    def __init__(self):
        super().__init__()
        self.model_id = os.getenv("FOUNDRY_EMBEDDING_MODEL_ID", "")
        if not self.model_id:
            raise RuntimeError("FOUNDRY_EMBEDDING_MODEL_ID not set.")

    def get_embeddings(self, text: str):
        payload = {
            "model": self.model_id,
            "input": text,
        }
        data = self._post("/v1/embeddings", payload)
        return data["data"][0]["embedding"]


class FoundryChatClient(FoundryBaseClient):
    def __init__(self):
        super().__init__()
        self.model_id = os.getenv("FOUNDRY_CHAT_MODEL_ID", "")
        if not self.model_id:
            raise RuntimeError("FOUNDRY_CHAT_MODEL_ID not set.")

    def get_completion(self, prompt: str, max_tokens: int = 800) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
        }
        data = self._post("/v1/chat/completions", payload)
        # Try OpenAI-like shape first
        if "choices" in data and data["choices"]:
            msg = data["choices"][0].get("message", {}).get("content", "")
            return msg or ""
        # Fallback: plain text field
        return data.get("text", "")


class FoundryVisionClient(FoundryBaseClient):
    def __init__(self):
        super().__init__()
        self.vision_model = os.getenv("FOUNDRY_VISION_MODEL_ID", "")
        if not self.vision_model:
            raise RuntimeError("FOUNDRY_VISION_MODEL_ID not set.")

    def analyze_table_image(self, image_bytes: bytes) -> Dict[str, any]:
        payload = {
            "model": self.vision_model,
            "image": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
        }
        data = self._post("/v1/vision/table/analyze", payload)
        return data

    def analyze_document(self, file_bytes: bytes) -> Tuple[Dict[str, Any], list]:
        """
        Returns a dict with at least {"content": markdown_str, "tables": optional list}
        and a list of errors if any.
        """
        payload = {
            "model": self.vision_model,
            "file": f"data:application/octet-stream;base64,{base64.b64encode(file_bytes).decode('utf-8')}",
        }
        data = self._post("/v1/vision/document/analyze", payload)
        errors = []
        result = {
            "content": data.get("content", ""),
        }
        if "tables" in data:
            result["tables"] = data["tables"]
        return result, errors

