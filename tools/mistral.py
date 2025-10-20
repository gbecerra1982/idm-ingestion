import os
import json
import base64
import logging
import time
from typing import Any, Dict

import requests


class MistralPixtralClient:
    """
    Lightweight client to call a Mistral Pixtral (multimodal OCR) HTTP endpoint.

    The client expects an API compatible with a simple JSON POST of an image and
    returns a structured table detection result. Because vendors and deployments
    vary, the request/response shapes are parameterized with environment vars.

    Environment variables:
    - PIXTRAL_API_URL: Base URL of the Pixtral endpoint
    - PIXTRAL_API_KEY: Bearer/API key
    - PIXTRAL_MODEL_ID: Optional model identifier sent to the service
    - PIXTRAL_TIMEOUT_SEC: Optional request timeout (default 60s)
    """

    def __init__(self):
        self.api_url = os.getenv("PIXTRAL_API_URL", "").rstrip("/")
        self.api_key = os.getenv("PIXTRAL_API_KEY", "")
        self.model_id = os.getenv("PIXTRAL_MODEL_ID", "")
        self.timeout = int(os.getenv("PIXTRAL_TIMEOUT_SEC", "60"))

        self.enabled = bool(self.api_url and self.api_key)

    def analyze_table_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Sends the image to Pixtral and returns a structured response.

        Expected Pixtral response shape (example/generic):
        {
          "rows": 20,
          "cols": 8,
          "cells": [
            {"row": 0, "col": 0, "rowspan": 1, "colspan": 2, "text": "Header A", "role": "header"},
            ...
          ],
          "confidence": 0.96
        }
        """
        if not self.enabled:
            raise RuntimeError(
                "Pixtral OCR is not configured. Set PIXTRAL_API_URL and PIXTRAL_API_KEY."
            )

        payload = {
            "image": f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}",
        }
        if self.model_id:
            payload["model"] = self.model_id

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        start = time.time()
        try:
            resp = requests.post(
                f"{self.api_url}/v1/table/analyze",
                data=json.dumps(payload),
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
            logging.info(
                f"[pixtral] Table analysis finished in {elapsed:.2f}s, confidence={data.get('confidence','-')}"
            )
            return data
        except Exception as e:
            logging.error(f"[pixtral] analyze_table_image error: {str(e)}")
            raise

