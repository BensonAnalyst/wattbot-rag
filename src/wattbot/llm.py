"""Chat model client for a Databricks serving endpoint (OpenAI-style chat completions over HTTPS).

Credentials come from environment variables or Kaggle Secrets, never from the code:
    DATABRICKS_HOST   e.g. https://dbc-xxxx.cloud.databricks.com
    DATABRICKS_TOKEN  a personal access token
"""

import os
import time


def chat_text(response: dict) -> str:
    """Reasoning models (gpt-oss) return content as a list of blocks; keep only the text blocks."""
    content = response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def make_databricks_llm(model: str = "databricks-gpt-oss-120b", host: str = None, token: str = None,
                        max_tokens: int = 4000, retries: int = 5):
    import requests

    host = (host or os.environ["DATABRICKS_HOST"]).rstrip("/")
    token = token or os.environ["DATABRICKS_TOKEN"]
    url = f"{host}/serving-endpoints/{model}/invocations"

    def llm(messages):
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
        for attempt in range(retries):
            try:
                resp = requests.post(url, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=300)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                return chat_text(resp.json())
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(5 * 2 ** attempt)

    return llm
