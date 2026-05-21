"""List Nebius OpenAI-compatible model IDs using the local .env file."""

from __future__ import annotations

import os
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv


def main() -> None:
    """Print available model IDs for the configured Nebius endpoint."""

    load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    base_url = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1").rstrip("/")

    if not api_key:
        raise SystemExit("NEBIUS_API_KEY is not set. Add it to .env first.")

    models_url = urljoin(f"{base_url}/", "models")
    response = requests.get(
        models_url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    models = payload.get("data", [])
    if not models:
        print("No models returned.")
        return

    for model in models:
        model_id = model.get("id")
        if model_id:
            print(model_id)


if __name__ == "__main__":
    main()
