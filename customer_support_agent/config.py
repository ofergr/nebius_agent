"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str
    dataset_name: str
    dataset_split: str
    max_iterations: int


def get_settings() -> Settings:
    """Return application settings with sensible local defaults."""

    return Settings(
        nebius_api_key=os.getenv("NEBIUS_API_KEY", ""),
        nebius_base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1"),
        nebius_model=os.getenv("NEBIUS_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        dataset_name=os.getenv(
            "DATASET_NAME", "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        ),
        dataset_split=os.getenv("DATASET_SPLIT", "train"),
        # Assignment-level ReAct loop budget. One logical iteration may include
        # multiple LangGraph node steps, so agent.py translates this value.
        max_iterations=int(os.getenv("MAX_ITERATIONS", "12")),
    )
