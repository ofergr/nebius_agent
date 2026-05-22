"""Application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


DEFAULT_MCP_HOST = "127.0.0.1"
DEFAULT_MCP_PORT = 8000
DEFAULT_MCP_PATH = "/mcp"


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    nebius_api_key: str
    nebius_base_url: str
    nebius_model: str
    dataset_name: str
    dataset_split: str
    max_iterations: int
    checkpoint_db_path: str
    user_profile_dir: str


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
        checkpoint_db_path=os.getenv("CHECKPOINT_DB_PATH", ".langgraph_checkpoints.sqlite"),
        user_profile_dir=os.getenv("USER_PROFILE_DIR", ".user_profiles"),
    )


def get_mcp_server_host() -> str:
    """Return the host used by the standalone MCP server."""

    return os.getenv("MCP_SERVER_HOST", DEFAULT_MCP_HOST)


def get_mcp_server_port() -> int:
    """Return the port used by the standalone MCP server."""

    return int(os.getenv("MCP_SERVER_PORT", str(DEFAULT_MCP_PORT)))


def get_mcp_server_path() -> str:
    """Return the HTTP path used by the standalone MCP server."""

    path = os.getenv("MCP_SERVER_PATH", DEFAULT_MCP_PATH).strip() or DEFAULT_MCP_PATH
    return path if path.startswith("/") else f"/{path}"


def get_mcp_server_url() -> str:
    """Return the default client URL for connecting to the standalone MCP server."""

    return os.getenv(
        "MCP_SERVER_URL",
        f"http://{get_mcp_server_host()}:{get_mcp_server_port()}{get_mcp_server_path()}",
    )
