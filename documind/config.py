from dataclasses import dataclass
from pathlib import Path
import os
import streamlit as st


def get_setting(key: str, default: str) -> str:
    # Local environment variables take priority
    value = os.getenv(key)
    if value is not None:
        return value

    # Fall back to Streamlit Cloud secrets
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    ai_provider: str = get_setting("AI_PROVIDER", "ollama").lower()

    ollama_host: str = get_setting(
        "OLLAMA_HOST", "http://localhost:11434"
    ).rstrip("/")

    ollama_model: str = get_setting(
        "OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"
    )

    groq_api_key: str = get_setting("GROQ_API_KEY", "")

    groq_model: str = get_setting(
        "GROQ_MODEL", "openai/gpt-oss-120b"
    )

    embedding_model: str = get_setting(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    request_timeout: int = int(
        get_setting("OLLAMA_TIMEOUT", "240")
    )

    data_dir: Path = Path(
        get_setting("DOCUMIND_DATA_DIR", "data")
    )


settings = Settings()