
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    ai_provider: str = os.getenv("AI_PROVIDER", "ollama").lower()

    ollama_host: str = os.getenv(
        "OLLAMA_HOST", "http://localhost:11434"
    ).rstrip("/")

    ollama_model: str = os.getenv(
        "OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"
    )

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    groq_model: str = os.getenv(
        "GROQ_MODEL", "llama-3.3-70b-versatile"
    )

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )

    request_timeout: int = int(
        os.getenv("OLLAMA_TIMEOUT", "240")
    )

    data_dir: Path = Path(
        os.getenv("DOCUMIND_DATA_DIR", "data")
    )


settings = Settings()
