"""
Configuration settings for SmartCache ML.
Loads environment variables or defaults for cache, embeddings, routing, and cost estimates.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    # Project paths
    project_root: Path = PROJECT_ROOT
    data_raw_path: Path = PROJECT_ROOT / "data" / "raw"
    data_processed_path: Path = PROJECT_ROOT / "data" / "processed"
    reports_path: Path = PROJECT_ROOT / "reports"

    # SQLite Cache Settings
    cache_db_path: Path = Path(os.getenv("CACHE_DB_PATH", str(PROJECT_ROOT / "data" / "cache.db")))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    max_cache_size: int = int(os.getenv("MAX_CACHE_SIZE", "1000"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "86400"))  # 24 hours

    # Sentence Transformer Model
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # LLM Provider ('mock' or 'ollama')
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_small_model: str = os.getenv("OLLAMA_SMALL_MODEL", "llama3.2:1b")
    ollama_large_model: str = os.getenv("OLLAMA_LARGE_MODEL", "llama3.2:3b")

    # Estimated Token Pricing (USD per 1,000 tokens)
    # Used for hypothetical cloud API cost comparisons
    price_per_1k_input_tokens_small: float = float(
        os.getenv("PRICE_PER_1K_INPUT_TOKENS_SMALL", "0.00015")
    )
    price_per_1k_output_tokens_small: float = float(
        os.getenv("PRICE_PER_1K_OUTPUT_TOKENS_SMALL", "0.0006")
    )
    price_per_1k_input_tokens_large: float = float(
        os.getenv("PRICE_PER_1K_INPUT_TOKENS_LARGE", "0.0015")
    )
    price_per_1k_output_tokens_large: float = float(
        os.getenv("PRICE_PER_1K_OUTPUT_TOKENS_LARGE", "0.006")
    )

    # API Configuration
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
