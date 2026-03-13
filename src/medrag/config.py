from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Anthropic
    anthropic_api_key: str = ""

    # LLM
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.0

    # 검색
    retriever_top_k: int = 10
    reranker_top_k: int = 5
    retriever_score_threshold: float = 0.3

    # ChromaDB
    chroma_persist_dir: Path = Path("./data/vectorstore")
    chroma_collection_name: str = "medrag"

    # 임베딩
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # 청킹
    chunk_size: int = 512
    chunk_overlap: int = 64


settings = Settings()
