from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field("knowledge_base", env="QDRANT_COLLECTION")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")

    chunk_size: int = Field(200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")

    top_k_results: int = Field(5, env="TOP_K_RESULTS")

    class Config:
        env_file = ("mcp.env", ".env")
        env_file_encoding = "utf-8"


settings = Settings()