from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    qdrant_url: str = Field(..., env="QDRANT_URL")
    qdrant_api_key: str | None = Field(None, env="QDRANT_API_KEY")
    qdrant_collection: str = Field("knowledge_base", env="QDRANT_COLLECTION")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")

    fast_llm_model: str = Field("gpt-4o-mini", env="FAST_LLM_MODEL")
    heavy_llm_model: str = Field("glm-4-5", env="HEAVY_LLM_MODEL")
    heavy_llm_api_key: str = Field(..., env="HEAVY_LLM_API_KEY")
    heavy_llm_base_url: str = Field(..., env="HEAVY_LLM_BASE_URL")

    top_k_results: int = Field(5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()