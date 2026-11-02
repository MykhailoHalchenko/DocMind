from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    faiss_index_path: str = Field("./data/knowledge_base.faiss", env="FAISS_INDEX_PATH")
    faiss_metadata_path: str = Field("./data/knowledge_base.json", env="FAISS_METADATA_PATH")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(1536, env="EMBEDDING_DIMENSIONS")

    fast_llm_model: str = Field("qwen/qwen3.5-397b-a17b", env="FAST_LLM_MODEL")
    fast_llm_base_url: str = Field("https://integrate.api.nvidia.com/v1", env="FAST_LLM_BASE_URL")
    fast_llm_api_key: str = Field(..., env="FAST_LLM_API_KEY")

    strong_llm_model: str = Field("z-ai/glm-5.1", env="STRONG_LLM_MODEL")
    strong_llm_base_url: str = Field("https://integrate.api.nvidia.com/v1", env="STRONG_LLM_BASE_URL")
    strong_llm_api_key: str = Field(..., env="STRONG_LLM_API_KEY")

    gemini_model: str = Field("gemini-3.5-pro", env="GEMINI_MODEL")
    gemini_api_key: str | None = Field(None, env="GEMINI_API_KEY")

    chunk_size: int = Field(200, env="CHUNK_SIZE")
    chunk_overlap: int = Field(50, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()