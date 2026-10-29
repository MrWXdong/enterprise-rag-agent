from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""

    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""

    database_url: str = ""

    retrieval_top_k: int = 2
    retrieval_neighbor_window: int = 0
    retrieval_max_chunks: int = 2

    retrieval_candidate_k: int = 5
    retrieval_max_k: int = 2
    retrieval_min_score: float = 0.55
    retrieval_min_relative_score: float = 0.85


settings = Settings()
