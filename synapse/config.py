from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_base_url: str = "http://localhost:30000/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "openbmb/MiniCPM5-1B"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # Database
    database_url: str = "postgresql://synapse:synapse@localhost:5432/synapse"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    session_ttl: int = 3600

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Retry
    json_parse_retries: int = 1
    tool_call_retries: int = 3
    tool_retry_max_delay: int = 8

    # Session
    max_turns: int = 20
    max_resume_age_hours: int = 24

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
