from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str
    openai_api_key: str
    # Postgres URL for telemetry persistence. Unset/empty -> telemetry is
    # only logged.
    database_url: str | None = None
    # LLM-as-a-judge quality evaluation of successful non-streaming
    # responses (see judge.py). Off by default: every evaluation is an
    # extra paid LLM call. The judge must be a concrete registry model.
    llm_judge_enabled: bool = False
    llm_judge_model: str = "gpt-4o-mini"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()