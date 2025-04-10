# config/settings.py
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseSettings(BaseModel):
    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))

class OpenAISettings(BaseModel):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

class Settings(BaseModel):
    database: DatabaseSettings = DatabaseSettings()
    openai: OpenAISettings = OpenAISettings()

def get_settings() -> Settings:
    return Settings()