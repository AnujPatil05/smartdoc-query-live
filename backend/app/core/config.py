# backend/app/core/config.py
from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Union

class Settings(BaseSettings):
    # App
    APP_NAME: str = "SmartDoc Query System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Google Gemini (replaces OpenAI)
    GOOGLE_API_KEY: str
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_DIMENSION: int = 768  # Gemini embedding dimension
    
    # Legacy OpenAI (optional, for migration)
    OPENAI_API_KEY: Optional[str] = None
    
    # CORS - accepts "*" or comma-separated origins or JSON array
    CORS_ORIGINS: Union[str, List[str]] = ["http://localhost:5173", "http://localhost:3000"]
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            if v.startswith("["):
                import json
                return json.loads(v)
            return [origin.strip() for origin in v.split(",")]
        return v
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: List[str] = ["application/pdf"]
    UPLOAD_DIR: str = "./uploads"
    
    # Chunking
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens
    
    # RAG
    TOP_K_CHUNKS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Caching
    QUERY_CACHE_TTL: int = 3600  # 1 hour
    EMBEDDING_CACHE_TTL: int = 2592000  # 30 days
    
    # Rate Limiting
    RATE_LIMIT_UPLOAD: str = "5/hour"
    RATE_LIMIT_QUERY: str = "20/minute"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()