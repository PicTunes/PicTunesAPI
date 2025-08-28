from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "Pictunes Backend"
    BASE_URL: str = "http://127.0.0.1:8000"
    CORS_ALLOW_ORIGINS: str = "*"
    STATIC_DIR: str = "app/static"
    OUTPUT_DIR: str = "app/static/out"
    SIMCLR_WEIGHTS: str = "weights/simclr.pt"
    SIMCLR_DEVICE: str = "cpu"
    INFER_MAX_CONCURRENCY: int = 4
    TORCH_NUM_THREADS: int = 4
    DATABASE_URL_ASYNC: str | None = None  # mysql+asyncmy://user:pass@IP:3306/pictunes?charset=utf8mb4
    AUTOLOAD_EMBEDDINGS_FROM_DB: bool = True
    TOPK: int = 3
    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    return Settings()