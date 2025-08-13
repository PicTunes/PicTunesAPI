# app/config.py
from pydantic import BaseSettings, AnyUrl
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "Pictunes Backend"
    BASE_URL: str = "http://127.0.0.1:8000"  # used for absolute URLs
    STATIC_DIR: str = "app/static"
    OUTPUT_DIR: str = "app/static/out"
    DB_URL: Optional[str] = None             # e.g. mysql+pymysql://user:pwd@localhost/pictunes
    CORS_ALLOW_ORIGINS: str = "*"            # comma-separated

    # SimCLR
    SIMCLR_WEIGHTS: str = "weights/simclr.pt"
    SIMCLR_DEVICE: str = "cpu"               # "cuda" if available

    class Config:
        env_file = ".env"

settings = Settings()
