# app/utils/paths.py
import os
from fastapi import Request
from app.config import settings

def abs_url(request: Request, path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    base = str(request.base_url).rstrip("/")
    return f"{base}/{path_or_url.lstrip('/')}"

def ensure_dirs():
    os.makedirs(settings.STATIC_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
