"""
Application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import image_processing, media_merge
from app.core.config import settings

app = FastAPI(
    title="PicTunes Backend API",
    description="API for PicTunes, a music and image processing application",
    version="1.0.0",
)

# Enable CORS for iOS app communication
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(image_processing.router, prefix="/api/v1")
app.include_router(media_merge.router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "PicTunes Backend API is running!"}