import io, os, uuid, requests, asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import torch
from sqlalchemy import select

from app.config import get_settings
from app.schemas import UploadResponse, MergeResponse, SimilarItem, Music
from app.services.simclr_service import SimCLRService
from app.services.media_service import join_image_audio_to_video
from app.utils.paths import abs_url, ensure_dirs
from app.db import engine, Base, check_db, SessionLocal
from app.models import ImageAsset

settings = get_settings()
app = FastAPI(title=settings.APP_NAME, version="1.0.0")

origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")] if settings.CORS_ALLOW_ORIGINS else ["*"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

simclr: Optional[SimCLRService] = None
EXECUTOR = ThreadPoolExecutor(max_workers=max(1, settings.INFER_MAX_CONCURRENCY))
INFER_SEMAPHORE = asyncio.Semaphore(max(1, settings.INFER_MAX_CONCURRENCY))

def _embed_sync(pil):
    return simclr.embed(pil)

@app.on_event("startup")
async def startup():
    ensure_dirs()
    torch.set_num_threads(max(1, settings.TORCH_NUM_THREADS))
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if engine is not None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)  # dev only; prefer Alembic in prod

    global simclr
    simclr = SimCLRService(weights_path=settings.SIMCLR_WEIGHTS, device=settings.SIMCLR_DEVICE, hidden_dim=128)

    if settings.AUTOLOAD_EMBEDDINGS_FROM_DB and SessionLocal is not None:
        try:
            async with SessionLocal() as session:
                res = await session.execute(select(ImageAsset.url, ImageAsset.label, ImageAsset.embedding_path))
                rows = res.all()
                items = [(url or "", label or "", emb or "") for (url, label, emb) in rows]
                simclr.load_database(items)
                print(f"[startup] Loaded {len(items)} embeddings from DB")
        except Exception as e:
            print(f"[startup] Failed to load embeddings from DB: {e}")

@app.get("/health")
async def health():
    db_ok = await check_db()
    return {"status": "ok", "db": db_ok}

@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    if simclr is None:
        raise HTTPException(status_code=500, detail="SimCLR service not initialized")
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {e}")

    async with INFER_SEMAPHORE:
        loop = asyncio.get_event_loop()
        q = await loop.run_in_executor(EXECUTOR, _embed_sync, pil)

    neighbors = simclr.topk(q, k=settings.TOPK)
    similar: List[SimilarItem] = []
    for n in neighbors:
        img_abs = abs_url(request, n["image_url"])
        score = max(0.0, min(1.0, float(n.get("similarity", 0.0))))
        similar.append(SimilarItem(imageUrl=img_abs, score=score, label=n.get("label") or ""))

    label = similar[0].label if similar else "unknown"
    music = [
        Music(title="Clair de Lune", composer="Debussy", start=30, end=60, link="https://youtu.be/CvFH_6DNRCY"),
        Music(title="Gymnopédie No.1", composer="Erik Satie", start=10, end=45, link="https://youtu.be/S-Xm7s9eGxU"),
    ]
    return UploadResponse(label=label, music=music, similar=similar, videoUrl=None)

@app.post("/merge", response_model=MergeResponse)
async def merge(request: Request, image: UploadFile = File(None), audio: UploadFile = File(None),
                image_url: Optional[str] = Form(None), audio_url: Optional[str] = Form(None)):
    if image is not None:
        img_bytes = await image.read()
    elif image_url:
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        img_bytes = r.content
    else:
        raise HTTPException(status_code=400, detail="Provide image or image_url")

    if audio is not None:
        aud_bytes = await audio.read()
        aud_ext = os.path.splitext(audio.filename or "")[1].lstrip(".") or "mp3"
    elif audio_url:
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        aud_bytes = r.content
        aud_ext = "mp3" if ".mp3" in audio_url else "m4a" if ".m4a" in audio_url else "mp3"
    else:
        raise HTTPException(status_code=400, detail="Provide audio or audio_url")

    img_name = f"img_{uuid.uuid4().hex[:8]}.jpg"
    img_path = os.path.join(settings.STATIC_DIR, img_name)
    with open(img_path, "wb") as f:
        f.write(img_bytes)

    aud_name = f"aud_{uuid.uuid4().hex[:8]}.{aud_ext}"
    aud_path = os.path.join(settings.STATIC_DIR, aud_name)
    with open(aud_path, "wb") as f:
        f.write(aud_bytes)

    try:
        out_path, dur = join_image_audio_to_video(img_path, aud_path, settings.OUTPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {e}")

    return MergeResponse(videoUrl=abs_url(request, out_path), duration=dur)