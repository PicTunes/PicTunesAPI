# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
from PIL import Image
import io, os, shutil, uuid, requests

from app.config import settings
from app.schemas import UploadResponse, SimilarItem, Music, MergeResponse
from app.services.simclr_service import SimCLRService
from app.services.media_service import join_image_audio_to_video
from app.utils.paths import abs_url, ensure_dirs
# from app.db import init_db, SessionLocal, ImageAsset  # enable when you connect DB

app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# CORS
allow_origins = [o.strip() for o in settings.CORS_ALLOW_ORIGINS.split(",")] if settings.CORS_ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# Globals
simclr: Optional[SimCLRService] = None

@app.on_event("startup")
def on_startup():
    ensure_dirs()
    global simclr
    simclr = SimCLRService(weights_path=settings.SIMCLR_WEIGHTS, device=settings.SIMCLR_DEVICE)

    # If you have a DB, load vector meta to memory
    # init_db()
    # with SessionLocal() as s:
    #     items = [(x.url, x.label, x.embedding_path) for x in s.query(ImageAsset).all()]
    # simclr.load_database(items)

    # with SessionLocal() as s:
    #     rows = s.execute(text("SELECT url, label, embedding_path FROM image_assets")).all()
    # items = [(r.url, r.label, r.embedding_path) for r in rows]
    # simclr.load_database(items)

@app.get("/health")
def health():
    return {"status": "ok"}

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

    # 1) embed + topk
    q = simclr.embed(pil)
    neighbors = simclr.topk(q, k=3)  # [{"image_url":..., "label":..., "similarity":...}, ...]

    # 2) build similar[] for client
    similar_items: List[SimilarItem] = []
    for n in neighbors:
        # If DB saved relative paths, you can serve via /static too
        image_abs = abs_url(request, n["image_url"])
        score = float(max(0.0, min(1.0, n.get("similarity", 0.0))))
        similar_items.append(SimilarItem(imageUrl=image_abs, score=score, label=n.get("label") or ""))

    # 3) label + music recommendation (demo)
    label = similar_items[0].label if similar_items else "unknown"
    music = [
        Music(title="Clair de Lune", composer="Debussy", start=30, end=60, link="https://youtu.be/CvFH_6DNRCY"),
        Music(title="Gymnopédie No.1", composer="Erik Satie", start=10, end=45, link="https://youtu.be/S-Xm7s9eGxU"),
    ]

    # 4) Optional: also return a quick teaser video now or let /merge handle it
    video_url = None

    return UploadResponse(label=label, music=music, similar=similar_items, videoUrl=video_url)

@app.post("/merge", response_model=MergeResponse)
async def merge_media(
    request: Request,
    image: UploadFile = File(None),
    audio: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None)
):
    """
    Accept either uploaded files or URLs for image and audio, then produce a video.
    """
    # Resolve image
    if image is not None:
        img_bytes = await image.read()
        img_name = f"img_{uuid.uuid4().hex[:8]}.jpg"
        img_path = os.path.join(settings.STATIC_DIR, img_name)
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    elif image_url:
        img_name = f"img_{uuid.uuid4().hex[:8]}.jpg"
        img_path = os.path.join(settings.STATIC_DIR, img_name)
        r = requests.get(image_url, timeout=20)
        r.raise_for_status()
        with open(img_path, "wb") as f:
            f.write(r.content)
    else:
        raise HTTPException(status_code=400, detail="Provide image file or image_url")

    # Resolve audio
    if audio is not None:
        aud_bytes = await audio.read()
        aud_name = f"aud_{uuid.uuid4().hex[:8]}.mp3"
        aud_path = os.path.join(settings.STATIC_DIR, aud_name)
        with open(aud_path, "wb") as f:
            f.write(aud_bytes)
    elif audio_url:
        aud_ext = "mp3"
        if audio_url.endswith(".m4a"): aud_ext = "m4a"
        aud_name = f"aud_{uuid.uuid4().hex[:8]}.{aud_ext}"
        aud_path = os.path.join(settings.STATIC_DIR, aud_name)
        r = requests.get(audio_url, timeout=30)
        r.raise_for_status()
        with open(aud_path, "wb") as f:
            f.write(r.content)
    else:
        raise HTTPException(status_code=400, detail="Provide audio file or audio_url")

    # Merge with FFmpeg
    try:
        out_path, dur = join_image_audio_to_video(img_path, aud_path, settings.OUTPUT_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FFmpeg failed: {e}")

    return MergeResponse(videoUrl=abs_url(request, out_path), duration=dur)
