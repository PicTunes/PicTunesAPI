from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from app.services.media_merger import MediaMerger
import io

router = APIRouter(prefix="/media", tags=["media-merge"])

@router.post("/merge")
async def merge_media(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    duration: int = Form(15)
):
    """Merge image with audio to create video"""
    try:
        result_bytes = await MediaMerger.merge_media_audio(image, audio, duration)

        return StreamingResponse(
            io.BytesIO(result_bytes),
            media_type="video/mp4",
            headers={"Content-Disposition": "attachment; filename=merged_video.mp4"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))