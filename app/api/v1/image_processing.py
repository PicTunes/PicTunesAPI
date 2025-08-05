"""
Image Processing API Endpoint
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.services.image_processor import ImageProcessor
import io

router = APIRouter(prefix="/images", tags=["image-processing"])

@router.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image and return processed result."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        processed_image_bytes = await ImageProcessor.process_image(file)

        return StreamingResponse(
            io.BytesIO(processed_image_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": "attachment; filename=processed_image.jpg"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))