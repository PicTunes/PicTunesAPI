from fastapi import FastAPI, Response, status, File, UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import os

from app.MediaMerger import MediaMergerClass, media_merger

class SomeBasicModel(BaseModel):
    pass

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the PicTunes API!"}

@app.get("/health/")
def health_check():
    return Response(status_code=status.HTTP_200_OK)

@app.get("/dbcon_check/")
def db_connection_check():

    return {"message": "db connection successful"}


@app.post("/upload/")
def img_analysis(img: UploadFile = File(...)):
    img_content = img.file.read()
    # store the image or process it as needed
    # store image in uploads folder
    os.makedirs("uploads", exist_ok=True)
    with open(f"uploads/{img.filename}", "wb") as f:
        f.write(img_content)
    
    # send image to analysis function
    # simclr(img_content, img.filename)
    

    return FileResponse(f"uploads/{img.filename}", media_type="video/mp4")

@app.post("/media_merger/")
async def merger(img: UploadFile = File(...), aud: UploadFile = File(...)):
    media_merger(img, aud)
    return {"message": "Media merged successfully"} # Response body

