from fastapi import FastAPI, Response, status, File, UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
import asyncio
import os
import warnings

from app.MediaMerger import media_merger
# from app.Calculator import calc
import app.SimCLRAnalyse as simclr_module

# Import PyTorch for device info
import torch

# Get device info from the SimCLR module
device = simclr_module.device
NUM_WORKERS = simclr_module.NUM_WORKERS

print("Device:", device)
print("Num Workers:", NUM_WORKERS)

# Use the models and precomputed data from SimCLRAnalyse module
print("Using models from SimCLRAnalyse module...")
simclr_model = simclr_module.simclr_model
logreg_model = simclr_module.logreg_model
precomputed_data = simclr_module.precomputed_data

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the PicTunes API!"}

# to test type curl -X 'GET' 'http://localhost:8001/calc?a=5&b=3&operation=add'
@app.get("/health/")
def health_check():
    return Response(status_code=status.HTTP_200_OK)

@app.get("/dbcon_check/")
def db_connection_check():

    return {"message": "db connection successful"}


@app.post("/upload")
async def img_analysis(image: UploadFile = File(...)):
    """
    Upload an image for classification and similarity search
    Returns predicted class and top 10 most similar images
    """
    if simclr_model is None or logreg_model is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={"message": "Models not loaded properly"}
        )
    
    if precomputed_data is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Features not pre-computed properly"}
        )
    
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        temp_file_path = os.path.join(upload_dir, image.filename)
        
        with open(temp_file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Perform analysis using pre-computed features
        all_class_matches, top_10_matches = simclr_module.fast_visualize_prediction(
            image_path=temp_file_path,
            simclr_model=simclr_model,
            logreg_model=logreg_model,
            precomputed_data=precomputed_data,
            class_names=simclr_module.class_names
        )
        
        # Clean up: delete the uploaded file after analysis
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[Upload] Cleaned up temporary file: {image.filename}")
        
        return {
            "status": "success",
            "all_class_matches": all_class_matches,
            "top_10_matches": top_10_matches
        }
    except Exception as e:
        import traceback
        # Clean up the file even if analysis fails
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[Upload] Cleaned up temporary file after error: {image.filename}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}
        )

# Alternative GET endpoint for testing with existing files
@app.get("/analyze")
async def analyze_existing_image(image_filename: str):
    """
    Analyze an existing image file (for testing purposes)
    Expects image to be in the uploads/ directory
    """
    if simclr_model is None or logreg_model is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            content={"message": "Models not loaded properly"}
        )
    
    if precomputed_data is None:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": "Features not pre-computed properly"}
        )
    
    try:
        image_path = f"uploads/{image_filename}"
        if not os.path.exists(image_path):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": f"Image not found: {image_filename}"}
            )
        
        # Use the pre-computed features instead of computing them on each request
        all_class_matches, top_10_matches = simclr_module.fast_visualize_prediction(
            image_path=image_path,
            simclr_model=simclr_model,
            logreg_model=logreg_model,
            precomputed_data=precomputed_data,
            class_names=simclr_module.class_names
        )
        
        return {
            "status": "success",
            "all_class_matches": all_class_matches,
            "top_10_matches": top_10_matches
        }
    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Analysis failed: {str(e)}", "traceback": traceback.format_exc()}
        )

@app.get("/media_merger/")
async def merger(img: str, aud: str):
    media_merger(img, aud)
    return {"message": "Media merged successfully"} # Response body

