from fastapi import FastAPI, Request, Response, status, File, Form,UploadFile
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from ffmpeg import output
from pydantic import BaseModel
import yaml
import mysql.connector as cnn
import asyncio
import os
import re
import warnings

from app.MediaMerger import media_merger
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

with open("secret.yaml", "r") as f:
    db_config = yaml.safe_load(f)

def get_db_connection():
    anime_db_connection = cnn.connect(
        host=db_config["db_connection"]["host"],
        user=db_config["db_connection"]["user"],
        passwd=db_config["db_connection"]["password"],
        database=db_config["db_connection"]["anime_database"]
    )
    anime_cursor = anime_db_connection.cursor()

    movie_db_connection = cnn.connect(
        host=db_config["db_connection"]["host"],
        user=db_config["db_connection"]["user"],
        passwd=db_config["db_connection"]["password"],
        database=db_config["db_connection"]["movie_database"]
    )
    movie_cursor = anime_db_connection.cursor()
    return anime_db_connection, movie_db_connection, anime_cursor, movie_cursor

def get_music_matches(matches, genre: str):
    music_matches = []
    for match in matches:
        result = None
        # detect for numeric id in the filename eg. ./dataset/Sports/30051.jpg -> 30051
        numeric_id = re.search(r'\d+', match['filename']).group(0)
        # then find the music_id in the music_table by the numeric id by mysql query
        anime_db_connection, movie_db_connection, anime_cursor, movie_cursor = get_db_connection()
        if genre == "Anime":
            anime_cursor.execute("SELECT * FROM pictunes_test_DB.music_table WHERE music_id = (SELECT music_id FROM pictunes_test_DB.link_table WHERE image_id = %s);", (numeric_id,))
            result = anime_cursor.fetchall()
        elif genre == "Movie":
            movie_cursor.execute("SELECT * FROM pictunes_movie_DB.music_table WHERE music_id = (SELECT music_id FROM pictunes_movie_DB.link_table WHERE image_id = %s);", (numeric_id,))
            result = movie_cursor.fetchall()
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Invalid genre"}
            )
        
        if result:
            music_matches.append({
                "music_id": result[0][0],
                "music_name": result[0][1],
                "artwork_title": result[0][2],
                "piece": result[0][3],
                "duration": result[0][4],
                "youtube_link": result[0][5],
                "composer": result[0][6],
                "kind": result[0][7]
            })
    return music_matches


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the PicTunes API!"}

@app.get("/health/")
def health_check():
    return Response(status_code=status.HTTP_200_OK)

@app.get("/dbcon_check/")
def db_connection_check():
    anime_db_connection, movie_db_connection, anime_cursor, movie_cursor = get_db_connection()
    try:
        anime_cursor.execute("SELECT 1")
        anime_result = anime_cursor.fetchone()
        anime_cursor.close()
        movie_cursor.execute("SELECT 1")
        movie_result = movie_cursor.fetchone()
        movie_cursor.close()
        anime_db_connection.close()
        movie_db_connection.close()
        return {"message": "db connection successful", "anime_db_connection": anime_result, "movie_db_connection": movie_result}
    except Exception as e:
        return {"message": f"db connection failed: {str(e)}", "anime_db_connection": anime_result, "movie_db_connection": movie_result}


@app.post("/upload")
async def img_analysis(image: UploadFile = File(...), genre: str = Form(...)):
    """
    Upload an image for classification and similarity search
    Returns predicted class and top 10 most similar images with URLs to access them
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
        # all_class_matches, all_top_matches, top_10_matches = simclr_module.fast_visualize_prediction(
        _, all_top_matches, _ = simclr_module.fast_visualize_prediction(
            image_path=temp_file_path,
            simclr_model=simclr_model,
            logreg_model=logreg_model,
            precomputed_data=precomputed_data,
            class_names=simclr_module.class_names
        )
        
        # threshold for classes
        matches = simclr_module.match_threshold(all_top_matches)

        # Clean up: delete the uploaded file after analysis
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[Upload] Cleaned up temporary file: {image.filename}")

        music_matches = get_music_matches(matches, genre)

        for match, music_match in zip(matches, music_matches):
            match["music_match"] = music_match
            # Add image URL for client to fetch
            match["image_url"] = f"/image/{match['class']}/{match['filename']}.jpg"

        return JSONResponse(
            content={
                "status": "success",
                "matches": matches,
            }
        )
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

@app.get("/image/{class_name}/{filename}")
async def get_image(class_name: str, filename: str):
    """
    Serve an image from the dataset
    """
    image_path = f"./dataset/{class_name}/{filename}"
    
    if not os.path.exists(image_path):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Image not found"}
        )
    
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.post("/media_merger/")
async def merger(img: UploadFile = File(...), aud: str = Form(...), genre: str = Form(...)):
    temp_file_path = None
    try:
        if genre is None:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"message": "Genre is required"}
            )
        
        # Save uploaded file temporarily
        upload_dir = "./processing"
        music_dir = "./Music_Data"
        os.makedirs(upload_dir, exist_ok=True)
        temp_file_path = os.path.join(upload_dir, img.filename)
        temp_aud_file_path = music_dir + "/" + genre + "/" + aud + ".mp3"

        if not os.path.exists(temp_aud_file_path):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"message": "Music file not found"}
            )
        
        if temp_file_path.endswith('.heic'):
            output_file_path = os.path.join(upload_dir, f"output_{img.filename[:-6]}_{aud}.mp4")
        else:
            output_file_path = os.path.join(upload_dir, f"output_{img.filename[:-5]}_{aud}.mp4")
        
        with open(temp_file_path, "wb") as buffer:
            content = await img.read()
            buffer.write(content)

        media_merger(temp_file_path, temp_aud_file_path, output_file_path)

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"[./processing] Cleaned up temporary file: {img.filename}")
        

        return FileResponse(path=f'{output_file_path}', media_type="video/mp4", filename=f'output_{img.filename[:-4]}_{aud}.mp4')

    except Exception as e:
        import traceback
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"message": f"Media merging failed: {str(e)}", "traceback": traceback.format_exc()}
        )

@app.get("/media_merger/cleanup/")
async def cleanup(filename: str):
    upload_dir = "./processing"
    os.makedirs(upload_dir, exist_ok=True)
    temp_file_path = os.path.join(upload_dir, filename)
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)

    return {"message": "File cleaned up successfully"}



