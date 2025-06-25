from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import json
from video_utils import extract_frames, compute_color_histogram
from db import init_qdrant, add_vector_to_db, search_vectors
from typing import List
from models import SearchRequest, SearchResult
from config import FRAME_INTERVAL, TEMP_DIR, FRAME_DIR, QDRANT_COLLECTION_NAME

import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create required dirs
os.makedirs("temp", exist_ok=True)
os.makedirs("storage/frames", exist_ok=True)

@app.on_event("startup")
def startup_event():
    init_qdrant()

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), interval: int = 1):
    video_path = f"temp/{file.filename}"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    frame_paths = extract_frames(video_path, interval)

    for idx, frame_path in enumerate(frame_paths):
        vector = compute_color_histogram(frame_path)
        add_vector_to_db(frame_path, vector, idx)

        if idx == 0:  # ✅ Print vector only for first frame
            print("Sample vector for testing:")
            print(json.dumps(vector))  # full vector or use vector[:10] for short preview

    return {"message": "Video processed", "frames_extracted": len(frame_paths)}


@app.post("/search/", response_model=List[SearchResult])
async def search_similar(request: SearchRequest):
    return search_vectors(request.query_vector, request.top_k)
