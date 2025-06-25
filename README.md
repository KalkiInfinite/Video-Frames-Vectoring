# Video-to-Image Similarity Search

A Streamlit + FastAPI-powered application that allows you to:

- Upload a video file.
- Automatically extract frames at user-defined intervals.
- Compute color histogram–based feature vectors.
- Store them in a vector database (Qdrant).
- Search for similar frames by uploading an image.
- View results side-by-side with similarity scores.

---

## Features

- **Video Upload & Frame Extraction**
  - Extracts frames every `n` seconds (user configurable).
  
- **Feature Vector Computation**
  - Uses simple color histograms for frame representation.
  
- **Similarity Search**
  - Upload any image to find visually similar video frames.
  
- **Qdrant Vector Database**
  - In-memory vector search for fast results.

- **Streamlit UI**
  - Side-by-side comparison of search image and top-matching frames.
  - Fully responsive, clean, and interactive interface.
  
---

## Tech Stack

- **Frontend**: Streamlit (for UI)
- **Backend**: FastAPI (for API endpoints)
- **Vector Database**: Qdrant (in-memory mode)
- **Feature Extraction**: Color Histograms (using OpenCV)
- **Image Processing**: PIL, NumPy
- **Video Handling**: OpenCV

---

## Installation & Usage

1. **Clone the repo**:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. **Install requirements**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:

```bash
streamlit run app.py
```

4. **(Optional) Run FastAPI server**:

```bash
uvicorn api:app --reload
```

---

## Project Structure

```
.
├── app.py                  # Streamlit frontend
├── api.py                  # FastAPI backend
├── db.py                   # Qdrant database logic
├── video_utils.py          # Frame extraction + histogram
├── storage/frames/         # Saved extracted frames
├── temp/                   # Temporary uploads
├── requirements.txt
└── README.md
```
