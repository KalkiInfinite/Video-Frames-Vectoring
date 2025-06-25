import streamlit as st
import os
import shutil
from video_utils import extract_frames, compute_color_histogram
from db import init_qdrant, add_vector_to_db, search_vectors
from PIL import Image
import numpy as np
import tempfile

# Config
FRAME_DIR = "storage/frames"
DEFAULT_FRAME_INTERVAL = 1
DEFAULT_TOP_K = 5

# Setup dirs
os.makedirs("temp", exist_ok=True)
os.makedirs(FRAME_DIR, exist_ok=True)

# Init DB
init_qdrant()

# Page config with custom styling
st.set_page_config(
    page_title="Video Similarity Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin: 1rem 0;
    }
    
    .results-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
        
    .info-box {
        background: #2c3e50;
        color: white;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #3498db;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    .info-box h3 {
        color: #ecf0f1;
        margin-bottom: 1rem;
    }
    
    .info-box ul {
        color: #bdc3c7;
    }
    
    .info-box li {
        margin-bottom: 0.5rem;
    }
    
    .info-box em {
        color: #95a5a6;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Configuration")

    FRAME_INTERVAL = st.number_input(
        "Frame Interval (seconds)", 
        min_value=1, max_value=10, value=DEFAULT_FRAME_INTERVAL, step=1,
        help="Set how frequently to extract frames from the video."
    )

    TOP_K = st.slider(
        "Top K Similar Frames", 
        min_value=1, max_value=20, value=DEFAULT_TOP_K, step=1,
        help="Set how many similar frames to retrieve during search."
    )

    st.info(f"Storage: {FRAME_DIR}")
    
    st.markdown("### Quick Tips")
    st.markdown("""
    - Upload videos in MP4, AVI, or MOV format  
    - Use clear, high-quality images for better search results  
    - The app extracts frames every few seconds (you can configure it)  
    - Color similarity is used for matching  
    """)

st.markdown("""
<div class="main-header">
    <h1>Video-to-Image Similarity Search</h1>
    <p>Powered by Qdrant Vector Database</p>
</div>
""", unsafe_allow_html=True)

# Create two main columns
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # --- Upload Video Section ---
    st.markdown("""
    <div class="section-header">
        <h2>Step 1: Upload Video</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "avi", "mov"],
            help="Upload your video file to extract frames for similarity search"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if video_file is not None:
        temp_video_path = os.path.join("temp", video_file.name)
        with open(temp_video_path, "wb") as f:
            shutil.copyfileobj(video_file, f)

        with st.spinner("Processing video and extracting frames..."):
            frame_paths = extract_frames(temp_video_path, FRAME_INTERVAL)
            st.session_state["frame_paths"] = frame_paths  # Save to session

            progress_bar = st.progress(0)
            for idx, frame_path in enumerate(frame_paths):
                vector = compute_color_histogram(frame_path)
                add_vector_to_db(frame_path, vector, idx)
                progress_bar.progress((idx + 1) / len(frame_paths))

        col_metric1, col_metric2, col_metric3 = st.columns(3)
        with col_metric1:
            st.metric("Frames Extracted", len(frame_paths))
        with col_metric2:
            st.metric("Vector Dimension", "768")
        with col_metric3:
            st.metric("Processing Speed", "Fast")

        st.success("Video processed successfully!")
        
        st.subheader("Extracted Frames Preview")

        frames_per_row = 4
        for i in range(0, len(frame_paths), frames_per_row):
            frame_cols = st.columns(frames_per_row)
            for j, col in enumerate(frame_cols):
                if i + j < len(frame_paths):
                    frame_path = frame_paths[i + j]
                    frame_filename = os.path.basename(frame_path)
                    
                    with col:
                        st.image(
                            frame_paths[i + j], 
                            caption=f"Frame {i + j + 1}",
                            use_container_width=True
                        )
                        st.markdown(
                            f"<p style='text-align: center; font-size: 0.9rem; color: #666;'>{frame_filename}</p>",
                            unsafe_allow_html=True
                        )

with col2:
    st.markdown("""
    <div class="section-header">
        <h2>Step 2: Search Similar Frames</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        search_image = st.file_uploader(
            "Upload an image to find similar frames", 
            type=["jpg", "png", "jpeg"],
            help="Upload an image to search for similar frames in the uploaded video"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    if search_image is not None:
        img = Image.open(search_image).convert("RGB")
        
        st.markdown("### Query Image")
        query_col1, query_col2, query_col3 = st.columns([1, 2, 1])
        with query_col2:
            st.image(img, caption="Your Search Image", use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            with st.spinner("Searching for similar frames..."):
                query_vector = compute_color_histogram(tmp.name)
                results = search_vectors(query_vector, TOP_K)

        st.markdown("""
        <div class="results-container">
            <h2 style="color: white; text-align: center; margin-bottom: 2rem;">
                Top Matching Frames
            </h2>
        </div>
        """, unsafe_allow_html=True)

        if results:
            # Create columns for results
            result_cols = st.columns(TOP_K)
            for i, r in enumerate(results):
                with result_cols[i]:
                    st.markdown(f"""
                    <div style="background: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                        <h4 style="text-align: center; color: #333;">Match #{i+1}</h4>
                    </div>
                    <p style="text-align: center; color: #666; font-size: 0.9rem;">
                        {os.path.basename(r['image_path'])}
                    </p>
                    """, unsafe_allow_html=True)
                    
                    st.image(
                        r["image_path"], 
                        use_container_width=True
                    )

                    score = r['score']
                    if score > 0.8:
                        score_color = "#4CAF50"  # Green
                    elif score > 0.6:
                        score_color = "#FF9800"  # Orange
                    else:
                        score_color = "#F44336"  # Red
                    
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0.5rem;">
                        <span style="background: {score_color}; color: white; padding: 0.3rem 1rem; 
                                   border-radius: 20px; font-weight: bold;">
                            Score: {score:.3f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No similar frames found. Try uploading a different image.")

st.markdown("---")
st.markdown("""
<div class="info-box">
    <h3>About This Application</h3>
    <p><strong>Features:</strong></p>
    <ul>
        <li><strong>Video Processing:</strong> Automatic frame extraction from uploaded videos</li>
        <li><strong>Smart Search:</strong> Color histogram-based similarity matching</li>
        <li><strong>Fast Results:</strong> Powered by Qdrant vector database</li>
        <li><strong>Visual Results:</strong> Side-by-side comparison of matching frames</li>
    </ul>
    <p><em>Note: All data is stored temporarily and will be cleared on application restart.</em></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Statistics")
    total_frames = len(st.session_state.get("frame_paths", []))
    st.metric("Total Frames", total_frames)