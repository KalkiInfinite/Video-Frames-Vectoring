# config.py

# Frame extraction settings
FRAME_INTERVAL = 1  # default interval in seconds

# Qdrant settings
QDRANT_COLLECTION_NAME = "frames"
QDRANT_VECTOR_SIZE = 512  # This should match the length of your feature vector
QDRANT_DISTANCE = "cosine"  # or "euclidean", based on use case

# Storage paths
TEMP_DIR = "temp"
FRAME_DIR = "storage/frames"
