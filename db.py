from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config import FRAME_INTERVAL, TEMP_DIR, FRAME_DIR, QDRANT_COLLECTION_NAME

# Using in-memory Qdrant instance
client = QdrantClient(":memory:")

def init_qdrant(collection_name="frames"):
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE)
    )

def add_vector_to_db(frame_path: str, vector: list, idx: int, collection_name="frames"):
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(id=idx, vector=vector, payload={"image_path": frame_path})
        ]
    )

def search_vectors(query_vector: list, top_k: int, collection_name="frames"):
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_vectors=True  
    )

    output = []
    for r in results:
        output.append({
            "score": r.score,
            "image_path": r.payload["image_path"],
            "vector": r.vector if r.vector is not None else []  
        })

    return output
