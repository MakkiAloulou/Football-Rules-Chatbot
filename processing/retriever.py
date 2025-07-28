import os
from sentence_transformers import SentenceTransformer
from utils.vector_db_utils import load_faiss_index

EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
INDEX_PATH = "data/embeddings/index.faiss"
METADATA_PATH = "data/embeddings/metadata.json"

# Load model
model = SentenceTransformer(EMBEDDING_MODEL)

# Load index and metadata (must already be created and saved)
if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
    index, metadata = load_faiss_index(INDEX_PATH, METADATA_PATH)
else:
    raise FileNotFoundError("❌ FAISS index or metadata file not found. Run embedding first.")

def retrieve_chunks(query: str, top_k: int = 5):
    """
    Search the vector DB for the most relevant chunks to a query.

    Args:
        query (str): User question.
        top_k (int): Number of results to return.

    Returns:
        List[Dict]: List of top matching chunks with metadata.
    """
    query_embedding = model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if 0 <= i < len(metadata):
            results.append(metadata[i])
    return results