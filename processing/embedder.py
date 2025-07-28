from sentence_transformers import SentenceTransformer
from typing import List, Dict
EMBEDDING_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
def embed_chunks(
    chunks: List[Dict],
    model_name: str = EMBEDDING_MODEL
) -> List[Dict]:
    """
    Adds SBERT embeddings to each text chunk.

    Args:
        chunks (List[Dict]): List of chunks with 'text' field.
        model_name (str): Pretrained Sentence-BERT model name.

    Returns:
        List[Dict]: Same chunks with added 'embedding' field (as a list of floats).
    """
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    print("🔄 Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()  # convert NumPy array to list for serialization
    return chunks