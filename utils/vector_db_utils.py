import faiss
import os
import numpy as np
import json
from typing import List, Dict
from processing.embedder import embed_chunks

# Create FAISS index from list of embeddings (2D numpy array)
def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# Save FAISS index and metadata
def save_faiss_index(chunks, index_path="data/embeddings/index.faiss", metadata_path="data/embeddings/metadata.json"):
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
    metadata = [{"chunk_id": chunk["chunk_id"], "source": chunk["source"], "text": chunk["text"]} for chunk in chunks]

    index = build_faiss_index(embeddings)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved FAISS index to: {index_path}")
    print(f"📝 Saved metadata to: {metadata_path}")


# Load FAISS index and metadata
def load_faiss_index(index_path="data/embeddings/index.faiss", metadata_path="data/embeddings/metadata.json"):
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# Embeds new chunks and updates the existing FAISS index and metadata
def update_index_with_new_chunks(
    new_chunks: List[Dict],
    index_path: str = "data/embeddings/index.faiss",
    metadata_path: str = "data/embeddings/metadata.json"
) -> None:
    """
    Adds new embedded chunks to an existing FAISS index and updates the metadata file.

    Args:
        new_chunks (List[Dict]): New document chunks (unembedded).
        index_path (str): Path to the existing FAISS index.
        metadata_path (str): Path to the existing metadata JSON file.
    """
    print("🔄 Embedding new chunks...")
    embedded_chunks = embed_chunks(new_chunks)
    embeddings = np.array([chunk["embedding"] for chunk in embedded_chunks]).astype("float32")

    print("📥 Loading existing index and metadata...")
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        save_faiss_index(embedded_chunks, index_path, metadata_path)
        return

    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("➕ Adding new embeddings to index...")
    index.add(embeddings)

    print("📝 Updating metadata...")
    new_metadata = [
        {"chunk_id": chunk["chunk_id"], "source": chunk["source"], "text": chunk["text"]}
        for chunk in embedded_chunks
    ]
    metadata.extend(new_metadata)

    print("💾 Saving updated index and metadata...")
    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Successfully added {len(new_chunks)} new chunks to the index.")