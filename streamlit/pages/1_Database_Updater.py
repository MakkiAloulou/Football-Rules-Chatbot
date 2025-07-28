import sys
import streamlit as st
import os
import uuid
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.pdf_parser import extract_text_from_pdf
from processing.chunker import chunk_documents
from utils.vector_db_utils import update_index_with_new_chunks

st.set_page_config(page_title="Database Updater", page_icon="📦")
st.title("📦 Database Updater")

# Initialize session state
if "chunked_docs" not in st.session_state:
    st.session_state.chunked_docs = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None
    
uploaded_file = st.file_uploader("Upload a PDF file to chunk", type=["pdf"])

if uploaded_file:
    file_type = uploaded_file.name.lower().split('.')[-1]

    if uploaded_file.name != st.session_state.last_file_name:
        st.session_state.last_file_name = uploaded_file.name
        st.info("📥 Processing file...")

        upload_dir = "data\\uploads\\uploaded_docs"
        os.makedirs(upload_dir, exist_ok=True)

        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
            st.success(f"✅ File saved as: {file_path}")

        st.info("📄 Parsing PDF...")
        docs = extract_text_from_pdf(file_path)
        st.success(f"✅ Extracted {len(docs)} pages from PDF!")

        st.info("🧩 Chunking content...")
        st.session_state.chunked_docs = chunk_documents(docs)
        if not st.session_state.chunked_docs:
            st.error("❌ No chunks generated! Please check the input file.")
        else:
            st.success(f"✅ {len(st.session_state.chunked_docs)} chunks generated!")
            update_index_with_new_chunks(
                st.session_state.chunked_docs,
                index_path="data/embeddings/index.faiss",
                metadata_path="data/embeddings/metadata.json"
            )
            st.success("✅ Embeddings stored in FAISS index!")

if st.session_state.chunked_docs:
    chunks = st.session_state.chunked_docs
    st.subheader("🧩 View Chunks")
    chunk_range = st.slider("Select chunk range", 0, len(chunks), (0, min(20, len(chunks))))

    import numpy as np

    for chunk in chunks[chunk_range[0]:chunk_range[1]]:
        st.markdown(f"**📄 Source:** `{chunk['source']}` | **🧩 Chunk ID:** `{chunk['chunk_id']}`")
        st.code(chunk["text"], language="markdown")
        
        if "embedding" in chunk:
            vec = np.array(chunk["embedding"])
            
            with st.expander("🔍 Embedding Summary"):
                st.write({
                    "Dimensions": len(vec),
                    "Min": float(vec.min()),
                    "Max": float(vec.max()),
                    "Mean": float(vec.mean()),
                    "Std Dev": float(vec.std())
                })
                st.markdown("**📊 Embedding Distribution:**")
                st.line_chart(vec)
        else:
            st.warning("⚠️ No embedding available for this chunk.")
        
        st.markdown("---")
