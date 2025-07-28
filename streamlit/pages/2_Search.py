import streamlit as st
import sys
import os

# Ensure root path is available for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from processing.retriever import retrieve_chunks

st.set_page_config(page_title="Retriever", page_icon="🔍")
st.title("🔍 Retriever")

st.markdown(
    """
    Ask a question and the app will return the most relevant chunks from the knowledge base.
    Make sure the FAISS index and metadata are already generated.
    """
)

query = st.text_input("💬 Ask your question:")

top_k = st.slider("🔢 Number of results to show", 1, 100, 5)

if query:
    st.info("🔎 Searching for the most relevant content...")
    try:
        results = retrieve_chunks(query, top_k=top_k)
        if results:
            st.success(f"✅ Top {len(results)} results:")
            for i, chunk in enumerate(results, 1):
                st.markdown(f"### 📄 Result {i}")
                st.markdown(f"**📚 Source:** `{chunk.get('source', 'N/A')}`")
                st.markdown(f"**🔢 Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
                st.code(chunk["text"], language="markdown")
                st.markdown("---")
        else:
            st.warning("⚠️ No results found.")
    except Exception as e:
        st.error(f"❌ Error during search: {e}")
