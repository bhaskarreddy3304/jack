import streamlit as st
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Document Analysis – RAG",
    layout="wide"
)

# --------------------------------------------------
# DARK THEME (STREAMLIT CLOUD SAFE)
# --------------------------------------------------
st.markdown("""
<style>
html, body, .stApp {
    background-color: #000000;
    color: #ffffff;
}
h1, h2, h3 {
    color: #ffffff;
}
input, textarea {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #8b5cf6 !important;
}
button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1px solid #8b5cf6 !important;
}
button:hover {
    background-color: #8b5cf6 !important;
    color: #000000 !important;
}
.answer-box {
    border-left: 4px solid #8b5cf6;
    padding: 14px;
    background-color: #050505;
}
.source-box {
    border: 1px solid #222;
    padding: 8px;
    margin-bottom: 6px;
    background-color: #020202;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>LEGAL DOCUMENT ANALYSIS & Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Semantic Search over Legal PDFs</p>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EMBEDDING MODEL (CLOUD SAFE)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# SIDEBAR – FILE UPLOAD
# --------------------------------------------------
st.sidebar.markdown("### Upload Legal PDFs")
uploaded_files = st.sidebar.file_uploader(
    "PDF files only",
    type=["pdf"],
    accept_multiple_files=True
)

# --------------------------------------------------
# TEXT SPLITTER
# --------------------------------------------------
def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# --------------------------------------------------
# BUILD VECTOR STORE
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_vector_store(files):
    texts = []
    sources = []

    for file in files:
        reader = PdfReader(file)
        full_text = ""
        for page in reader.pages:
            if page.extract_text():
                full_text += page.extract_text()

        chunks = split_text(full_text)
        texts.extend(chunks)
        sources.extend([file.name] * len(chunks))

    embeddings = model.encode(texts)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, texts, sources

# --------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------
if uploaded_files:
    index, texts, sources = build_vector_store(uploaded_files)

    query = st.text_input("Ask a legal question")

    if query:
        with st.spinner("Searching documents..."):
            query_embedding = model.encode([query]).astype("float32")
            distances, indices = index.search(query_embedding, 4)

        st.markdown("### Answer (Relevant Extracts)")
        for i in indices[0]:
            st.markdown(
                f"<div class='answer-box'>{texts[i]}</div>",
                unsafe_allow_html=True
            )

        st.markdown("### Source Documents")
        for i in indices[0]:
            st.markdown(
                f"<div class='source-box'>{sources[i]}</div>",
                unsafe_allow_html=True
            )
else:
    st.info("Upload legal PDF files from the sidebar to begin.")
