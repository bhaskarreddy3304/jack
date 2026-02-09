import streamlit as st
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Legal Document Analysis",
    layout="wide"
)

# --------------------------------------------------
# DARK THEME
# --------------------------------------------------
st.markdown("""
<style>
html, body, .stApp { background-color: #000; color: #fff; }
h1, h2, h3 { color: #fff; }
input, textarea {
    background-color: #000 !important;
    color: #fff !important;
    border: 1px solid #8b5cf6 !important;
}
button {
    background-color: #000 !important;
    color: #fff !important;
    border: 1px solid #8b5cf6 !important;
}
button:hover { background-color: #8b5cf6 !important; color: #000 !important; }
.answer-box {
    border-left: 4px solid #8b5cf6;
    padding: 14px;
    background-color: #050505;
    margin-bottom: 10px;
}
.source-box {
    border: 1px solid #222;
    padding: 8px;
    background-color: #020202;
    margin-bottom: 6px;
}
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1 style='text-align:center;'>LEGAL DOCUMENT ANALYSIS & Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Semantic Search on Legal PDFs</p>", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("### Upload Legal PDF Files")
uploaded_files = st.sidebar.file_uploader(
    "PDF only",
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
# BUILD SEARCH INDEX (NO FAISS)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def build_search_index(files):
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

    nn = NearestNeighbors(n_neighbors=4, metric="cosine")
    nn.fit(embeddings)

    return nn, texts, sources

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
if uploaded_files:
    nn, texts, sources = build_search_index(uploaded_files)

    query = st.text_input("Ask a legal question")

    if query:
        with st.spinner("Searching documents..."):
            query_embedding = model.encode([query])
            distances, indices = nn.kneighbors(query_embedding)

        st.markdown("### Relevant Results")
        for idx in indices[0]:
            st.markdown(
                f"<div class='answer-box'>{texts[idx]}</div>",
                unsafe_allow_html=True
            )

        st.markdown("### Source Documents")
        for idx in indices[0]:
            st.markdown(
                f"<div class='source-box'>{sources[idx]}</div>",
                unsafe_allow_html=True
            )
else:
    st.info("Upload legal PDF documents from the sidebar to begin.")
