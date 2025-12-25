import streamlit as st
import engine
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stylo Python", layout="wide")

st.title("🔬 Stylometry Lab")

# 1. Sidebar Setup
with st.sidebar:
    st.header("Settings")
    is_tagged = st.toggle("Use TreeTagger Files")
    has_known = st.toggle("Include Known Authors")
    mfw = st.slider("MFW Limit", 100, 2000, 500)
    
    tag_col = 0
    if is_tagged:
        mode = st.selectbox("Feature Layer", ["Word", "POS Tag", "Lemma"])
        tag_col = {"Word": 0, "POS Tag": 1, "Lemma": 2}[mode]

# 2. File Upload
uploader_label = "Upload TreeTagger (.tsv)" if is_tagged else "Upload Raw (.txt)"
files = st.file_uploader(uploader_label, accept_multiple_files=True)

if files:
    corpus = {}
    for f in files:
        if is_tagged:
            corpus[f.name] = engine.parse_treetagger(f, tag_col)
        else:
            corpus[f.name] = f.read().decode("utf-8").lower().split()

    # 3. Process
    z_matrix = engine.build_frequency_matrix(corpus, mfw_limit=mfw)
    delta_matrix = engine.calculate_delta(z_matrix)

    # 4. Outputs
    tab1, tab2 = st.tabs(["Clustering", "Data Matrix"])
    
    with tab1:
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 7))
        linked = linkage(z_matrix, 'ward')
        dendrogram(linked, labels=list(corpus.keys()), ax=ax, orientation='left')
        st.pyplot(fig)
        

    with tab2:
        st.subheader("Z-Score Matrix (Top 10 Features)")
        st.dataframe(z_matrix.iloc[:, :10])
