import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# --- 1. ENGINE LOGIC (PASTED DIRECTLY HERE) ---

def parse_treetagger(file, column_index):
    """Parses a TreeTagger file and returns the selected column as a list."""
    df = pd.read_csv(file, sep='\t', names=['word', 'tag', 'lemma'], quoting=3)
    return df.iloc[:, column_index].astype(str).tolist()

def build_frequency_matrix(corpus_dict, mfw_limit=500):
    """Converts a dictionary of {doc_name: [tokens]} into a frequency matrix."""
    all_tokens = [token for tokens in corpus_dict.values() for token in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix = []
    for doc, tokens in corpus_dict.items():
        counts = pd.Series(tokens).value_counts()
        row = counts.reindex(top_features, fill_value=0)
        matrix.append(row)
        
    df_matrix = pd.DataFrame(matrix, index=corpus_dict.keys())
    # Handle division by zero if std is 0
    df_zscore = (df_matrix - df_matrix.mean()) / df_matrix.std().replace(0, 1)
    return df_zscore

def calculate_delta(zscore_matrix):
    """Calculates Manhattan distance (Burrows Delta) between all documents."""
    distances = pdist(zscore_matrix, metric='cityblock') / zscore_matrix.shape[1]
    return pd.DataFrame(squareform(distances), 
                        index=zscore_matrix.index, 
                        columns=zscore_matrix.index)

# --- 2. STREAMLIT UI CODE ---

st.set_page_config(page_title="Stylo Python", layout="wide")
st.title("🔬 Stylometry Lab")

# Sidebar Setup
with st.sidebar:
    st.header("Settings")
    is_tagged = st.toggle("Use TreeTagger Files")
    has_known = st.toggle("Include Known Authors")
    mfw = st.slider("MFW Limit", 100, 2000, 500)
    
    tag_col = 0
    if is_tagged:
        mode = st.selectbox("Feature Layer", ["Word", "POS Tag", "Lemma"])
        tag_col = {"Word": 0, "POS Tag": 1, "Lemma": 2}[mode]

# File Upload
uploader_label = "Upload TreeTagger (.tsv)" if is_tagged else "Upload Raw (.txt)"
files = st.file_uploader(uploader_label, accept_multiple_files=True)

if files:
    corpus = {}
    for f in files:
        if is_tagged:
            # We call the function directly now, no 'engine.' prefix
            corpus[f.name] = parse_treetagger(f, tag_col)
        else:
            corpus[f.name] = f.read().decode("utf-8").lower().split()

    if len(corpus) > 1: # Need at least 2 files to cluster
        # Process
        z_matrix = build_frequency_matrix(corpus, mfw_limit=mfw)
        delta_matrix = calculate_delta(z_matrix)

        # Outputs
        tab1, tab2 = st.tabs(["Clustering", "Data Matrix"])
        
        with tab1:
            st.subheader("Dendrogram")
            fig, ax = plt.subplots(figsize=(10, 7))
            # Perform clustering
            linked = linkage(z_matrix, 'ward')
            dendrogram(linked, labels=list(corpus.keys()), ax=ax, orientation='left')
            st.pyplot(fig)
            

        with tab2:
            st.subheader("Z-Score Matrix (Top 10 Features)")
            st.dataframe(z_matrix.iloc[:, :10])
    else:
        st.info("Please upload at least two files to see a comparison.")
