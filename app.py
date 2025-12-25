import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Simple Stylo", layout="wide")

st.title("🔬 Simple Stylometry Lab")
st.write("Upload 2 or more .txt files to see how they cluster based on word usage.")

# 1. Simple Settings
mfw = st.sidebar.slider("Most Frequent Words", 100, 1000, 500)

# 2. File Upload (Untagged only)
files = st.file_uploader("Upload Raw Text Files (.txt)", accept_multiple_files=True)

if files and len(files) > 1:
    # 3. Processing Logic (Consolidated)
    corpus = {}
    for f in files:
        text = f.read().decode("utf-8").lower()
        # Basic tokenization: words only
        tokens = [word for word in text.split() if word.isalpha()]
        corpus[f.name] = tokens

    # Find the Most Frequent Words across all texts
    all_tokens = [t for tokens in corpus.values() for t in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw).index
    
    # Build Matrix
    matrix_data = []
    for name, tokens in corpus.items():
        counts = pd.Series(tokens).value_counts()
        row = counts.reindex(top_features, fill_value=0)
        matrix_data.append(row)
        
    df = pd.DataFrame(matrix_data, index=corpus.keys())
    
    # Standardize (Z-Scores)
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # 4. Display Results
    tab1, tab2 = st.tabs(["Cluster Map", "Word Frequencies"])

    with tab1:
        st.subheader("How similar are these texts?")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ward's method is the industry standard for Stylo
        linkage_matrix = linkage(z_scores, method='ward')
        dendrogram(linkage_matrix, labels=list(corpus.keys()), ax=ax, orientation='left')
        st.pyplot(fig)
        

    with tab2:
        st.subheader("Top Word Z-Scores (First 10 MFW)")
        st.dataframe(z_scores.iloc[:, :10])

elif files:
    st.info("Please upload at least two files to compare them.")
