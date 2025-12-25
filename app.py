import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Stylometry Lab", layout="wide")

st.title("🔬 Stylometry Lab")
st.markdown("Upload multiple `.txt` files to see how they cluster based on **Most Frequent Words (MFW)**.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Analysis Settings")
    mfw_limit = st.slider("MFW (Most Frequent Words)", 50, 1000, 500)
    st.info("Higher MFW looks at common grammar/style; lower MFW focuses on core vocabulary.")

# --- FILE UPLOAD ---
files = st.file_uploader("Upload Raw Text Files (.txt)", accept_multiple_files=True)

if files and len(files) > 1:
    corpus = {}
    
    # Process files into tokens
    for f in files:
        try:
            text = f.read().decode("utf-8").lower()
            # Simple tokenization: keep only alphabetic words
            tokens = [word for word in text.split() if word.isalpha()]
            corpus[f.name] = tokens
        except Exception as e:
            st.error(f"Error reading {f.name}: {e}")

    # 1. Identify Global MFW
    all_tokens = [t for tokens in corpus.values() for t in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    # 2. Build Frequency Matrix
    matrix_data = []
    for name, tokens in corpus.items():
        counts = pd.Series(tokens).value_counts()
        row = counts.reindex(top_features, fill_value=0)
        matrix_data.append(row)
        
    df = pd.DataFrame(matrix_data, index=corpus.keys())
    
    # 3. Standardization (Z-Scores / Burrows's Delta prep)
    # We replace 0 std with 1 to avoid division by zero errors
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # --- OUTPUTS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dendrogram", "🧪 Distance Matrix", "📋 Z-Score Data"])

    with tab1:
        st.subheader("Stylistic Clustering")
        st.write("This tree shows how 'close' texts are stylistically. Shorter horizontal branches indicate higher similarity.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ward's linkage provides the most distinct clusters for stylometry
        linkage_matrix = linkage(z_scores, method='ward')
        dendrogram(linkage_matrix, labels=list(corpus.keys()), ax=ax, orientation='left')
        
        plt.title(f"Cluster Analysis (Top {mfw_limit} MFW)")
        plt.xlabel("Distance (Ward)")
        st.pyplot(fig)
        

    with tab2:
        st.subheader("Manhattan Distance Matrix")
        # Calculate the actual "Delta" distance
        dist_matrix = squareform(pdist(z_scores, metric='cityblock'))
        dist_df = pd.DataFrame(dist_matrix, index=corpus.keys(), columns=corpus.keys())
        st.dataframe(dist_df.style.background_gradient(cmap='Blues'))

    with tab3:
        st.subheader("MFW Feature Table")
        st.write("The standardized frequency of your top features.")
        st.dataframe(z_scores)

elif files:
    st.warning("⚠️ Please upload at least **two** files to perform a comparative analysis.")
else:
    st.info("👋 Welcome! Please upload your text files in the sidebar or main area to begin.")
