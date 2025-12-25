import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Stylometry Lab", layout="wide")

st.title("🔬 Stylometry Lab")

# --- SETTINGS & DATA SOURCE ---
with st.sidebar:
    st.header("1. Data Source")
    data_mode = st.radio("Choose Corpus", ["Upload My Own", "Pre-loaded: UNRESTRICTED-10"])
    
    st.header("2. Analysis Settings")
    mfw_limit = st.slider("MFW (Most Frequent Words)", 50, 1000, 500)

# --- DATA INGESTION ---
corpus = {}

if data_mode == "Pre-loaded: UNRESTRICTED-10":
    # List of files in your preloaded1 folder
    base_url = "https://raw.githubusercontent.com/prihantoro-corpus/stylo/main/preloaded1/"
    file_names = [f"text{i}.txt" for i in range(1, 11)] # Assuming names are text1.txt to text10.txt
    
    st.info("Fetching UNRESTRICTED-10 corpus from GitHub...")
    
    for name in file_names:
        try:
            response = requests.get(base_url + name)
            if response.status_code == 200:
                text = response.text.lower()
                tokens = [word for word in text.split() if word.isalpha()]
                corpus[name] = tokens
        except Exception as e:
            st.error(f"Failed to load {name}")

else:
    files = st.file_uploader("Upload Raw Text Files (.txt)", accept_multiple_files=True)
    if files:
        for f in files:
            text = f.read().decode("utf-8").lower()
            tokens = [word for word in text.split() if word.isalpha()]
            corpus[f.name] = tokens

# --- ANALYSIS ENGINE ---
if len(corpus) > 1:
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
    
    # 3. Standardization (Z-Scores)
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # --- OUTPUTS ---
    tab1, tab2 = st.tabs(["📊 Dendrogram", "🧪 Distance Matrix"])

    with tab1:
        st.subheader("Stylistic Clustering")
        fig, ax = plt.subplots(figsize=(10, 8))
        linkage_matrix = linkage(z_scores, method='ward')
        dendrogram(linkage_matrix, labels=list(corpus.keys()), ax=ax, orientation='left')
        plt.title(f"Cluster Analysis: {data_mode}")
        st.pyplot(fig)
        

    with tab2:
        st.subheader("Manhattan Distance Matrix")
        dist_matrix = squareform(pdist(z_scores, metric='cityblock'))
        dist_df = pd.DataFrame(dist_matrix, index=corpus.keys(), columns=corpus.keys())
        st.dataframe(dist_df.style.background_gradient(cmap='Greens'))
        

elif data_mode == "Upload My Own":
    st.info("Please upload at least two files to begin.")
