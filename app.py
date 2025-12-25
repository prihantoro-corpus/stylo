import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# --- 1. DATA LOADING (CACHED) ---
@st.cache_data(show_spinner="Fetching Corpus...")
def load_github_corpus():
    api_url = "https://api.github.com/repos/prihantoro-corpus/stylo/contents/preloaded1"
    raw_base_url = "https://raw.githubusercontent.com/prihantoro-corpus/stylo/main/preloaded1/"
    loaded_corpus = {}
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            txt_files = [file['name'] for file in response.json() if file['name'].endswith('.txt')]
            for name in txt_files:
                r = requests.get(raw_base_url + name)
                if r.status_code == 200:
                    loaded_corpus[name] = [w for w in r.text.lower().split() if w.isalpha()]
    except: pass
    return loaded_corpus

st.set_page_config(page_title="Lexical Explorer", layout="wide")
st.title("🔬 Scenario 1: Lexical Explorer")

# --- 2. SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Parameters")
    data_mode = st.radio("Corpus", ["Upload My Own", "Pre-loaded: UNRESTRICTED-10"])
    mfw_limit = st.slider("MFW Limit", 50, 2000, 500)
    viz_mode = st.selectbox("Map Type", ["PCA (Linear)", "MDS (Distance-based)"])

# --- 3. CORE ANALYTICS ENGINE ---
corpus = load_github_corpus() if data_mode == "Pre-loaded: UNRESTRICTED-10" else {}
if data_mode == "Upload My Own":
    files = st.file_uploader("Upload .txt", accept_multiple_files=True)
    for f in files:
        corpus[f.name] = [w for w in f.read().decode("utf-8").lower().split() if w.isalpha()]

if len(corpus) > 2:
    # A. Matrix Building
    all_tokens = [t for tokens in corpus.values() for t in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix_data = [pd.Series(tokens).value_counts().reindex(top_features, fill_value=0) for tokens in corpus.values()]
    df = pd.DataFrame(matrix_data, index=corpus.keys())
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # --- 4. OUTPUT TABS ---
    tabs = st.tabs(["🌳 Clustering", "🗺️ Spatial Map", "📈 Loadings", "🕸️ Bootstrap Network", "📊 Data"])

    # TAB 1: DENDROGRAM
    with tabs[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage(z_scores, method='ward'), labels=list(corpus.keys()), ax=ax, orientation='left')
        st.pyplot(fig)

    # TAB 2: PCA / MDS MAP
    with tabs[1]:
        st.subheader(f"{viz_mode} Visualization")
        model = PCA(n_components=2) if "PCA" in viz_mode else MDS(n_components=2, dissimilarity='precomputed')
        
        # If MDS, we need a distance matrix first
        coords = model.fit_transform(z_scores) if "PCA" in viz_mode else model.fit_transform(squareform(pdist(z_scores, metric='cityblock')))
        
        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1], c='red')
        for i, txt in enumerate(corpus.keys()):
            ax.annotate(txt, (coords[i, 0], coords[i, 1]))
        st.pyplot(fig)
        

    # TAB 3: LOADING PLOTS (Marker Words)
    with tabs[2]:
        st.subheader("Word Contributions (PCA Loadings)")
        pca_full = PCA(n_components=2).fit(z_scores)
        loadings = pd.DataFrame(pca_full.components_.T, index=top_features, columns=['PC1', 'PC2'])
        st.write("Words driving the differences:")
        st.dataframe(loadings.sort_values('PC1', ascending=False).head(20))
        

    # TAB 4: BOOTSTRAP NETWORK (Simplified Consensus)
    with tabs[3]:
        st.subheader("Consensus Network")
        G = nx.Graph()
        dist_matrix = squareform(pdist(z_scores, metric='cityblock'))
        # Connect nodes if they are "close" (below median distance)
        threshold = np.median(dist_matrix) * 0.6
        for i, name_i in enumerate(corpus.keys()):
            for j, name_j in enumerate(corpus.keys()):
                if i < j and dist_matrix[i, j] < threshold:
                    G.add_edge(name_i, name_j, weight=1/dist_matrix[i, j])
        
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color='lightblue', font_size=8, ax=ax)
        st.pyplot(fig)
        

    # TAB 5: CSV EXPORTS
    with tabs[4]:
        col1, col2 = st.columns(2)
        dist_df = pd.DataFrame(squareform(pdist(z_scores, metric='cityblock')), index=corpus.keys(), columns=corpus.keys())
        col1.download_button("Download Distance Matrix (CSV)", dist_df.to_csv(), "distances.csv")
        col2.download_button("Download Z-Scores (CSV)", z_scores.to_csv(), "zscores.csv")
        st.dataframe(z_scores.head(10))

else:
    st.info("Upload at least 3 files to enable Spatial Mapping and Networks.")
