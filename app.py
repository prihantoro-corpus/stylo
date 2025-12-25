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
    
    st.header("Exclusion List (Stopwords)")
    use_exclusion = st.checkbox("Enable Stopword Filtering", value=False)
    
    # Pre-defined list of common English function words
    default_stops = "the, and, of, to, a, in, is, it, that, was, as, for, with, on, be, at, by, this, had, not, are, but, from, or, have, an, they, which, one, you, were, her, all, she, there, would, their, we, him, been, has, when, who, will, no, if, out, so, said, what, up, its, about, into, than, them, can, only, other, new, some, could, time, these, two, may, then, do, first, any, my, now, such, like, our, over, man, me, even, most, made, after, also, did, many, before, must, through, back, years, where, much, your, way, well, down, should, because, each, just, those, people, mr, how"
    
    # Button to auto-fill the text area
    if st.button("Load Default English Stopwords"):
        st.session_state.stop_text = default_stops
    
    # The text area looks for content in session_state first
    stop_input = st.text_area(
        "Edit excluded words:",
        value=st.session_state.get('stop_text', ""),
        placeholder="Type words here, separated by commas..."
    )
    
    stop_list = [w.strip().lower() for w in stop_input.split(",") if w.strip()]

# --- 3. CORE ANALYTICS ENGINE ---
corpus = load_github_corpus() if data_mode == "Pre-loaded: UNRESTRICTED-10" else {}
if data_mode == "Upload My Own":
    files = st.file_uploader("Upload .txt", accept_multiple_files=True)
    if files:
        for f in files:
            corpus[f.name] = [w for w in f.read().decode("utf-8").lower().split() if w.isalpha()]

if len(corpus) > 2:
    # APPLY EXCLUSION LIST
    processed_corpus = {}
    for name, tokens in corpus.items():
        if use_exclusion and stop_list:
            processed_corpus[name] = [w for w in tokens if w not in stop_list]
        else:
            processed_corpus[name] = tokens

    # Matrix Building & Normalization
    all_tokens = [t for tokens in processed_corpus.values() for t in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix_data = [pd.Series(tokens).value_counts().reindex(top_features, fill_value=0) for tokens in processed_corpus.values()]
    df = pd.DataFrame(matrix_data, index=processed_corpus.keys())
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # --- 4. OUTPUT TABS ---
    tabs = st.tabs(["🌳 Clustering", "🗺️ Spatial Map", "📈 Loadings", "🕸️ Bootstrap Network", "📊 Data"])

    with tabs[0]:
        st.subheader("Dendrogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        dendrogram(linkage(z_scores, method='ward'), labels=list(corpus.keys()), ax=ax, orientation='left')
        st.pyplot(fig)
        

    with tabs[1]:
        st.subheader(f"{viz_mode} Visualization")
        model = PCA(n_components=2) if "PCA" in viz_mode else MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
        coords = model.fit_transform(z_scores) if "PCA" in viz_mode else model.fit_transform(squareform(pdist(z_scores, metric='cityblock')))
        
        fig, ax = plt.subplots()
        ax.scatter(coords[:, 0], coords[:, 1], c='skyblue', edgecolors='navy')
        for i, txt in enumerate(corpus.keys()):
            ax.annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=9)
        st.pyplot(fig)
        

    with tabs[2]:
        st.subheader("Word Contributions (Loadings)")
        pca_full = PCA(n_components=2).fit(z_scores)
        loadings = pd.DataFrame(pca_full.components_.T, index=top_features, columns=['PC1', 'PC2'])
        st.write("Top 20 words influencing the primary separation (PC1):")
        st.dataframe(loadings.sort_values('PC1', ascending=False).head(20))

    with tabs[3]:
        st.subheader("Consensus Network")
        G = nx.Graph()
        dist_matrix = squareform(pdist(z_scores, metric='cityblock'))
        threshold = np.percentile(dist_matrix, 25) # Connect the top 25% closest pairs
        for i, name_i in enumerate(corpus.keys()):
            for j, name_j in enumerate(corpus.keys()):
                if i < j and dist_matrix[i, j] < threshold:
                    G.add_edge(name_i, name_j)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color='plum', edge_color='gray', node_size=800, font_size=8, ax=ax)
        st.pyplot(fig)
        

    with tabs[4]:
        st.subheader("Export & Raw Data")
        st.dataframe(z_scores)
        st.download_button("Download Z-Scores (CSV)", z_scores.to_csv(), "zscores.csv")

else:
    st.info("Please upload or load at least 3 texts to begin.")
