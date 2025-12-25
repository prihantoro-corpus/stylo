import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# --- 1. DATA LOADING ---
@st.cache_data(show_spinner="Connecting to GitHub...")
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
        else:
            st.error("GitHub API limit reached or folder moved.")
    except Exception as e:
        st.error(f"Connection Error: {e}")
    return loaded_corpus

st.set_page_config(page_title="Stylometry Lab", layout="wide")
st.title("🔬 Stylometry Lab")

# --- 2. SIDEBAR ---
with st.sidebar:
    st.header("Data Source")
    data_mode = st.radio("Choose Corpus", ["Upload My Own", "Pre-loaded: UNRESTRICTED-10"])
    mfw_limit = st.slider("MFW Limit (Top Words)", 50, 1000, 500)

# --- 3. PROCESSING ---
corpus = {}
if data_mode == "Pre-loaded: UNRESTRICTED-10":
    corpus = load_github_corpus()
else:
    files = st.file_uploader("Upload .txt files", accept_multiple_files=True)
    if files:
        for f in files:
            text = f.read().decode("utf-8").lower()
            corpus[f.name] = [w for w in text.split() if w.isalpha()]

if len(corpus) > 1:
    with st.spinner("Calculating Z-Scores and Distances..."):
        # A. Feature Extraction
        all_tokens = [t for tokens in corpus.values() for t in tokens]
        top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
        
        matrix_data = []
        for name, tokens in corpus.items():
            counts = pd.Series(tokens).value_counts()
            matrix_data.append(counts.reindex(top_features, fill_value=0))
            
        df = pd.DataFrame(matrix_data, index=corpus.keys())
        
        # B. Z-SCORE FEATURE (Standardization)
        z_scores = (df - df.mean()) / df.std().replace(0, 1)

        # C. DISTANCE MATRIX FEATURE (Manhattan/Delta)
        # Often called the confusion matrix in attribution tasks
        dist_matrix = squareform(pdist(z_scores, metric='cityblock'))
        dist_df = pd.DataFrame(dist_matrix, index=corpus.keys(), columns=corpus.keys())

        # --- 4. TABS (THE THREE FEATURES) ---
        tab1, tab2, tab3 = st.tabs(["📊 Dendrogram", "🧪 Distance Matrix", "📋 Z-Score Table"])

        with tab1:
            st.subheader("Hierarchical Clustering")
            fig, ax = plt.subplots(figsize=(10, 7))
            linkage_matrix = linkage(z_scores, method='ward')
            dendrogram(linkage_matrix, labels=list(corpus.keys()), ax=ax, orientation='left')
            st.pyplot(fig)
            

        with tab2:
            st.subheader("Distance Matrix (Burrows's Delta)")
            st.write("Lower values (darker blue) indicate higher stylistic similarity.")
            st.dataframe(dist_df.style.background_gradient(cmap='Blues'))
            

        with tab3:
            st.subheader("MFW Z-Scores")
            st.write("Normalized frequencies for the top features.")
            st.dataframe(z_scores)

else:
    st.info("Please provide at least two texts to compare.")
