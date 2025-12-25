import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram

# --- CACHING FUNCTION ---
@st.cache_data(show_spinner="Downloading UNRESTRICTED-10 from GitHub...")
def load_preloaded_corpus():
    base_url = "https://raw.githubusercontent.com/prihantoro-corpus/stylo/main/preloaded1/"
    # List your EXACT filenames here
    file_names = [
        "blog1.txt", "blog2.txt", "blog3.txt", "blog4.txt", "blog5.txt",
        "news1.txt", "news2.txt", "news3.txt", "news4.txt", "news5.txt"
    ]
    loaded_corpus = {}
    for name in file_names:
        try:
            r = requests.get(base_url + name, timeout=5)
            if r.status_code == 200:
                tokens = [w for w in r.text.lower().split() if w.isalpha()]
                loaded_corpus[name] = tokens
        except:
            continue
    return loaded_corpus

st.set_page_config(page_title="Stylometry Lab", layout="wide")
st.title("🔬 Stylometry Lab")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Data Source")
    data_mode = st.radio("Choose Corpus", ["Upload My Own", "Pre-loaded: UNRESTRICTED-10"])
    
    st.header("2. Analysis Settings")
    mfw_limit = st.slider("MFW Limit", 50, 1000, 500)

# --- LOGIC ---
corpus = {}

if data_mode == "Pre-loaded: UNRESTRICTED-10":
    corpus = load_preloaded_corpus()
    if not corpus:
        st.error("Could not reach GitHub. Check filenames or connection.")
else:
    files = st.file_uploader("Upload .txt files", accept_multiple_files=True)
    if files:
        for f in files:
            text = f.read().decode("utf-8").lower()
            corpus[f.name] = [w for w in text.split() if w.isalpha()]

if len(corpus) > 1:
    # Build Matrix
    all_tokens = [t for tokens in corpus.values() for t in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix_data = []
    for name, tokens in corpus.items():
        counts = pd.Series(tokens).value_counts()
        matrix_data.append(counts.reindex(top_features, fill_value=0))
        
    df = pd.DataFrame(matrix_data, index=corpus.keys())
    z_scores = (df - df.mean()) / df.std().replace(0, 1)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    linkage_matrix = linkage(z_scores, method='ward')
    dendrogram(linkage_matrix, labels=list(corpus.keys()), ax=ax, orientation='left')
    st.pyplot(fig)
