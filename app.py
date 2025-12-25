import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import networkx as nx
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# --- 1. DATA LOADING & MULTI-LAYER PARSING ---
@st.cache_data(show_spinner="Fetching Corpora...")
def load_corpus(folder):
    api_url = f"https://api.github.com/repos/prihantoro-corpus/stylo/contents/{folder}"
    raw_base_url = f"https://raw.githubusercontent.com/prihantoro-corpus/stylo/main/{folder}/"
    corpus = {}
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            files = [f['name'] for f in response.json() if f['name'].endswith(('.txt', '.tsv'))]
            for name in files:
                r = requests.get(raw_base_url + name)
                if r.status_code == 200:
                    lines = r.text.strip().split('\n')
                    # Detect TreeTagger (3 columns)
                    if '\t' in lines[0]:
                        data = [line.split('\t') for line in lines if '\t' in line]
                        corpus[name] = {
                            'word': [row[0].lower() for row in data],
                            'tag': [row[1] for row in data],
                            'lemma': [row[2].lower() for row in data]
                        }
                    else:
                        words = [w for w in r.text.lower().split() if w.isalpha()]
                        corpus[name] = {'word': words, 'tag': [], 'lemma': []}
    except: pass
    return corpus

def build_matrix(corpus_dict, layer, mfw_limit, stops=[]):
    all_tokens = []
    for doc in corpus_dict.values():
        all_tokens.extend([t for t in doc[layer] if t not in stops])
    top_feats = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    matrix = [pd.Series(doc[layer]).value_counts().reindex(top_feats, fill_value=0) for doc in corpus_dict.values()]
    df = pd.DataFrame(matrix, index=corpus_dict.keys())
    return (df - df.mean()) / df.std().replace(0, 1), top_feats

# --- 2. APP CONFIG & SIDEBAR ---
st.set_page_config(page_title="Stylo-Lab Professional", layout="wide")
st.title("🔬 Stylometry Lab: Lexical, Structural & Attribution")

with st.sidebar:
    st.header("Selection")
    data_source = st.radio("Corpus", ["UNRESTRICTED-10", "TAGGED-10", "KNOWN-10", "Upload Files"])
    mfw_limit = st.slider("MFW Limit", 50, 1000, 500)
    use_stop = st.checkbox("Filter Stopwords")
    stop_list = st.text_area("Stopwords", "the, and, of, to, a, in").lower().split(",")

# --- 3. DATA PROCESSING ---
raw_data = {}
if data_source == "UNRESTRICTED-10": raw_data = load_corpus("preloaded1")
elif data_source == "TAGGED-10": raw_data = load_corpus("preloaded2")
elif data_source == "KNOWN-10":
    k = load_corpus("preloaded3/known3")
    q = load_corpus("preloaded3/question3")
    raw_data = {**k, **q}
else:
    uploaded = st.file_uploader("Upload files", accept_multiple_files=True)
    # [Upload logic here...]

if len(raw_data) > 2:
    z_word, feats_word = build_matrix(raw_data, 'word', mfw_limit, stop_list if use_stop else [])

    # --- SCENARIO 1: LEXICAL EXPLORER ---
    st.header("📦 Scenario 1: Lexical Explorer")
    t1, t2, t3, t4, t5 = st.tabs(["🌳 Dendrogram", "🗺️ PCA", "📈 Loadings", "🕸️ Network", "📊 CSV Data"])
    with t1:
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linkage(z_word, 'ward'), labels=list(raw_data.keys()), orientation='left', ax=ax)
        st.pyplot(fig)
    with t2:
        pca = PCA(n_components=2).fit_transform(z_word)
        fig, ax = plt.subplots()
        ax.scatter(pca[:,0], pca[:,1])
        for i, txt in enumerate(raw_data.keys()): ax.annotate(txt, (pca[i,0], pca[i,1]))
        st.pyplot(fig)
    # [T3, T4, T5 logic for Loadings, Networks, and CSVs...]

    # --- SCENARIO 2: STRUCTURAL EXPLORER ---
    if any(raw_data[next(iter(raw_data))]['tag']):
        st.divider()
        st.header("🧬 Scenario 2: Structural Explorer")
        z_tag, feats_tag = build_matrix(raw_data, 'tag', 100)
        z_lemma, feats_lemma = build_matrix(raw_data, 'lemma', mfw_limit)
        st.subheader("Grammar Profile (POS Ratios)")
        # ... [POS Ratio Bar Chart Code] ...
        

    # --- SCENARIO 3: ATTRIBUTION ---
    if data_source == "KNOWN-10":
        st.divider()
        st.header("🔍 Scenario 3: Lexical Attribution")
        k_idx = [i for i in z_word.index if i.startswith('K-')]
        q_idx = [i for i in z_word.index if i.startswith('Q-')]
        
        at1, at2, at3 = st.tabs(["🗺️ Attribution Zones", "🎯 Accuracy/Confusion", "🏆 Delta Rank"])
        
        with at1:
            st.subheader("PCA with Authorship Zones")
            labels = [n.split('-')[1] for n in k_idx]
            pca_mod = PCA(n_components=2)
            coords = pca_mod.fit_transform(z_word)
            fig, ax = plt.subplots()
            # SVM Zones
            svc = SVC(kernel='linear').fit(coords[:len(k_idx)], labels)
            # [Background mesh grid drawing code...]
            ax.scatter(coords[:len(k_idx),0], coords[:len(k_idx),1], c='blue', label='Known')
            ax.scatter(coords[len(k_idx):,0], coords[len(k_idx):,1], c='red', marker='x', label='Questioned')
            st.pyplot(fig)
            

        with at2:
            dist_mat = cdist(z_word.loc[q_idx], z_word.loc[k_idx], metric='cityblock')
            st.write("Distance Matrix (Lower is better)")
            st.dataframe(pd.DataFrame(dist_mat, index=q_idx, columns=k_idx).style.background_gradient(cmap='RdYlGn_r'))
            
        with at3:
            results = []
            for i, q in enumerate(q_idx):
                match = k_idx[np.argmin(dist_mat[i])]
                results.append(f"The text **{q}** is likely written by the same person who wrote **{match}**.")
            st.info("### 📝 Narration")
            for r in results: st.write(r)

else:
    st.info("Load a corpus to begin.")
