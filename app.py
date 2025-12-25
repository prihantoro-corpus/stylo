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

# --- 1. DATA LOADING & PARSING ---
@st.cache_data(show_spinner="Fetching Corpora...")
def load_github_corpus(folder):
    api_url = f"https://api.github.com/repos/prihantoro-corpus/stylo/contents/{folder}"
    raw_base_url = f"https://raw.githubusercontent.com/prihantoro-corpus/stylo/main/{folder}/"
    loaded_corpus = {}
    try:
        response = requests.get(api_url)
        if response.status_code == 200:
            files = [f['name'] for f in response.json() if f['name'].endswith(('.txt', '.tsv'))]
            for name in files:
                r = requests.get(raw_base_url + name)
                if r.status_code == 200:
                    lines = r.text.strip().split('\n')
                    # Parse TreeTagger (3 cols) vs Raw (1 col)
                    if folder == "preloaded2" or name.endswith('.tsv'):
                        data = [line.split('\t') for line in lines if '\t' in line]
                        loaded_corpus[name] = {
                            'word': [row[0].lower() for row in data],
                            'tag': [row[1] for row in data],
                            'lemma': [row[2].lower() for row in data]
                        }
                    else:
                        words = [w for w in r.text.lower().split() if w.isalpha()]
                        loaded_corpus[name] = {'word': words, 'tag': [], 'lemma': []}
    except: pass
    return loaded_corpus

def build_z_matrix(corpus_dict, layer, mfw_limit, stop_list=[]):
    all_tokens = []
    for doc in corpus_dict.values():
        all_tokens.extend([t for t in doc[layer] if t not in stop_list])
    
    top_feats = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix = []
    for doc in corpus_dict.values():
        counts = pd.Series(doc[layer]).value_counts()
        matrix.append(counts.reindex(top_feats, fill_value=0))
    
    df = pd.DataFrame(matrix, index=corpus_dict.keys())
    return (df - df.mean()) / df.std().replace(0, 1), top_feats

# --- 2. UI SETUP ---
st.set_page_config(page_title="Stylo-Lab Professional", layout="wide")
st.title("🔬 Stylometry Lab: Lexical & Structural")

with st.sidebar:
    st.header("Data Selection")
    data_source = st.radio("Corpus", ["Upload Files", "UNRESTRICTED-10 (Raw)", "TAGGED-10 (TreeTagger)"])
    mfw_limit = st.slider("MFW Limit", 50, 1000, 500)
    
    st.header("Exclusion")
    use_stop = st.checkbox("Filter Stopwords")
    stop_input = st.text_area("Stopwords (comma separated)", "the, and, of, to, a, in, is")
    stop_list = [w.strip().lower() for w in stop_input.split(",") if w.strip()]

# --- 3. DATA PROCESSING ---
raw_data = {}
if "UNRESTRICTED" in data_source:
    raw_data = load_github_corpus("preloaded1")
elif "TAGGED" in data_source:
    raw_data = load_github_corpus("preloaded2")
else:
    files = st.file_uploader("Upload .txt or .tsv", accept_multiple_files=True)
    for f in files:
        content = f.read().decode("utf-8")
        if f.name.endswith('.tsv'):
            data = [line.split('\t') for line in content.strip().split('\n') if '\t' in line]
            raw_data[f.name] = {'word': [r[0].lower() for r in data], 'tag': [r[1] for r in data], 'lemma': [r[2].lower() for r in data]}
        else:
            raw_data[f.name] = {'word': [w for w in content.lower().split() if w.isalpha()], 'tag': [], 'lemma': []}

if len(raw_data) > 2:
    # --- SCENARIO 1: LEXICAL EXPLORER (WORDS) ---
    st.header("📦 Scenario 1: Lexical Explorer (Words)")
    z_word, feats_word = build_z_matrix(raw_data, 'word', mfw_limit, stop_list if use_stop else [])
    
    tabs1 = st.tabs(["🌳 Dendrogram", "🗺️ PCA Map", "📈 Loadings", "🕸️ Network", "📊 Data"])
    with tabs1[0]:
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linkage(z_word, 'ward'), labels=list(raw_data.keys()), orientation='left', ax=ax)
        st.pyplot(fig)
        
    with tabs1[1]:
        pca = PCA(n_components=2).fit_transform(z_word)
        fig, ax = plt.subplots()
        ax.scatter(pca[:,0], pca[:,1])
        for i, txt in enumerate(raw_data.keys()): ax.annotate(txt, (pca[i,0], pca[i,1]))
        st.pyplot(fig)
    with tabs1[2]:
        loadings = pd.DataFrame(PCA(n_components=2).fit(z_word).components_.T, index=feats_word, columns=['PC1', 'PC2'])
        st.dataframe(loadings.sort_values('PC1', ascending=False).head(15))
    with tabs1[3]:
        # Simple proximity network
        G = nx.from_numpy_array(squareform(pdist(z_word, 'cityblock')) < np.median(pdist(z_word)), create_using=nx.Graph)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, labels={i:n for i,n in enumerate(raw_data.keys())}, ax=ax)
        st.pyplot(fig)
    with tabs1[4]:
        st.download_button("Download Word Matrix", z_word.to_csv(), "word_matrix.csv")

    st.divider()

    # --- SCENARIO 2: STRUCTURAL EXPLORER (TAGS/LEMMAS) ---
    st.header("🧬 Scenario 2: Structural Explorer (Tags & Lemmas)")
    
    if not any(raw_data[next(iter(raw_data))]['tag']):
        st.warning("Scenario 2 requires TreeTagger files. Please load TAGGED-10 or upload .tsv files.")
    else:
        z_tag, feats_tag = build_z_matrix(raw_data, 'tag', 100) # Tags are finite, so lower MFW
        z_lemma, feats_lemma = build_z_matrix(raw_data, 'lemma', mfw_limit)

        tabs2 = st.tabs(["🌳 Dendrograms", "🗺️ Grammar Map", "📈 Tag Loadings", "📊 Grammar Profile", "🧪 Data"])
        
        with tabs2[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Lemma-based")
                fig1, ax1 = plt.subplots()
                dendrogram(linkage(z_lemma, 'ward'), labels=list(raw_data.keys()), orientation='left', ax=ax1)
                st.pyplot(fig1)
            with col2:
                st.subheader("Tag-based")
                fig2, ax2 = plt.subplots()
                dendrogram(linkage(z_tag, 'ward'), labels=list(raw_data.keys()), orientation='left', ax=ax2)
                st.pyplot(fig2)

        with tabs2[1]:
            st.subheader("Grammar-based PCA (POS Tags)")
            pca_tag = PCA(n_components=2).fit_transform(z_tag)
            fig, ax = plt.subplots()
            ax.scatter(pca_tag[:,0], pca_tag[:,1], color='green')
            for i, txt in enumerate(raw_data.keys()): ax.annotate(txt, (pca_tag[i,0], pca_tag[i,1]))
            st.pyplot(fig)
            

        with tabs2[2]:
            st.subheader("Tag Contributions")
            loadings_tag = pd.DataFrame(PCA(n_components=2).fit(z_tag).components_.T, index=feats_tag, columns=['PC1', 'PC2'])
            st.dataframe(loadings_tag.sort_values('PC1', ascending=False))

        with tabs2[3]:
            st.subheader("Grammar Profile (POS Ratios)")
            profile_data = []
            for name, d in raw_data.items():
                ratios = pd.Series(d['tag']).value_counts(normalize=True).head(10)
                profile_data.append(ratios)
            profile_df = pd.DataFrame(profile_data, index=raw_data.keys()).fillna(0)
            st.bar_chart(profile_df)
            

        with tabs2[4]:
            st.download_button("Download Tag Matrix", z_tag.to_csv(), "tag_matrix.csv")

else:
    st.info("Please load or upload files to begin analysis.")
