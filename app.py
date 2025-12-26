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
def load_corpus(folder_path):
    """
    Recursively fetches files from GitHub to handle nested subfolders
    like preloaded3/known3 and preloaded3/question3.
    """
    api_base = "https://api.github.com/repos/prihantoro-corpus/stylo/contents"
    raw_base = "https://raw.githubusercontent.com/prihantoro-corpus/stylo/main"
    corpus = {}

    def fetch_recursive(current_path):
        api_url = f"{api_base}/{current_path}"
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                items = response.json()
                for item in items:
                    if item['type'] == 'dir':
                        # Dig into subdirectories
                        fetch_recursive(f"{current_path}/{item['name']}")
                    elif item['name'].endswith(('.txt', '.tsv')):
                        raw_url = f"{raw_base}/{current_path}/{item['name']}"
                        r = requests.get(raw_url)
                        if r.status_code == 200:
                            lines = r.text.strip().split('\n')
                            # Detect TreeTagger (3 columns)
                            if '\t' in lines[0]:
                                data = [
                                    line.split('\t') for line in lines
                                    if '\t' in line
                                ]
                                corpus[item['name']] = {
                                    'word': [
                                        row[0].lower() for row in data
                                        if len(row) > 0
                                    ],
                                    'tag':
                                    [row[1] for row in data if len(row) > 1],
                                    'lemma': [
                                        row[2].lower() for row in data
                                        if len(row) > 2
                                    ]
                                }
                            else:  # Plain Text
                                words = [
                                    w for w in r.text.lower().split()
                                    if w.isalpha()
                                ]
                                corpus[item['name']] = {
                                    'word': words,
                                    'tag': [],
                                    'lemma': []
                                }
        except:
            pass

    fetch_recursive(folder_path)
    return corpus


def build_matrix(corpus_dict, layer, mfw_limit, stops=[]):
    all_tokens = []
    for doc in corpus_dict.values():
        all_tokens.extend([t for t in doc[layer] if t not in stops])

    if not all_tokens:
        return pd.DataFrame(), []

    top_feats = pd.Series(all_tokens).value_counts().head(mfw_limit).index

    matrix = []
    for doc in corpus_dict.values():
        counts = pd.Series(doc[layer]).value_counts()
        matrix.append(counts.reindex(top_feats, fill_value=0))

    df = pd.DataFrame(matrix, index=corpus_dict.keys())
    # Standardize (Z-score) and handle empty/constant data
    df_std = df.std().replace(0, 1)
    z_scores = (df - df.mean()) / df_std
    return z_scores.fillna(0), top_feats


# --- 2. APP CONFIG & SIDEBAR ---
st.set_page_config(page_title="Stylo-Lab Professional", layout="wide")
st.title("🔬 Stylometry Lab: Lexical, Structural & Attribution")

with st.sidebar:
    st.header("Selection")
    data_source = st.radio(
        "Corpus", ["UNRESTRICTED-10", "TAGGED-10", "KNOWN-10", "Upload Files"])
    mfw_limit = st.slider("MFW Limit", 50, 2000, 500)
    use_stop = st.checkbox("Filter Stopwords", value=True)
    stop_input = st.text_area(
        "Stopwords", "the, and, of, to, a, in, is, it, that, was").lower()
    stop_list = [w.strip() for w in stop_input.split(",") if w.strip()]

    st.markdown("---")
    st.header("Network Settings")
    net_threshold = st.slider("Connection Sensitivity (Percentile)", 5, 95, 25)

# --- 3. DATA PROCESSING ---
raw_data = {}
if data_source == "UNRESTRICTED-10":
    raw_data = load_corpus("preloaded1")
elif data_source == "TAGGED-10":
    raw_data = load_corpus("preloaded2")
elif data_source == "KNOWN-10":
    raw_data = load_corpus("preloaded3")
else:
    uploaded = st.file_uploader("Upload .txt or .tsv files",
                                accept_multiple_files=True)
    for f in uploaded:
        content = f.read().decode("utf-8")
        if f.name.endswith('.tsv'):
            data = [
                line.split('\t') for line in content.strip().split('\n')
                if '\t' in line
            ]
            raw_data[f.name] = {
                'word': [r[0].lower() for r in data],
                'tag': [r[1] for r in data],
                'lemma': [r[2].lower() for r in data]
            }
        else:
            raw_data[f.name] = {
                'word': [w for w in content.lower().split() if w.isalpha()],
                'tag': [],
                'lemma': []
            }

# --- 4. ANALYTICS ENGINES ---
if len(raw_data) >= 2:
    z_word, feats_word = build_matrix(raw_data, 'word', mfw_limit,
                                      stop_list if use_stop else [])

    # --- SCENARIO 1: LEXICAL EXPLORER ---
    st.header("📦 Scenario 1: Lexical Explorer (Words)")
    t1, t2, t3, t4, t5 = st.tabs([
        "🌳 Dendrogram", "🗺️ PCA Map", "📈 Loadings", "🕸️ Network", "📊 CSV Data"
    ])

    with t1:
        st.subheader("Hierarchical Clustering")
        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(linkage(z_word, 'ward'),
                   labels=list(raw_data.keys()),
                   orientation='left',
                   ax=ax)
        st.pyplot(fig)

        # New suggestion logic
        st.info("💡 **Cluster Suggestion**")
        last = linkage(z_word, 'ward')[-10:, 2]
        acceleration = np.diff(last, 2)
        suggested_k = acceleration.argmax() + 2
        st.write(
            f"Based on variance, **{suggested_k} clusters** is likely optimal."
        )

    with t2:
        st.subheader("PCA Visualization")
        pca_coords = PCA(n_components=2).fit_transform(z_word)
        fig, ax = plt.subplots()
        ax.scatter(pca_coords[:, 0],
                   pca_coords[:, 1],
                   c='skyblue',
                   edgecolors='navy')
        for i, txt in enumerate(raw_data.keys()):
            ax.annotate(txt, (pca_coords[i, 0], pca_coords[i, 1]), size=8)
        st.pyplot(fig)

    with t3:
        pca_model = PCA(n_components=2).fit(z_word)
        loadings = pd.DataFrame(pca_model.components_.T,
                                index=feats_word,
                                columns=['PC1', 'PC2'])
        st.write("Top Words driving differences (PC1):")
        st.dataframe(loadings.sort_values('PC1', ascending=False).head(20))

#--------------nih
    with t4:
        st.subheader("Document Network")
        G = nx.Graph()
        dist_matrix = squareform(pdist(z_word, metric='cityblock'))
        # Use the slider value from the sidebar here:
        threshold = np.percentile(dist_matrix, net_threshold)
        for i, ni in enumerate(raw_data.keys()):
            #--------------------------- nih
            for j, nj in enumerate(raw_data.keys()):
                if i < j and dist_matrix[i, j] < threshold: G.add_edge(ni, nj)
        fig, ax = plt.subplots()
        nx.draw(G, with_labels=True, node_color='orange', ax=ax, font_size=8)
        st.pyplot(fig)

    with t5:
        st.download_button("Download Z-Scores (CSV)", z_word.to_csv(),
                           "lexical_zscores.csv")

        search_query = st.text_input(
            "🔍 Search for a specific word in the Z-scores:")
        if search_query:
            # Filters columns that contain the search string
            filtered_df = z_word[z_word.columns[z_word.columns.str.contains(
                search_query.lower())]]
            st.dataframe(filtered_df)
        else:
            st.dataframe(z_word)

    # --- SCENARIO 2: STRUCTURAL EXPLORER (Only if Tags Exist) ---
    if any(raw_data[next(iter(raw_data))]['tag']):
        st.divider()
        st.header("🧬 Scenario 2: Structural Explorer (Tags & Lemmas)")

        z_tag, feats_tag = build_matrix(raw_data, 'tag', 100)
        z_lemma, feats_lemma = build_matrix(raw_data, 'lemma', mfw_limit)

        st2_t1, st2_t2, st2_t3, st2_t4 = st.tabs([
            "🌳 Dual Dendrograms", "🗺️ Grammar Map", "📈 Tag Loadings",
            "📊 Grammar Profile"
        ])
        with st2_t1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Lemma-based Clustering")
                f1, a1 = plt.subplots()
                dendrogram(linkage(z_lemma, 'ward'),
                           labels=list(raw_data.keys()),
                           orientation='left',
                           ax=a1)
                st.pyplot(f1)
            with col2:
                st.write("Tag-based (Grammar) Clustering")
                f2, a2 = plt.subplots()
                dendrogram(linkage(z_tag, 'ward'),
                           labels=list(raw_data.keys()),
                           orientation='left',
                           ax=a2)
                st.pyplot(f2)

        with st2_t2:
            pca_tag = PCA(n_components=2).fit_transform(z_tag)
            fig, ax = plt.subplots()
            ax.scatter(pca_tag[:, 0], pca_tag[:, 1], c='green')
            for i, txt in enumerate(raw_data.keys()):
                ax.annotate(txt, (pca_tag[i, 0], pca_tag[i, 1]))
            st.pyplot(fig)

        with st2_t3:
            tag_loads = pd.DataFrame(
                PCA(n_components=2).fit(z_tag).components_.T,
                index=feats_tag,
                columns=['PC1', 'PC2'])
            st.dataframe(tag_loads.sort_values('PC1', ascending=False))

        with st2_t4:
            st.subheader("POS Ratios")
            profile_data = []
            for name, d in raw_data.items():
                ratios = pd.Series(
                    d['tag']).value_counts(normalize=True).head(10)
                profile_data.append(ratios)
            profile_df = pd.DataFrame(profile_data,
                                      index=raw_data.keys()).fillna(0)
            st.bar_chart(profile_df)

# --- SCENARIO 3: ATTRIBUTION ---
    # We allow this for KNOWN-10 OR Uploaded files so long as K/Q naming exists
    if data_source == "KNOWN-10" or data_source == "Upload Files":
        st.divider()
        st.header("🔍 Scenario 3: Lexical Attribution")

        k_idx = [i for i in z_word.index if i.startswith('K-')]
        q_idx = [i for i in z_word.index if i.startswith('Q-')]

        with st.expander("📂 View Loaded Data Inventory"):
            st.write(f"Known Files: {len(k_idx)} found.")
            st.write(f"Questioned Files: {len(q_idx)} found.")

        if len(k_idx) >= 2 and len(q_idx) >= 1:
            # 1. Global Calculations for all tabs
            labels = [n.split('-')[1] if '-' in n else "Unknown" for n in k_idx]
            pca_mod = PCA(n_components=2).fit(z_word)
            coords = pca_mod.transform(z_word)

            at1, at2, at3, at4 = st.tabs([
                "🗺️ Attribution Zones", "🎯 Accuracy/Confusion", "🏆 Delta Rank", "🔑 Known Markers"
            ])

            with at1:
                st.subheader("Authorship Zones (SVM)")
                fig, ax = plt.subplots(figsize=(10, 7))
                if len(set(labels)) > 1:
                    try:
                        svc = SVC(kernel='linear').fit(coords[:len(k_idx)], labels)
                        x_min, x_max = coords[:, 0].min() - 1, coords[:, 0].max() + 1
                        y_min, y_max = coords[:, 1].min() - 1, coords[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
                        Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
                        label_map = {name: i for i, name in enumerate(sorted(list(set(labels))))}
                        Z_num = np.array([label_map[z] for z in Z]).reshape(xx.shape)
                        ax.contourf(xx, yy, Z_num, alpha=0.15, cmap='coolwarm')
                    except: pass
                
                ax.scatter(coords[:len(k_idx), 0], coords[:len(k_idx), 1], c='blue', label='Known')
                ax.scatter(coords[len(k_idx):, 0], coords[len(k_idx):, 1], c='red', marker='X', s=100, label='Questioned')
                for i, txt in enumerate(z_word.index):
                    ax.annotate(txt, (coords[i, 0], coords[i, 1]), size=8)
                ax.legend()
                st.pyplot(fig)

            with at2:
                dist_mat = cdist(z_word.loc[q_idx], z_word.loc[k_idx], metric='cityblock')
                st.write("### Distance Matrix (Manhattan Distance)")
                st.dataframe(pd.DataFrame(dist_mat, index=q_idx, columns=k_idx).style.background_gradient(cmap='RdYlGn_r'))

            with at3:
                results = []
                dist_mat = cdist(z_word.loc[q_idx], z_word.loc[k_idx], metric='cityblock')
                k_internal_dists = pdist(z_word.loc[k_idx], metric='cityblock')
                outlier_threshold = np.mean(k_internal_dists) + np.std(k_internal_dists)

                for i, q in enumerate(q_idx):
                    dists = dist_mat[i]
                    min_dist = np.min(dists)
                    sorted_indices = np.argsort(dists)
                    match_idx, runner_up_idx = sorted_indices[0], sorted_indices[1]
                    confidence = (dists[runner_up_idx] - dists[match_idx]) / dists[runner_up_idx]
                    is_outlier = min_dist > outlier_threshold
                    results.append({
                        "Questioned": q, "Top Match": k_idx[match_idx], 
                        "Confidence": f"{confidence:.2%}",
                        "Status": "Outlier" if is_outlier else "Closely Similar",
                        "Dist_Val": min_dist
                    })

                def color_status(val):
                    color = 'red' if val == 'Outlier' else 'green'
                    return f'color: {color}; font-weight: bold'

                df_display = pd.DataFrame(results).drop(columns=['Dist_Val'])
                st.subheader("🏆 Attribution & Delta Rank")
                st.dataframe(df_display.style.map(color_status, subset=['Status']), use_container_width=True)
                st.download_button("📥 Download Results (CSV)", df_display.to_csv(index=False), "attribution.csv")

                st.info("### 📝 Automated Authorship Narration")
                similar_texts = [r['Questioned'] for r in results if r['Status'] == "Closely Similar"]
                outlier_texts = [r['Questioned'] for r in results if r['Status'] == "Outlier"]
                
                if similar_texts:
                    st.markdown("**✅ Possibility One (High Similarity):**")
                    st.write(f"The texts **{', '.join(similar_texts)}** fall within expected variance.")
                if outlier_texts:
                    st.markdown("**⚠️ Possibility Two (Stylistic Outliers):**")
                    st.write(f"The texts **{', '.join(outlier_texts)}** are statistically distant.")
                
                st.divider()
                st.subheader("🏁 Final Verdict Summary")
                cv1, cv2, cv3 = st.columns(3)
                cv1.metric("Total", len(results))
                cv2.metric("Attributed", len(similar_texts))
                cv3.metric("Outliers", len(outlier_texts))

            with at4:
                st.subheader("🔑 Key Stylistic Markers of Known Texts")
                known_mean_coord = np.mean(coords[:len(k_idx), 0])
                direction = 1 if known_mean_coord > 0 else -1
                weights = pca_mod.components_[0] * direction
                marker_df = pd.DataFrame({'Word': list(feats_word), 'Weight': weights}).sort_values('Weight', ascending=False)
                top_markers = marker_df[marker_df['Weight'] > 0].head(15)
                
                if not top_markers.empty:
                    col_m1, col_m2 = st.columns([1, 2])
                    with col_m1: st.dataframe(top_markers, hide_index=True)
                    with col_m2: st.bar_chart(top_markers.set_index('Word'))

        else:
            st.warning("Insufficient data. Ensure filenames start with 'K-' and 'Q-'.")

else:
    st.info("Please load or upload at least 2 files to generate analytics.")
