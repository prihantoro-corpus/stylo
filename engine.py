import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

def parse_treetagger(file, column_index):
    """Parses a TreeTagger file and returns the selected column as a list."""
    # TreeTagger is usually Tab-Separated
    df = pd.read_csv(file, sep='\t', names=['word', 'tag', 'lemma'], quoting=3)
    # 0=Word, 1=Tag, 2=Lemma
    return df.iloc[:, column_index].astype(str).tolist()

def build_frequency_matrix(corpus_dict, mfw_limit=500):
    """Converts a dictionary of {doc_name: [tokens]} into a frequency matrix."""
    # Flatten all tokens to find global MFW
    all_tokens = [token for tokens in corpus_dict.values() for token in tokens]
    top_features = pd.Series(all_tokens).value_counts().head(mfw_limit).index
    
    matrix = []
    for doc, tokens in corpus_dict.items():
        counts = pd.Series(tokens).value_counts()
        # Reindex ensures all docs have the same columns in the same order
        row = counts.reindex(top_features, fill_value=0)
        matrix.append(row)
        
    df_matrix = pd.DataFrame(matrix, index=corpus_dict.keys())
    # Z-score normalization (standard in Burrows Delta)
    df_zscore = (df_matrix - df_matrix.mean()) / df_matrix.std()
    return df_zscore

def calculate_delta(zscore_matrix):
    """Calculates Manhattan distance (Burrows Delta) between all documents."""
    distances = pdist(zscore_matrix, metric='cityblock') / zscore_matrix.shape[1]
    return pd.DataFrame(squareform(distances), 
                        index=zscore_matrix.index, 
                        columns=zscore_matrix.index)
