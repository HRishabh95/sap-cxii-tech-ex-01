import re
import numpy as np
import pandas as pd
import faiss
from gensim.models import Word2Vec
from scipy.sparse import spmatrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

# 384-dim, 6 layers — best CPU speed/quality tradeoff for semantic search
# swap to all-mpnet-base-v2 for higher quality, paraphrase-multilingual-MiniLM-L12-v2 for multilingual
_SBERT_MODEL = "all-MiniLM-L6-v2"

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 50


def _product_text(row: pd.Series) -> str:
    fields = ["product_name", "brand", "meta_keywords"]
    parts = [str(row[f]).strip() for f in fields if pd.notna(row.get(f)) and str(row.get(f, "")).strip()]
    return " ".join(parts)


def _tokenize(text: str) -> List[str]:
    return re.sub(r"[^a-z0-9\s]", "", str(text).lower()).split()


def _product_tokens(row: pd.Series) -> List[str]:
    tokens = []
    for f in ["product_name", "brand", "meta_keywords"]:
        val = row.get(f, "")
        if pd.notna(val):
            tokens.extend(_tokenize(str(val)))
    return tokens


def _hnsw_index(dim: int) -> faiss.Index:
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH
    return index


# --- TF-IDF ---

def build_tfidf_matrix(df: pd.DataFrame) -> Tuple[spmatrix, TfidfVectorizer]:
    # sublinear_tf: replaces tf with 1+log(tf) so a word 100x isn't 100x more important
    # ngram_range=(1,2): includes bigrams so "cotton shirt" is a feature, not just "cotton" and "shirt"
    texts = df.apply(_product_text, axis=1).tolist()
    vectorizer = TfidfVectorizer(max_features=10_000, sublinear_tf=True, ngram_range=(1, 2))
    return vectorizer.fit_transform(texts), vectorizer


def find_similar_by_tfidf(product_id: str, num_similar: int, df: pd.DataFrame, matrix: spmatrix) -> List[str]:
    hits = df.index[df["uniq_id"] == product_id]
    if len(hits) == 0:
        raise ValueError(f"product_id '{product_id}' not found")
    idx = int(hits[0])
    scores = cosine_similarity(matrix[idx], matrix).flatten()
    scores[idx] = -2.0
    return df.iloc[np.argsort(scores)[::-1][:num_similar]]["uniq_id"].tolist()


# --- Word2Vec ---

def build_word2vec_index(df: pd.DataFrame, vector_size: int = 100, epochs: int = 10):
    corpus = [t for t in df.apply(_product_tokens, axis=1).tolist() if t]
    model = Word2Vec(sentences=corpus, vector_size=vector_size, window=5, min_count=2, workers=4, epochs=epochs)

    def mean_vec(tokens):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0).astype("float32") if vecs else np.zeros(vector_size, dtype="float32")

    embeddings = np.vstack([mean_vec(_product_tokens(row)) for _, row in df.iterrows()])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)

    index = _hnsw_index(vector_size)
    index.add(embeddings.astype("float32"))
    return index, embeddings.astype("float32"), model


def find_similar_by_word2vec(product_id: str, num_similar: int, df: pd.DataFrame, index: faiss.Index, embeddings: np.ndarray) -> List[str]:
    hits = df.index[df["uniq_id"] == product_id]
    if len(hits) == 0:
        raise ValueError(f"product_id '{product_id}' not found")
    idx = int(hits[0])
    _, ids = index.search(embeddings[idx: idx + 1], num_similar + 1)
    return df.iloc[[i for i in ids[0] if i != idx][:num_similar]]["uniq_id"].tolist()


# --- Sentence-transformers (HNSW) ---
# Algorithm: Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2018)
# https://arxiv.org/abs/1603.09320

def build_text_index(df: pd.DataFrame):
    model = SentenceTransformer(_SBERT_MODEL)
    texts = df.apply(_product_text, axis=1).tolist()
    vecs = model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    vecs = vecs.astype("float32")
    index = _hnsw_index(vecs.shape[1])
    index.add(vecs)
    return index, vecs, model


def find_similar_by_text(product_id: str, num_similar: int, df: pd.DataFrame, index: faiss.Index, embeddings: np.ndarray) -> List[str]:
    hits = df.index[df["uniq_id"] == product_id]
    if len(hits) == 0:
        raise ValueError(f"product_id '{product_id}' not found")
    idx = int(hits[0])
    _, ids = index.search(embeddings[idx: idx + 1], num_similar + 1)
    return df.iloc[[i for i in ids[0] if i != idx][:num_similar]]["uniq_id"].tolist()
