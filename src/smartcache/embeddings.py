"""
Embedding Utilities for SmartCache ML.
Loads sentence embedding model and computes normalized vectors and cosine similarity using NumPy.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from smartcache.config import settings

_MODEL_INSTANCE: SentenceTransformer = None


def get_embedding_model(model_name: str = settings.embedding_model_name) -> SentenceTransformer:
    """Lazy singleton loader for the sentence embedding model."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = SentenceTransformer(model_name)
    return _MODEL_INSTANCE


def encode_query(
    text: Union[str, List[str]],
    model_name: str = settings.embedding_model_name,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate dense vector representation(s) for query string(s).
    
    Args:
        text: Single string or list of query strings
        model_name: HuggingFace model identifier
        normalize: If True, vectors are L2-normalized so dot-product equals cosine similarity.
        
    Returns:
        np.ndarray: Shape (d,) for single string or (N, d) for list of strings (float32).
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False
    )
    return embeddings.astype(np.float32)


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two 1D vectors:
    cos_sim(A, B) = (A . B) / (||A|| * ||B||)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_matrix_cosine_similarities(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarities between a normalized query vector (d,) and a matrix of doc vectors (N, d).
    Since both are L2 normalized, this is a fast matrix-vector product.
    
    Returns:
        np.ndarray of shape (N,) containing similarity scores in range [-1.0, 1.0].
    """
    if doc_matrix.ndim == 1:
        doc_matrix = doc_matrix.reshape(1, -1)
    
    # Compute dot product
    scores = np.dot(doc_matrix, query_vec)
    return scores
