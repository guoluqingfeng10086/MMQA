from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from typing import Union
_model = None
# your embedding model path
_model_path = r""

def get_model():
    global _model
    if _model is None:
        print("📦 Lazy loading of embedded model bge-large...")
        _model = SentenceTransformer(_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
        print(f"The model has been loaded onto the device.: {device}")
    return _model

def combine_embeddings(q_vec, geo_vec, method='weighted_sum', weight=0.5):
    """
    Combine question and geological context embeddings.

    method:
        - 'weighted_sum': (1-w)*q + w*g
        - 'concat': concatenation (requires model support)
    """
    if method == 'weighted_sum':
        return (1 - weight) * q_vec + weight * geo_vec
    elif method == 'concat':
        return np.concatenate([q_vec, geo_vec])
    else:
        raise ValueError("Unsupported combination method.")

def embed(text: str, tag: str = "") -> np.ndarray:
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True)
    if tag:
        print(f"✅Embedding complete [{tag}]，Vector Dimension: {vec.shape}")
    return vec

def sim(a: np.ndarray, b: Union[np.ndarray, list[np.ndarray]]) -> float:
    """
    Calculate the total similarity for compatible vectors or lists of vectors.
    Vectors should be normalized unit vectors.
    """
    if a is None or b is None:
        print("The input for sim is None.")
        return 0.0

    try:
        if isinstance(b, list):
            return sum(np.dot(a, v) for v in b if v is not None)
        else:
            return float(np.dot(a, b))
    except Exception as e:
        print(f"❌ calculation error: {e}")
        return 0.0
