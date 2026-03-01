import numpy as np

# Input two vectors and return their similarity (normalized dot product)
def sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return float(np.dot(vec1, vec2))

# Calculate the sum of similarities between the question vector and a set of vectors (such as segments or description vectors on a link).
def score_vector_list(query_vec: np.ndarray, vecs: list[np.ndarray]) -> float:
    return sum(sim(query_vec, v) for v in vecs)

# Calculate the vector score (description + paragraph) in the link.
def score_path(query_vec: np.ndarray, desc_embs: list[np.ndarray], para_embs: list[np.ndarray], alpha=1.0, beta=1.0) -> float:
    return alpha * score_vector_list(query_vec, desc_embs) + beta * score_vector_list(query_vec, para_embs)
