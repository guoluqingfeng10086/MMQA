from typing import Optional, List, Tuple, Union
from sentence_transformers import SentenceTransformer
import torch
import lancedb
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# === Model and Database Path Configuration ===
MODEL_PATH = ""  # Embedded model path, we choose bge-large-en-1.5
LANCEDB_PATH = ""  # Vector database path

# === Load Model and Vector Database ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer(MODEL_PATH).to(device)
db = lancedb.connect(LANCEDB_PATH)
table = db.open_table("documents")

def get_top_texts_for_entity(
    entity_name: str,
    query_vec: Optional[np.ndarray] = None,
    blocked_sources: Optional[List[str]] = None,
    topk: int = 3,
    reranker=None
) -> List[str]:
    """
    Retrieve top-k paragraphs relevant to an entity.
    Supports blocked sources and optional reranking.
    """
    if query_vec is None:
        query_vec = model.encode([f"Find scientific paragraphs about: {entity_name}"], normalize_embeddings=True)[0]

    blocked_list = blocked_sources or []
    query = table.search(query_vec.tolist(), query_type="vector")
    if blocked_list:
        condition = " AND ".join([f"file != '{f}'" for f in blocked_list])
        query = query.where(condition)

    result = query.select(["text", "file", "page"]).limit(15).to_list()

    paragraphs = [r["text"].strip() for r in result if len(r["text"].strip()) > 50]

    if reranker is not None and paragraphs:
        pairs = [(entity_name, para) for para in paragraphs]
        scores = reranker.compute_score(pairs)
        paragraphs = [p for p, _ in sorted(zip(paragraphs, scores), key=lambda x: x[1], reverse=True)]
    return paragraphs[:topk]

