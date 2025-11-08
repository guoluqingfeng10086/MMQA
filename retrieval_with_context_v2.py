import numpy as np
from typing import List, Tuple, Dict
from embedding_utils import embed
from text_retrival import get_top_texts_for_entity
from graph_query import query_direct_description
import warnings
from path_selector import select_final_3hop_paths,select_final_3hop_paths_with_extra_1hop,select_general_paths
warnings.filterwarnings("ignore", category=FutureWarning)

USE_GEO_CONTEXT = True
USE_TEXT_RETRIEVAL = True
BLOCKED_SOURCES = {

}

def retrieve_for_formation_analysis_v2(
    question: str,
    entity: str,
    lat: float,
    lon: float,
    q_vec: np.ndarray,
    geo_vec: np.ndarray,
    topk: int = 8,
    reranker=None,
    blocked_sources=None,
    text_weight: float = 0.6,
    desc_weight: float = 0.4
) -> Tuple[List[Dict], List[str], List[Dict]]:
    """
    For the formation_analysis task, starting from a single entity:
    Retrieve its 3-hop knowledge graph paths (including triples and paragraphs)
    Retrieve its related paragraphs as textual evidence
    Returns: (paths, paragraphs)
    """
    print(f"\n🌋 Formation retrieval: entity={entity}")
    q_mix = text_weight * q_vec + desc_weight * geo_vec
    paths, extra_1hop = select_final_3hop_paths_with_extra_1hop(entity, q_mix, topk=topk)
    top_texts = get_top_texts_for_entity(
        entity,
        query_vec=q_mix,
        blocked_sources=blocked_sources or BLOCKED_SOURCES,
        reranker=reranker,
        topk=3
    )
    return paths, top_texts, extra_1hop

def retrieve_for_general_question_v2(
    question: str,
    entities: List[str],
    q_vec: np.ndarray,
    desc_vec: np.ndarray,
    topk_path: int = 6, # Number of paths to retrieve can be adjusted
    reranker=None,
    blocked_sources=None,
    text_weight: float = 0.6,
    desc_weight: float = 0.4
) -> Tuple[List[Dict], str, List[str]]:
    """
    For `general_qa` or `reasoning_qa` tasks:
    Perform 2-hop path retrieval starting from all entities
    Concatenate entity descriptions as context
    Retrieve related paragraphs as supplementary information
    Returns:(paths, concatenated_descriptions, paragraphs)`
    """
    print(f"\n🔍 General QA retrieval: entities={entities}")
    paths = select_general_paths(question, entities, topk2=topk_path)
    print("📘 Concatenating entity description information...")
    descriptions = [query_direct_description(e) for e in entities if query_direct_description(e)]
    context_text = "\n".join(descriptions)
    desc_vec = embed(context_text)
    print("📑 Retrieving related paragraphs (weighted question + description vectors)...")
    instr = "Generate a representation for this sentence to use to retrieve related articles:"
    print(instr + question + context_text)
    q_mix = embed([instr + question + "\nkey entity descriptions:" + context_text], tag="General QA")
    q_mix = text_weight * q_mix + desc_weight * desc_vec
    top_texts = get_top_texts_for_entity(
        entity_name="",
        query_vec=q_mix,
        blocked_sources=blocked_sources or BLOCKED_SOURCES,
        reranker=reranker,
        topk=3
    )
    return paths, context_text, top_texts


