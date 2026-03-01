import warnings
import torch
from FlagEmbedding import FlagReranker
from embedding_utils import embed
from intent_classifier import classify_intent_and_extract_entities
from geo_context_summary import query_all_geological_info, format_question_with_context,summarize_geological_context
from graph_query import query_genesis_triples_for
from answer_generator import (
    generate_full_formation_answer_v2,
    generate_general_answer_v2
)
from retrieval_with_context_v2 import (
    retrieve_for_formation_analysis_v2,
    retrieve_for_general_question_v2
)
warnings.filterwarnings("ignore", category=FutureWarning)
# drives query_all_geological_info
features_for_query = ["epoch", "hirise", "crater", "valley", "mineral"]
# Controls which fields are for  summarized for embedding
INCLUDE_FOR_QUESTION_CONTEXT = ["hirise_top3"]
# Controls which geological data fields are summarized
INCLUDE_FOR_GEO_SUMMARY = ["epoch","hirise_all", "craters", "mineral_data"]

USE_RERANKER = False
BLOCKED_SOURCES = {

}
TEXT_WEIGHT = 0.6
DESC_WEIGHT = 0.4
local_model_path = ""  # Path to local reranker model
reranker = None
if USE_RERANKER:
    reranker = FlagReranker(local_model_path, use_fp16=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reranker.model.to(device)
    reranker.model.half()
    print(f"Reranker loaded successfully. Device: {device}")
else:
    print("Reranker is disabled.")

def run_MMAgent(question: str) -> str:
    """
        Run the MMAgent V2 pipeline for a given user question.
        Depending on the identified intent, this function executes:
        - Formation analysis with geological context
        - General or reasoning QA with path and text retrieval
        Returns the generated final answer.
    """
    print(f"\n📥 User question: {question}")
    info = classify_intent_and_extract_entities(question)
    intent = info["intent"]
    minerals = info.get("minerals", [])
    geo_entities = info.get("geo_entities", [])
    coords = info.get("coordinates", [])
    lat, lon = coords[0] if coords else (None, None)

    all_entities = minerals + geo_entities
    print(f"\n🧭 Detected intent: {intent}")
    print(f"🔍 Mineral entities: {minerals}")
    print(f"🏞️ Geological entities: {geo_entities}")
    print(f"📌 Coordinates: {(lat, lon)}")

    if intent == "formation_analysis" and all_entities and lat is not None:
        print("\n📊 Embedding question and geological context...")
        instr = "Generate a representation for this sentence to use to retrieve related articles:"
        q_vec = embed([instr + question], tag="general qa")
        geo_context = query_all_geological_info(lat, lon, features=features_for_query)
        geo_context_str = format_question_with_context(
            question,
            geo_context,
            include=INCLUDE_FOR_QUESTION_CONTEXT
        )
        geo_vec = embed(geo_context_str, tag="geo background")
        print("\n🚀 Start performing multi-entity causal link retrieval...")
        all_paths = []
        all_genesis_triples = []
        all_top_texts = []
        all_extra_1hop = []
        for entity in all_entities:
            print(f"\n🌐 entity retrieval：{entity}")
            paths, top_texts, extra_1hop = retrieve_for_formation_analysis_v2(
                question=question,
                entity=entity,
                lat=lat,
                lon=lon,
                q_vec=q_vec,
                geo_vec=geo_vec,
                blocked_sources=BLOCKED_SOURCES,
                text_weight=TEXT_WEIGHT,
                desc_weight=DESC_WEIGHT,
                reranker=reranker
            )
            genesis = query_genesis_triples_for(entity)

            all_paths.extend(paths)
            all_genesis_triples.extend(genesis)
            all_top_texts.extend(top_texts)
            all_extra_1hop.extend(extra_1hop)

        geo_summary = summarize_geological_context(
            **geo_context,
            include=INCLUDE_FOR_GEO_SUMMARY
        )
        result = generate_full_formation_answer_v2(
            mineral=", ".join(minerals),
            paths=all_paths,
            genesis_triples=all_genesis_triples,
            geo_context_str=geo_summary,
            top_texts=all_top_texts,
            extra_1hop_triples=all_extra_1hop,
            question=question
        )
        print("\n📤 Prompt (formation analysis):\n", result["prompt"])
        print("\n🧠 Generated answer:\n", result["answer"])
        return result["answer"]
    elif intent in ["reasoning_qa", "general_qa"] and all_entities:
        print("\n📊 Embedding question and entity descriptions...")
        all_desc_text = "; ".join(all_entities)
        desc_vec = embed(all_desc_text, tag="entity_description")
        instr = "Generate a representation for this sentence to use to retrieve related articles:"
        q_vec = embed([instr + question], tag="general_question")

        print("\n🚀 Retrieving multi-entity general QA information...")
        all_paths, all_contexts, all_top_texts = retrieve_for_general_question_v2(
            question=question,
            entities=all_entities,
            q_vec=q_vec,
            desc_vec=desc_vec,
            blocked_sources=BLOCKED_SOURCES,
            text_weight=TEXT_WEIGHT,
            desc_weight=DESC_WEIGHT,
            reranker=reranker
        )
        answer, prompt = generate_general_answer_v2(
            question=question,
            paths=all_paths,
            entity_context_str=all_contexts,
            top_texts=all_top_texts
        )
        print("\n📤 Prompt (general QA):\n", prompt)
        print("\n🧠 Generated answer:\n", answer)
        return answer
    else:
        print("⚠️ No valid intent or entities detected. Skipped.")
        return ""

if __name__ == "__main__":
    q1 = "Please introduce the characteristics of jarosite."
    q2 = " At 109.9°E, 25.1°N on Mars, sulfate was detected. What could be the formation mechanism?"
    answer = run_MMAgent(q2)
