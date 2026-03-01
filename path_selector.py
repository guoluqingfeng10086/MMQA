import numpy as np
from graph_query import query_direct_genesis_neighbors, query_direct_neighbors, query_khop_paths
from link_scorer import score_path,sim
from embedding_utils import embed
from graph_query import query_relation_between,query_direct_description
from graph_query import query_direct_neighbors, query_node_labels_and_neighbors
from typing import List, Dict, Tuple, Optional
BLOCKED_SOURCES = {

}
def select_top1hop_genesis(mineral: str, query_vec: np.ndarray, topk: int = 5) -> List[Dict]:
    neighbors = query_direct_neighbors(mineral)
    scored = []
    print(f"\n🔍 [1-hop] Scoring entities connected to“{mineral}”, Prioritize genesis entities")
    candidates = []
    for n in neighbors:
        node_name = n["name"]
        labels, neighbor_count = query_node_labels_and_neighbors(node_name)

        if "genesis" in [l.lower() for l in labels] and neighbor_count > 1:
            n["is_genesis"] = True
        else:
            n["is_genesis"] = False

        score = score_path(query_vec, [n["desc_emb"]], [n["para_emb"]])
        n["score"] = score
        candidates.append(n)

    # Prioritize genesis entities(connections > 1), then add other high-scoring genes.
    primary = [c for c in candidates if c["is_genesis"]]
    secondary = [c for c in candidates if not c["is_genesis"]]

    # Sort and regroup by score
    primary = sorted(primary, key=lambda x: x["score"], reverse=True)
    secondary = sorted(secondary, key=lambda x: x["score"], reverse=True)
    final = (primary + secondary)[:topk]
    for c in final:
        print(f"  - Candidate entity: {c['name']} | genesis: {c['is_genesis']} | score: {c['score']:.4f}")
    return final


# Adjusting k can adjust the search depth,k=1-d=3,k=2-d=4
def expand_genesis_to_2hop(genesis_node: Dict, query_vec: np.ndarray, start_entity: str = None) -> List[Dict]:
    all_paths = query_khop_paths(genesis_node["name"], k=1)
    scored = []
    print(f"\n🔍 [2-hop] Expanding paths from the genesis entity \"{genesis_node['name']}\":")
    for path in all_paths:
        if start_entity and start_entity.lower() in [n.lower() for n in path["path"][1:]]:
            print(f"  ⚠️Jump back to the main entity, skip.: {path['path']}")
            continue

        lowered = [n.lower() for n in path["path"]]
        if len(set(lowered)) < len(lowered):
            print(f"  ⚠️ Entities that are repeated should be skipped.: {path['path']}")
            continue

        score = score_path(query_vec, path["desc_embs"], path["para_embs"])
        path["score"] = score
        print(f"  - Path: {path['path']} | score: {score:.4f}")
        scored.append(path)

    return sorted(scored, key=lambda x: x["score"], reverse=True)[:2]


def expand_2hop_to_3hop(path2: Dict, query_vec: np.ndarray, start_entity: str = None) -> Dict:
    tail = path2["path"][-1]
    candidates = query_direct_neighbors(tail)

    forbidden = set(n.lower() for n in path2["path"])
    if start_entity:
        forbidden.add(start_entity.lower())

    candidates = [c for c in candidates if c["name"].lower() not in forbidden]

    print(f"\n🔍 [3-hop] Expanding candidate entities from \"{tail}\":")
    for c in candidates:
        score = score_path(query_vec, [c["desc_emb"]], [c["para_emb"]])
        print(f"  - Candidate entity: {c['name']} | score: {score:.4f}")
        c["score"] = score

    if len(candidates) == 0:
        print("  ⚠️ No extensible entity, return to the original path")
        return path2

    best = max(candidates, key=lambda c: c["score"])
    print(f"✅ Select entity: {best['name']}")

    rel_info = query_relation_between(tail, best["name"])
    triple = (tail, rel_info["rel_type"], best["name"])
    source = rel_info["source"]

    new_path = {
        "path": path2["path"] + [best["name"]],
        "desc_embs": path2["desc_embs"] + [best["desc_emb"]],
        "para_embs": path2["para_embs"] + [best["para_emb"]],
        "score": path2["score"] + best["score"],
        "descriptions": path2.get("descriptions", []) + [best["description"]],
        "paragraphs": path2.get("paragraphs", []) + [best["paragraph"]],
        "triples": path2.get("triples", []) + [triple],
        "sources": path2.get("sources", []) + [source]
    }
    return new_path


def select_final_3hop_paths(mineral: str, query_vec: np.ndarray, topk: int = 3) -> List[Dict]:
    top1hop = select_top1hop_genesis(mineral, query_vec)
    all_2hop = []

    for g in top1hop:
        exps = expand_genesis_to_2hop(g, query_vec, start_entity=mineral)
        for path in exps:
            rel_info = query_relation_between(mineral, g["name"])
            triple = (mineral, rel_info["rel_type"], g["name"])
            source = rel_info["source"]

            path["path"] = [mineral, g["name"]] + path["path"][1:]
            path["desc_embs"] = [g["desc_emb"]] + path["desc_embs"]
            path["para_embs"] = [g["para_emb"]] + path["para_embs"]
            path["triples"] = [triple] + path.get("triples", [])
            path["sources"] = [source] + path.get("sources", [])
            path["descriptions"] = [g.get("description", "")] + path.get("descriptions", [])
            path["paragraphs"] = [g.get("paragraph", "")] + path.get("paragraphs", [])

            all_2hop.append(path)

    all_3hop = [expand_2hop_to_3hop(p, query_vec, start_entity=mineral) for p in all_2hop]
    all_3hop = dedup_paths_by_triples(all_3hop)

    final = sorted(all_3hop, key=lambda x: x["score"], reverse=True)[:topk]
    print("\n📌 [The final selected path Top-k]")
    for i, p in enumerate(final):
        print(f"\n[Path {i+1}] Score: {p['score']:.4f}")
        for j, t in enumerate(p["triples"]):
            print(f"  - Triple {j+1}: {t} [# {j+1}]")
    return final


def select_final_3hop_paths_with_extra_1hop(
    entity: str,
    query_vec: np.ndarray,
    topk: int = 8,
    extra_1hop_k: int = 15,
    blocked_sources: Optional[set] = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    Select the final 3-hop paths and include additional high-scoring 1-hop triples.
    Automatically excludes sources listed in blocked_sources.
    """
    blocked_sources = blocked_sources or BLOCKED_SOURCES

    # === Step 1: Retrieve 1-hop candidate nodes ===
    top1hop = select_top1hop_genesis(entity, query_vec, topk=80)  # 先多取一些备用

    expandable = []
    non_expandable = []
    expansion_results = {}

    for g in top1hop:
        exps = expand_genesis_to_2hop(g, query_vec, start_entity=entity)
        # Filter out paths from blocked_sources
        exps = [p for p in exps if all(s not in blocked_sources for s in p.get("sources", []))]
        expansion_results[g["name"].lower()] = exps
        if exps:
            expandable.append(g)
        else:
            non_expandable.append(g)
    # === Step 2: Select top 5 expandable nodes ===
    expandable = sorted(expandable, key=lambda x: x["score"], reverse=True)[:5]
    expandable_names = set(g["name"].lower() for g in expandable)

    # === Step 3: Construct 2-hop paths ===
    all_2hop = []
    for g in expandable:
        exps = expansion_results[g["name"].lower()]
        for path in exps:
            rel_info = query_relation_between(entity, g["name"])
            if rel_info["source"] in blocked_sources:
                print(f"⛔ Skipping triple from blocked source: {(entity, rel_info['rel_type'], g['name'])}")
                continue

            triple = (entity, rel_info["rel_type"], g["name"])
            source = rel_info["source"]

            path["path"] = [entity, g["name"]] + path["path"][1:]
            path["desc_embs"] = [g["desc_emb"]] + path["desc_embs"]
            path["para_embs"] = [g["para_emb"]] + path["para_embs"]
            path["triples"] = [triple] + path.get("triples", [])
            path["sources"] = [source] + path.get("sources", [])
            path["descriptions"] = [g.get("description", "")] + path.get("descriptions", [])
            path["paragraphs"] = [g.get("paragraph", "")] + path.get("paragraphs", [])

            all_2hop.append(path)

    # === Step 4: Expand to 3-hop paths and remove duplicates ===
    all_3hop = [expand_2hop_to_3hop(p, query_vec, start_entity=entity) for p in all_2hop]
    all_3hop = [p for p in all_3hop if all(s not in blocked_sources for s in p.get("sources", []))]
    all_3hop = dedup_paths_by_triples(all_3hop)

    final_paths = sorted(all_3hop, key=lambda x: x["score"], reverse=True)[:topk]

    print("\n📌 [Final Top-k Selected Paths]")
    for i, p in enumerate(final_paths):
        print(f"\n[Path {i+1}] Score: {p['score']:.4f}")
        for j, t in enumerate(p["triples"]):
            print(f"  - Triple {j+1}: {t} [# {j+1}]")
            print(f"    ↳ Source: {p['sources'][j]}")# Print data sources to trace the origin of knowledge

    # === Step 5: Additional high-scoring 1-hop triples ===
    remaining_candidates = [n for n in top1hop if n["name"].lower() not in expandable_names]
    remaining_candidates = sorted(
        remaining_candidates,
        key=lambda x: (not x.get("is_genesis", False), -x["score"])
    )[:50]

    extra_1hop = []
    for n in remaining_candidates[:extra_1hop_k]:
        rel_info = query_relation_between(entity, n["name"])
        if rel_info["source"] in blocked_sources:
            print(f"⛔ Skip 1-hop triples from blocked sources: {(entity, rel_info['rel_type'], n['name'])}")
            continue
        triple = (entity, rel_info["rel_type"], n["name"])
        extra_1hop.append({
            "triple": triple,
            "source": rel_info["source"],
            "description": n.get("description", ""),
            "paragraph": n.get("paragraph", ""),
            "score": n["score"]
        })

    return final_paths, extra_1hop

def select_general_paths(question: str, entities: list, topk2=6, topk1=6, max_check_expandable=45):
    print(f"\n🧪 question: {question}")
    q_vec = embed(question)

    all_1hop = []
    entity_scores = {}

    print("🔍 Phase 1: Retrieve all 1-hop paths and compute scores...")
    for ent in entities:
        desc = query_direct_description(ent)
        desc_vec = embed(desc) if desc else np.zeros_like(q_vec)
        q_mix = 0.6 * q_vec + 0.4 * desc_vec

        hop1 = query_khop_paths(ent, k=1)
        print(f"→ Entity {ent} is connected to {len(hop1)} entities")
        for p in hop1:
            tail_entity = p["path"][1]
            score = sum(sim(q_mix, v) for v in p["desc_embs"] + p["para_embs"])
            all_1hop.append({**p, "score": score})
            entity_scores[tail_entity] = score

    print("🔍 Phase 2: Check if the top high-scoring 1-hop paths are expandable...")
    sorted_1hop = sorted(all_1hop, key=lambda x: x["score"], reverse=True)
    extendable_1hop = []
    check_count = 0
    evaluated_entities = set()

    for p in sorted_1hop:
        if check_count >= max_check_expandable:
            break
        tail = p["path"][-1]
        evaluated_entities.add(tail)
        hop2 = query_khop_paths(tail, k=1)
        if hop2:
            extendable_1hop.append(p)
        check_count += 1

    print(f"📌 Evaluated entities (up to {max_check_expandable}):")
    print("   ", ", ".join(list(evaluated_entities)[:10]) + (" ..." if len(evaluated_entities) > 10 else ""))
    print(f"✅ Number of expandable paths: {len(extendable_1hop)}")

    print("🔍 Phase 3: Expand each expandable 1-hop path to 2-hop...")
    all_candidate_2hop = []
    for p in extendable_1hop:
        topic_entity_lower = p["path"][0].lower()
        tail = p["path"][-1]

        print(f"→ Expanding entity: {tail}...")
        count = 0
        hop2 = query_khop_paths(tail, k=1)
        for h in hop2:
            merged_path = p["path"] + h["path"][1:]
            merged_path_lower = [n.lower() for n in merged_path]

            if len(set(merged_path_lower)) < len(merged_path_lower):
                print("  ⚠️ Skip closed loop path", merged_path)
                continue
            if topic_entity_lower in merged_path_lower[1:]:
                print("  ⚠️ Skip back path", merged_path)
                continue

            score = sum(sim(q_vec, v) for v in h["desc_embs"] + h["para_embs"])
            merged_triples = p["triples"] + h["triples"]
            merged_sources = p["sources"] + h["sources"]
            triple_source_pairs = list(dict.fromkeys((t, s) for t, s in zip(merged_triples, merged_sources)))
            merged_triples = [ts[0] for ts in triple_source_pairs]
            merged_sources = [ts[1] for ts in triple_source_pairs]

            all_candidate_2hop.append({
                "path": merged_path,
                "desc_embs": p["desc_embs"] + h["desc_embs"],
                "para_embs": p["para_embs"] + h["para_embs"],
                "score": p["score"] + score,
                "triples": merged_triples,
                "sources": merged_sources,
                "descriptions": p["descriptions"] + h["descriptions"],
                "paragraphs": p["paragraphs"] + h["paragraphs"]
            })

            count += 1
            if count >= 2:
                break
    print("🔍 Phase 4: Select final topk2 paths from candidate 2-hop paths...")
    final_paths = sorted(all_candidate_2hop, key=lambda x: x["score"], reverse=True)[:topk2]
    print("📌 Selected 2-hop paths:")
    for i, p in enumerate(final_paths):
        print(f"  #{i+1}: {' -> '.join(p['path'])} (score={p['score']:.3f})")

    print("🔍 Phase 5: Select topk1 supplementary paths from remaining 1-hop paths...")
    final_used_1hop_entities = {p["path"][1].lower() for p in final_paths}
    remaining_1hop = [p for p in all_1hop if p["path"][1].lower() not in final_used_1hop_entities]
    remaining_1hop = sorted(remaining_1hop, key=lambda x: x["score"], reverse=True)
    extra_1hop = remaining_1hop[:topk1]
    print("📌 Selected additional 1-hop entities:", ", ".join([p["path"][1] for p in extra_1hop]))
    print(f"✅ Path selection completed: 2-hop = {len(final_paths)} paths, additional 1-hop = {len(extra_1hop)} paths")
    return final_paths + extra_1hop


def dedup_paths_by_triples(paths: List[Dict]) -> List[Dict]:
    """
    # Remove duplicate paths based on triples; order-sensitive, case-insensitive
    """
    seen = set()
    deduped = []

    for p in paths:
        triple_signature = tuple([
            f"{h.lower()}--{r.lower()}--{t.lower()}"
            for (h, r, t) in p["triples"]
        ])
        if triple_signature in seen:
            print(f"[⚠️Skipping duplicate-triple path]: {p['triples']}")
            continue
        seen.add(triple_signature)
        deduped.append(p)
    return deduped



