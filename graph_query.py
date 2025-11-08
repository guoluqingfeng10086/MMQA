from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Any, Tuple
from functools import lru_cache
# Neo4j settings
_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("username", "password"))
# Enter your neo4j username and password

@lru_cache(maxsize=128)
def query_direct_description(entity_name: str) -> str:
    """
    Query the description field of a specific entity
    """
    query = """
    MATCH (n)
    WHERE toLower(n.name) = toLower($name)
    RETURN n.description AS description
    LIMIT 1
    """
    with _driver.session() as session:
        result = session.run(query, name=entity_name).single()
        return result["description"] if result and result["description"] else ""

def query_direct_neighbors(entity_name: str) -> List[Dict[str, Any]]:
    """
    Returns the 1-hop neighbors connected to the specified entity, along with the corresponding triples and source.
    """
    query = """
    MATCH (n)-[r]-(m)
    WHERE toLower(n.name) = toLower($name)
      AND m.gnn_embedding_v1 IS NOT NULL AND m.paragraph_embedding_v1 IS NOT NULL
    RETURN m.name AS name,
           type(r) AS rel_type,
           r.source AS source,
           m.gnn_embedding_v1 AS desc_emb,
           m.paragraph_embedding_v1 AS para_emb,
           m.description AS description,
           m.paragraph AS paragraph
    """
    with _driver.session() as session:
        records = session.run(query, name=entity_name)
        neighbors = []
        for r in records:
            neighbors.append({
                "name": r["name"],
                "desc_emb": np.array(r["desc_emb"], dtype=np.float32),
                "para_emb": np.array(r["para_emb"], dtype=np.float32),
                "description": r.get("description", ""),
                "paragraph": r.get("paragraph", ""),
                "triple": (entity_name, r["rel_type"], r["name"]),
                "source": r.get("source", "unknown")  # ✅ 加入三元组的关系来源
            })
        return neighbors


def query_direct_genesis_neighbors(entity_name: str) -> List[Dict[str, Any]]:
    """
    Returns a 1-hop neighborhood of type genesis.
    """
    query = """
       MATCH (n) WHERE toLower(n.name) = toLower($name)
       MATCH (n)--(m:genesis)
       WHERE m.gnn_embedding_v1 IS NOT NULL AND m.paragraph_embedding_v1 IS NOT NULL
       RETURN m.name AS name,
              m.gnn_embedding_v1 AS desc_emb,
              m.paragraph_embedding_v1 AS para_emb,
              m.description AS description,
              m.paragraph AS paragraph
    """
    with _driver.session() as session:
        records = session.run(query, name=entity_name)
        neighbors = []
        for r in records:
            neighbors.append({
                "name": r["name"],
                "desc_emb": np.array(r["desc_emb"], dtype=np.float32),
                "para_emb": np.array(r["para_emb"], dtype=np.float32),
                "description": r.get("description", ""),
                "paragraph": r.get("paragraph", "")
            })
        return neighbors

@lru_cache(maxsize=1024)
def query_khop_paths(start: str, k: int) -> List[Dict[str, Any]]:
    """
    Expand the k-hop paths of a given entity, including all entity information, relation types, and source.
    Return: path (list of names), desc/para embedding lists, triples, source, description, paragraph.
    """
    query = f"""
    MATCH (start) WHERE toLower(start.name) = toLower($name)
    MATCH p=(start)-[*1..{k}]-(end)
    WHERE ALL(n IN nodes(p) WHERE n.gnn_embedding_v1 IS NOT NULL AND n.paragraph_embedding_v1 IS NOT NULL)
      AND size(nodes(p)) = {k + 1}
      AND ALL(i IN range(0, size(nodes(p))-2) WHERE toLower(nodes(p)[i].name) <> toLower(nodes(p)[i+1].name))
    WITH p, [n IN nodes(p) | toLower(n.name)] AS name_lowers, nodes(p) AS nds
    WHERE size(name_lowers) = size(apoc.coll.toSet(name_lowers))  
    RETURN nds AS path_nodes, relationships(p) AS rels
    """
    results = []
    with _driver.session() as session:
        records = session.run(query, name=start)
        for record in records:
            try:
                nodes = record["path_nodes"]
                rels = record["rels"]
                path, desc_embs, para_embs = [], [], []
                descriptions, paragraphs = [], []
                triples, sources = [], []

                for node in nodes:
                    path.append(node["name"])
                    desc_embs.append(np.array(node["gnn_embedding_v1"], dtype=np.float32))
                    para_embs.append(np.array(node["paragraph_embedding_v1"], dtype=np.float32))
                    descriptions.append(node.get("description", ""))
                    paragraphs.append(node.get("paragraph", ""))

                if len(set(name.lower() for name in path)) < len(path):
                    print(f"[!!!Skip closed loop path]: {path}")
                    continue

                for rel in rels:
                    triples.append((rel.start_node["name"], rel.type, rel.end_node["name"]))
                    sources.append(rel.get("source", "unknown"))

                seen_set = set()
                triples_dedup, sources_dedup = [], []
                for t, s in zip(triples, sources):
                    key = (t[0].lower(), t[1], t[2].lower())
                    if key not in seen_set:
                        seen_set.add(key)
                        triples_dedup.append(t)
                        sources_dedup.append(s)

                triples = triples_dedup
                sources = sources_dedup
                results.append({
                    "path": path,
                    "desc_embs": desc_embs,
                    "para_embs": para_embs,
                    "descriptions": descriptions,
                    "paragraphs": paragraphs,
                    "triples": triples,
                    "sources": sources
                })
            except Exception as e:
                print("[!]Path resolution failed:", e)
                continue
    return results



def query_relation_between(entity1: str, entity2: str) -> Dict[str, str]:
    """
    Query the direct relationship type and source between two entities
    """
    query = """
    MATCH (a)-[r]-(b)
    WHERE toLower(a.name) = toLower($e1) AND toLower(b.name) = toLower($e2)
    RETURN type(r) AS rel_type, r.source AS source
    LIMIT 1
    """
    with _driver.session() as session:
        result = session.run(query, e1=entity1, e2=entity2).single()
        if result:
            return {
                "rel_type": result["rel_type"],
                "source": result.get("source", "unknown")
            }
        else:
            return {
                "rel_type": "related_to",
                "source": "unknown"
            }


def query_genesis_triples_for(mineral: str):
    """
    Retrieve the genetic mechanism ternary sequence and origin associated with a specific mineral.
    Return format:
    [
        {"triple": ("sulfate", "related_to", "Aqueous Process"), "source": "...", "paragraph": "..."},
        ...
    ]
    """
    with _driver.session() as session:
        query = """
        MATCH (m)-[r]->(g:genesis)
        WHERE m.name = $mineral
        RETURN m.name AS head, type(r) AS rel, g.name AS tail, r.source AS source, r.paragraph AS paragraph
        """
        result = session.run(query, mineral=mineral)
        return [
            {
                "triple": (record["head"], record["rel"], record["tail"]),
                "source": record["source"],
                "paragraph": record["paragraph"]
            }
            for record in result
        ]

# Queries that prioritize deep links
def query_node_labels_and_neighbors(entity_name: str) -> Tuple[List[str], int]:
    """
    Query the label and number of neighbors of a given entity
    """
    query = """
    MATCH (n)-[]-(m)
    WHERE toLower(n.name) = toLower($name)
    RETURN labels(n) AS labels, count(DISTINCT m) AS neighbor_count
    LIMIT 1
    """
    with _driver.session() as session:
        result = session.run(query, name=entity_name).single()
        if result:
            return result["labels"], result["neighbor_count"]
        else:
            return [], 0


def get_topic_entities(mineral: str, max_hop: int = 3):
    return [mineral]


def query_one_hop_edges_undirected(entity: str, blocked_sources: List[str] = []) -> List[Tuple[str, str, str, str]]:
    """
    Retrieve the 1-hop adjacency edges of a given entity, treating it as an undirected graph.
    Return format: (entity, relation, neighbor, source)
    """
    results = []
    with _driver.session() as session:
        cypher = """
        MATCH (a)-[r]-(b)
        WHERE toLower(a.name) = toLower($entity)
        RETURN a.name AS a_name, type(r) AS rel, b.name AS b_name, r.source AS source
        """
        records = session.run(cypher, entity=entity)
        for record in records:
            source = record["source"] or ""
            if source in blocked_sources:
                continue
            a = record["a_name"]
            b = record["b_name"]
            rel = record["rel"]
            if a.lower() == entity.lower():
                results.append((a, rel, b, source))  # entity 是 a
            else:
                results.append((b, rel, a, source))  # entity 是 b，统一返回格式
    return results

def query_one_hop_edges_with_raw_direction(entity: str, blocked_sources: List[str] = []) -> List[Tuple[str, str, str, str]]:
    """
    Return the one-hop adjacent edge of the entity, preserving the true direction. (head, relation, tail, source)
    """
    results = []
    with _driver.session() as session:
        cypher = """
        MATCH (a)-[r]-(b)
        WHERE toLower(a.name) = toLower($entity) OR toLower(b.name) = toLower($entity)
        RETURN a.name AS head, type(r) AS rel, b.name AS tail, r.source AS source
        """
        records = session.run(cypher, entity=entity)

        for record in records:
            h, r, t, s = record["head"], record["rel"], record["tail"], record["source"] or ""
            if s in blocked_sources:
                continue
            results.append((h, r, t, s))
    return results
