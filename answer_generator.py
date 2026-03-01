from openai import OpenAI
from prompt import (
    FORMATION_SYSTEM_PROMPT,
    FORMATION_USER_PROMPT_TEMPLATE,
    GENERAL_SYSTEM_PROMPT,
    GENERAL_USER_PROMPT_TEMPLATE
)
from proxy_config import API_KEY, BASE_URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def generate_full_formation_answer_v2(
    mineral,
    paths,
    genesis_triples,
    geo_context_str,
    top_texts=None,
    extra_1hop_triples=None,
    question: str = ""
):
    """
    Generate a detailed mineral formation reasoning answer using Knowledge Graph paths,
    geological context, and 1-hop supplementary knowledge.

    Args:
        mineral (str): Target mineral name.
        paths (list[dict]): Multi-hop Knowledge Graph paths with triples, sources, and paragraphs.
        genesis_triples (list[dict]): Additional genesis-related triples.
        geo_context_str (str): Regional geological background text.
        top_texts (list[str], optional): Retrieved literature paragraphs.
        extra_1hop_triples (list[dict], optional): Supplementary 1-hop triples and text.
        question (str): The user’s question.
    Returns:
        dict: {"answer": str, "prompt": str}
    """
    path_lines = []
    citation_lines = []
    triple_to_cite = {}
    citation_counter = 1
    for i, path in enumerate(paths):
        path_lines.append(f"[Path {i + 1}]")
        for j, triple in enumerate(path["triples"]):
            head, rel, tail = triple
            citation_id = f"[# {citation_counter}]"
            source = path["sources"][j]
            para = path["paragraphs"][j]

            if isinstance(para, dict):
                head_para = para.get("head", "")
                tail_para = para.get("tail", "")
            elif isinstance(para, str):
                head_para = para
                tail_para = ""
            else:
                head_para = tail_para = ""
            path_lines.append(f"- Triple {j + 1}: {triple} {citation_id}")
            if head_para:
                path_lines.append(f"  ↳ {head} paragraph: {head_para}")
            if tail_para:
                path_lines.append(f"  ↳ {tail} paragraph: {tail_para}")
            citation_lines.append(f"{citation_id} Triple: {triple} | Source: {source}")
            triple_to_cite[str(triple)] = citation_id
            citation_counter += 1

    # Append additional genesis-related triples
    for item in genesis_triples:
        triple = item["triple"]
        triple_str = f"{triple}"
        source = item["source"]
        if triple_str not in triple_to_cite:
            citation_id = f"[# {citation_counter}]"
            path_lines.append(f"- {triple_str} {citation_id}")
            citation_lines.append(f"{citation_id} Triple: {triple_str} | Source: {source}")
            triple_to_cite[triple_str] = citation_id
            citation_counter += 1

    # Include supplementary 1-hop triples and their textual descriptions for context
    extra_1hop_lines = []
    if extra_1hop_triples:
        for item in extra_1hop_triples:
            triple = item["triple"]
            triple_str = str(triple)
            citation_id = f"[# {citation_counter}]"
            source = item["source"]
            para = item.get("paragraph", "")
            desc = item.get("description", "")

            extra_1hop_lines.append(f"- Triple: {triple} {citation_id}")
            if para:
                extra_1hop_lines.append(f"  ↳ Paragraph: {para}")
            elif desc:
                extra_1hop_lines.append(f"  ↳ Description: {desc}")

            citation_lines.append(f"{citation_id} Triple: {triple_str} | Source: {source}")
            citation_counter += 1

    user_prompt = FORMATION_USER_PROMPT_TEMPLATE.format(
        question=question.strip(),
        path_lines=path_lines,
        extra_1hop_lines=extra_1hop_lines,
        geo_context=geo_context_str,
        top_texts=top_texts
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": FORMATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )

    return {
        "answer": response.choices[0].message.content.strip(),
        "prompt": user_prompt,
    }

def generate_general_answer_v2(question, paths, entity_context_str=None, top_texts=None):
    """
    Generate a general QA answer (non-genesis reasoning) using Knowledge Graph paths
    and optionally retrieved literature paragraphs.
    """
    path_lines = []
    for i, path in enumerate(paths):
        path_lines.append(f"[Path {i + 1}]")

        triples = path.get("triples", []) or []
        paras = path.get("paragraphs", []) or []

        same_len = isinstance(paras, list) and len(paras) == len(triples)

        for j, triple in enumerate(triples):
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                path_lines.append(f"- Triple {j + 1}: {triple}")
                if j < len(paras) and isinstance(paras[j], str) and paras[j].strip():
                    path_lines.append(f"  ↳ Paragraph: {paras[j].strip()}")
                continue

            head, rel, tail = triple
            path_lines.append(f"- Triple {j + 1}: {triple}")
            if "sources" in path and j < len(path["sources"]):
                src = path["sources"][j]
                if isinstance(src, str) and src.strip():
                    path_lines.append(f"  ↳ Source: {src}")

            # Align paragraphs with entities only if the number of paragraphs matches the number of triples
            if same_len:
                p = paras[j]
                if isinstance(p, dict):
                    head_para = (p.get("head") or "").strip()
                    tail_para = (p.get("tail") or "").strip()
                    if head_para:
                        path_lines.append(f"  ↳ {head} paragraph: {head_para}")
                    if tail_para and tail_para != head_para:
                        path_lines.append(f"  ↳ {tail} paragraph: {tail_para}")
                elif isinstance(p, str) and p.strip():
                    path_lines.append(f"  ↳ Paragraph: {p.strip()}")
            else:
                # If counts differ, attach a generic paragraph only when text is available
                if j < len(paras):
                    p = paras[j]
                    if isinstance(p, str) and p.strip():
                        path_lines.append(f"  ↳ Paragraph: {p.strip()}")
                    elif isinstance(p, dict):
                        hp = (p.get("head") or "").strip()
                        tp = (p.get("tail") or "").strip()
                        if hp:
                            path_lines.append(f"  ↳ {head} 段落: {hp}")
                        if tp and tp != hp:
                            path_lines.append(f"  ↳ {tail} 段落: {tp}")

    user_prompt = GENERAL_USER_PROMPT_TEMPLATE.format(
        question=question,
        paths="\n".join(path_lines),
        texts="\n".join(top_texts) if top_texts else "No extra texts retrieved."
    )

    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": GENERAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.5
    )
    return response.choices[0].message.content.strip(), user_prompt

