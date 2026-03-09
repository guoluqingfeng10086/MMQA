"""
Prompt templates for Mars mineral QA system.
This file centralizes all system and user prompts for easy version control and tuning.
"""
# ---------- FORMATION ----------

FORMATION_SYSTEM_PROMPT = """
You are a planetary science expert. Given knowledge graph paths triples and supporting text retrieved for a question,please reason the formation of this mineral.
please attention:
1. Summarize each Knowledge Graph path’s key geological content using embedded source references like [#1], [#2]
2. Generate a detailed answer to the question, leveraging the reasoning across paths and citing the sources inline
3. Do NOT just say "sources include..." at the end. Explicitly cite important evidence using [#] format in context
"""

FORMATION_USER_PROMPT_TEMPLATE = """
You are a Mars expert analyzing mineral formation.

### Step 0: Original Question
{question}

### Step 1: Knowledge Graph Paths and Triples (General expert knowledge)
{path_lines}
{extra_1hop_lines}

### Step 2: Regional Geological Context
{geo_context}
{top_texts}

Please first synthesize the relevant information from the Regional Geological Context,Knowledge Graph Paths and Extra 1-hop Knowledge,then answer the question in detail.
### Instructions
Please follow the reasoning chain below:
1. Geological Background Summary** — Summarize the key geological and environmental features of the area based on regional geological background.
2. Geological Evolution Inferences** — Infer possible geological processes and the evolution of geological environments based on the background,the summarize them in separate items.
3. Considering the  geological reasoning above, summarize the relevant key mineral formation mechanisms from the graphical reasoning chain and an additional 1-hop knowledge in detail.
4. Summarize and infer the possible genesis of the minerals in a well-structured bullet-point format(You may also draw on generally accepted domain knowledge to round out the explanation when appropriate).
While geological evolution should be the primary basis for your reasoning, you may also reference relevant knowledge graph information as supporting evidence.
"""

# ---------- General QA Prompts ----------
GENERAL_SYSTEM_PROMPT = """
You are a planetary science expert. Given knowledge graph paths triples and supporting text retrieved for a question,
Please generate a detailed answer to the question.
Your reasoning mainly refers to the semantic logic of the graph paths!!!!!
"""

GENERAL_USER_PROMPT_TEMPLATE = """
You are a Mars knowledge expert answering the question below.

Question: {question}

### Retrieved Knowledge Graph Paths
{paths}

Retrieved Literature Paragraphs
{texts}

Please first synthesize the relevant information from the knowledge graph paths and retrieved literature paragraphs, then answer the question in detail(use bullet points).
Your reasoning mainly refers to the semantic logic of the Knowledge graph paths. You may also draw on generally accepted domain knowledge to round out the explanation when appropriate.
Ensure that the output content is comprehensive, logically coherent, and detailed, and list the answer in separate items.
"""

# --------------INTENT---------------
INTENT_PROMPT_TEMPLATE = """
You are an assistant for a Martian mineral science system.
Given a user question, classify it into one of the following intent types:\n
1.general_qa: simple descriptive questions about minerals or locations.\n
2.reasoning_qa: explanatory questions requiring general scientific reasoning.\n
3.formation_analysis: questions that involve analyzing mineral presence at specific Mars locations (usually mention coordinates + minerals), aiming to explain the formation cause.\n
Also extract:\n"
- Martian mineral names (e.g., Fe-smectite, olivine)\n
- Geological features (e.g., crater, valley, ridge)\n
- Geographic coordinates (if mentioned) as [lat, lon] format.\n
Return JSON in this format: {\"intent\": \"...\", \"minerals\": [...], \"geo_entities\": [...], \"coordinates\": [[lat, lon], ...]}
"""

# -------------MMQA_simple---------------
FORMATION_ONLY_GEO_PROMPT = """
You are a planetary science expert. Answer the following question based solely on regional geological background and mineral type.

### Question
{question}

### Regional Geological Context
{geo_context_str}

### Instructions
Please follow the reasoning chain below:
1. Geological Background Summary — Summarize the key geological and environmental features of the area.
2. Geological Evolution Inferences — Infer possible geological processes and the evolution of geological environments based on the background.
3. Mineral Formation Inference — Based on the geological reasoning above and the mineral types ({minerals}), infer the possible genesis of the minerals in a well-structured bullet-point format. You may also reference generally accepted domain knowledge.
"""




# -------------Additional Prompts---------------
LLM_filter = """

You are a planetary geology expert responsible for filtering knowledge-graph reasoning paths.

### Question
{question}

### Geological Background
{geo_context}

### Retrieved Knowledge Graph Paths
{paths}
### Instructions
Evaluate each path and remove those that are unlikely to contribute useful geological reasoning.

Use the following considerations:
1. Geological consistency  
The path should not contradict the geological background or well-established geological knowledge.
2. Semantic relevance  
The path should contain information that can help understand or answer the query.
3. Informational contribution  
Prefer paths that provide useful or complementary knowledge rather than repeating trivial or redundant information.
4. Scientific plausibility  
Relations and statements in the path should be geologically reasonable.

### Output
Return the remaining non-redundant knowledge-graph paths that are relevant and geologically plausible.  Keep the original path format unchanged.  Only remove paths that are clearly redundant or irrelevant.

"""

LLM_combine_PEOMPT  = """
You are a planetary geology expert. Combine the user query, the available geological data (if available), and the retrieved entity-related text segments into a geological knowledge context.

### Question
{question}

### Geological Data
{geo_data}

### Entity-related Text Segments
{text_segments}

### Instructions
1. Keep the query and geological data unchanged.
2. Condense the entity-related text segments according to the query and geological data, removing redundant or irrelevant information while preserving geological consistency.

### Output
A merged geological knowledge context.
"""