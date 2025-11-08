# 🌍 Project Overview
This project develops a **Mars Mineral Question Answering System (MMQA)** — a graph-driven reasoning framework that combines **knowledge graphs**, **geological constraints**, and **LLMs** to support scientific interpretation of Martian minerals.
The system is built upon a unified reasoning paradigm called **RBG²**.  
RBG² retrieves multi-hop knowledge graph paths guided by geological representations and verifies them through domain consistency checks, ensuring both **semantic relevance** and **geological plausibility**.  
These graph-derived reasoning chains are then integrated with geological context and text evidence to generate interpretable conclusions.
This approach allows MMQA to perform multi-level reasoning — from factual description to causal interpretation — enabling a transparent and traceable analysis of Martian mineral formation mechanisms.
The system handles **three major types of tasks**:

### 1. General QA  
Answers factual or descriptive questions about Martian minerals based on structured and textual knowledge.  
**Example:**  
> “Please introduce the olivine on Mars.”

### 2. General Reasoning QA  
Performs multi-hop reasoning across multiple sources to infer scientific insights or logical relations.  
**Example:**  
> “Please speculate on the water activity in Valles Marineris.”

### 3. Formation Analysis (Multi-source Data Reasoning QA)  
Integrates geological background, mineralogical data, and literature knowledge to infer the formation mechanisms of minerals.  
**Example:**  
> “At 109.9°E, 25.1°N on Mars, sulfate was detected. What could be the formation mechanism?”

## 📁 Project Structure and File Descriptions
This section introduces the major modules of the MMQA system and their core functions.

### 🔹 Core Modules
- **`MMAgentV2.py`** — The main controller that runs the complete Mars Mineral QA pipeline, integrating all retrieval, reasoning, and generation modules.
- **`MMQAsimple.py`** — A simplified version of the pipeline used for quick demonstration or geological-only reasoning without graph retrieval.

### 🔹 Knowledge Graph and Text Retrieval
- **`graph_query.py`** — Handles knowledge graph exploration and path retrieval (1-hop to 3-hop). Supports filtering, path reconstruction, and source tracking.
- **`embedding_utils.py`** — Provides management of the embedding model and implements functions for generating, combining, and comparing semantic embeddings of text
- **`path_selector.py`** — Implements scoring and selection of graph paths using embedding similarity and domain constraints.
- **`link_scorer.py`** — Defines algorithms for ranking and evaluating multi-hop relation chains based on semantic and geological relevance.
- **`retrieval_with_context_v2.py`** — Integrates knowledge graph retrieval with text-based context retrieval; builds structured inputs for reasoning.
- **`text_retrival.py`** — Performs dense or hybrid retrieval from corpus paragraphs associated with entities to provide textual evidence.

### 🔹 Geological Background Retrieval 
- **`geo_context_loader.py`** — Loads geological data from local or external sources (epochs, craters, valleys, HiRISE images, mineral data).
- **`geo_context_summary.py`** — Summarizes multi-source geological information into structured text for model input and reasoning.

### 🔹 Intent recognition and reasoning analysis
- **`answer_generator.py`** — Generates the final natural language answers using LLMs; supports both general QA and formation reasoning.
- **`prompt.py`** — Stores all system and user prompt templates for different task types (general QA, reasoning QA, formation analysis, Intent recognition)
- **`proxy_config.py`** — Manages API and proxy configuration, storing keys and base URLs for OpenAI-compatible interfaces.

### 🔹 Documentation
- **`README.md`** — Project documentation, including overview, structure, usage instructions, and dataset information.
