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


## 🔧 How to Use the Code

### 1. Required Data

Before running the MMAgent.py, make sure you have access to the following data:

#### **Data**
**Mars Mineral Knowledge Graph (MMKG)**  
   A structured graph containing mineral entities, geological relationships, and provenance information.  

**Mars Mineral Text Embedding Corpus**  
   A text vector database derived from scientific literature, mission reports, and mineralogical datasets.  

**Multi-source Geological Data**  
   Supplementary datasets providing contextual geological information such as terrain, mineral associations, and formation environments.  
   - **Mineral abundance and surface thermal inertia:** [ASU Mars Data Portal](https://mars.asu.edu/data/)  
   - **Surface albedo:** [ESA Planetary Science Archive](https://www.cosmos.esa.int/web/psa/mars-maps)  
   - **Geologic timescale and units:** [USGS SIM 3292 – Mars Global Geologic GIS Database](https://pubs.usgs.gov/sim/3292/)  
     → Data file: `SIM3292_MarsGlobalGeologicGIS_20M/SIM3292_Shapefiles/SIM3292_Global_Geology.shp`  
   - **Topography and elevation:** [USGS Blended Global DEM (MOLA + HRSC)](https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m)  
   - **HiRISE imagery:** [University of Arizona HiRISE Archive](https://www.uahirise.org/anazitisi.php)  
     *(Crawler script included; manual collection not required)*  
   - **Impact craters:** [Robbins Crater Database](https://craters.sjrdesign.net/)  
   - **Paleolake basins:** [UT Austin Shared Data – Goudge Lab](https://www.jsg.utexas.edu/goudge/shared-data/)  
   - **Valley networks:** [AGU Earth and Space Science – Global Valley Network Database](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2018EA000362)

> **Note:**  
> The Mars Mineral Knowledge Graph and text corpus will be publicly released after the corresponding paper is published.  
> In the near future, we also plan to open-source the full pipeline for **knowledge graph construction, representation learning, and knowledge fusion**, ensuring full compatibility with the algorithms presented here.


### 2. Required Model

1. **GPT Series Models**  
   Used for intent classification, reasoning, and answer generation.  
   The API endpoint and model configuration can be set in `proxy_config.py`.

2. **Embedding Model: [`bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)**  
   Used to generate vector embeddings for entities, paragraphs, and questions.



### 3. Running Without Graph or Corpus Data

Even without the Mars Mineral Knowledge Graph or text corpus,  
you can still perform **geological data–driven mineral formation analysis** using the multi-source datasets above.

#### **Steps**

**Download Geological Datasets**  
   Obtain and store the datasets listed above locally.

**Set Data Paths**  
   In `geo_context_summary.py`, specify the local file paths for each dataset.

**Configure API Access**  
   In `proxy_config.py`, set your OpenAI (or compatible) API key and base URL:
   ```python
   API_KEY = "your_api_key"
   BASE_URL = "your_base_url"

You can execute the minimal configuration of MMAgent using only the embedded geological reasoning module.  
Set the following feature variables in your configuration:

```python
FEATURES_FOR_QUERY = ["epoch", "hirise", "crater", "valley", "mineral"]
INCLUDE_FOR_GEO_SUMMARY = ["epoch", "hirise_all", "craters", "mineral_data"]
```
> **Note:**  
> This simplified version does not rely on the knowledge graph or text corpus.
> The reasoning process is based solely on geological data, and thus its traceability, comprehensiveness, and answer reliability are considerably lower than those of the full version. It is provided for reference only and intended for lightweight testing or demonstration purposes.

---


