import warnings
from intent_classifier import classify_intent_and_extract_entities
from geo_context_summary import query_all_geological_info, format_question_with_context, summarize_geological_context
from openai import OpenAI
warnings.filterwarnings("ignore", category=FutureWarning)
# drives query_all_geological_info
FEATURES_FOR_QUERY = ["epoch", "hirise", "crater", "valley", "mineral"]
# Controls which fields are for  summarized for embedding
INCLUDE_FOR_QUESTION_CONTEXT = ["hirise_top3"]
# Controls which geological data fields are summarized
INCLUDE_FOR_GEO_SUMMARY = ["epoch", "hirise_all", "craters", "mineral_data"]
from proxy_config import API_KEY, BASE_URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
from prompt import FORMATION_ONLY_GEO_PROMPT
def generate_geo_only_formation_answer(minerals: list, geo_context_str: str, question: str) -> str:
    """
    Generate mineral formation reasoning based solely on geological background.
    No knowledge graph paths or external text retrieval are used.
    Returns a readable text directly.
    """
    # Fill in the placeholders in the prompt template
    prompt_text = FORMATION_ONLY_GEO_PROMPT.format(
        question=question,
        geo_context_str=geo_context_str,
        minerals=', '.join(minerals)
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.5
    )
    return response.choices[0].message.content.strip()

def run_geo_only_formation(question: str) -> str:
    """
    Pipeline for formation analysis without a knowledge graph.
    Retrieves geological context and generates a formation reasoning answer.
    """
    print(f"\n📥 User question: {question}")

    # Classify intent and extract entities
    info = classify_intent_and_extract_entities(question)
    intent = info["intent"]
    minerals = info.get("minerals", [])
    coords = info.get("coordinates", [])
    lat, lon = coords[0] if coords else (None, None)

    # Skip if intent or coordinates are invalid
    if intent != "formation_analysis" or lat is None:
        print("⚠️ No valid formation analysis intent or coordinates found. Skipping.")
        return ""

    # Query regional geological information
    geo_context = query_all_geological_info(lat, lon, features=FEATURES_FOR_QUERY)

    # Format question by including relevant geological context
    question_with_context = format_question_with_context(
        question,
        geo_context,
        include=INCLUDE_FOR_QUESTION_CONTEXT
    )

    # Summarize geological background fields for reasoning
    geo_summary = summarize_geological_context(
        **geo_context,
        include=INCLUDE_FOR_GEO_SUMMARY
    )

    # Generate mineral formation reasoning directly as readable text
    answer = generate_geo_only_formation_answer(
        minerals=minerals,
        geo_context_str=geo_summary,
        question=question_with_context
    )

    # Print the answer directly
    print("\n🧠 Generated answer:\n", answer)
    return answer

if __name__ == "__main__":
    # Example question
    q1 = "At 109.9°E, 25.1°N on Mars, sulfate was detected. What could be the formation mechanism?"
    run_geo_only_formation(q1)
