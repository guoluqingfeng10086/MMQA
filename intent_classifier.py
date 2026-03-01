from openai import OpenAI
import json
import re
from prompt import INTENT_PROMPT_TEMPLATE

from proxy_config import API_KEY, BASE_URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)
def classify_intent_and_extract_entities(question: str) -> dict:
    """
    Given a user question, identify its intent type (general_qa / reasoning / formation_analysis),
    and extract mineral entities, geological entities, and latitude and longitude coordinates.
    """
    system_prompt = INTENT_PROMPT_TEMPLATE
    user_prompt = f"Question: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.5
    )
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        result.setdefault("intent", "unknown")
        result.setdefault("minerals", [])
        result.setdefault("geo_entities", [])
        result.setdefault("coordinates", [])
    except json.JSONDecodeError:
        print("!!!JSON parsing failed; attempting to roll back the parsing....")
        result = {
            "intent": re.search(r'"intent":\s*"(.+?)"', content).group(1) if re.search(r'"intent":\s*"(.+?)"', content) else "unknown",
            "minerals": re.findall(r'"minerals":\s*\[(.*?)\]', content),
            "geo_entities": re.findall(r'"geo_entities":\s*\[(.*?)\]', content),
            "coordinates": re.findall(r'"coordinates":\s*\[(.*?)\]', content),
        }
        # Process strings into lists
        for key in ["minerals", "geo_entities", "coordinates"]:
            if isinstance(result[key], list) and len(result[key]) == 1 and isinstance(result[key][0], str):
                inner = result[key][0]
                result[key] = [x.strip().strip('"') for x in inner.split(",") if x.strip()]
            elif not isinstance(result[key], list):
                result[key] = []
    return result

if __name__ == "__main__":
    q = "Explain the possible genesis of hematite on Mars."
    q = "At 2.13N,0.42W on Mars, Jarosite was detected. What could be the formation mechanism?"
    result = classify_intent_and_extract_entities(q)
    print("Complete results：", result)
    coords = result.get("coordinates", [])
    lat, lon = (coords[0] if coords else (None, None))
    print(f"Analyzed coordinates：lat = {lat}, lon = {lon}")
