FORMATION_SYSTEM_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on four criteria: 
Comprehensiveness,Diversity,Empowerment and Logicality.

---Goal---
You will evaluate two answers to the same question based on four criteria: 
Comprehensiveness, Diversity, Empowerment and Logicality.
- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?
- Logicality: How logically does the answer respond to all parts of the question?

For each criterion, pick the better answer (Answer 1 or Answer 2) and explain why. 
Then select an overall winner and explain why.      
!!!Important: The numbering (Answer 1 / Answer 2) is arbitrary. Do NOT assume that Answer 1 is better just because it appears first. 
Judge each criterion solely based on content quality, not position.

"""

FORMATION_USER_PROMPT_TEMPLATE = """
Question: {query}
Here are the two answers:
Answer 1:{answer1}
Answer 2:{answer2}
For each criterion, choose the better answer (Answer 1 or Answer 2). Then select an overall winner.

Return ONLY JSON (no code fences, no extra text) in this exact shape:{{
  "Comprehensiveness": {{ "Winner": "Answer 1" | "Answer 2", "Explanation": "<why>" }},
  "Diversity":         {{ "Winner": "Answer 1" | "Answer 2", "Explanation": "<why>" }},
  "Empowerment":       {{ "Winner": "Answer 1" | "Answer 2", "Explanation": "<why>" }},
  "Logicality":        {{ "Winner": "Answer 1" | "Answer 2", "Explanation": "<why>" }},
  "Overall Winner":    {{ "Winner": "Answer 1" | "Answer 2", "Explanation": "< Summarize why this answer is the overall winner based on the three criteria >" }}
}}

"""