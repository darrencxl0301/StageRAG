# #stagerag/prompts.py
class PromptTemplates:
    """Centralized prompt templates for StageRAG system"""
    
    NORMALIZE_INPUT_1B = """Your task is to process the information below. You MUST NOT use any prior knowledge. Your output must be derived STRICTLY and SOLELY from the provided text.

Normalize and clean the following user input for better downstream processing.

Input: {user_input}

Tasks:
1.  Expand informal expressions (e.g., "gonna" to "going to").
2.  Fix obvious spelling errors.
3.  Correct grammatical errors.

Requirements:
-   If the input is already well-written, clear, and grammatically correct, simply return the original input.
-   Keep the original meaning intact.
-   Do not add any new information.
-   Make the expression clearer for information retrieval.

Normalized input:"""


    FILTER_ORGANIZE_3B = """Your task is to process the information below. You MUST NOT use any prior knowledge. Your output must be derived STRICTLY and SOLELY from the provided text.

Filter and organize relevant information for the following question:

Question: {question}

Retrieved Information:
{rag_text}

Tasks:
1. Select information that can answer the question
2. Remove irrelevant and duplicate content
3. Organize into clear knowledge points
4. Extract key elements

Output format:
**Relevant Information:**
- Key fact 1
- Key fact 2

**Key Elements:**
- Entities: [names, products, organizations]
- Numbers: [amounts, percentages, quantities]
- Time: [dates, deadlines, periods]
- Actions: [steps, methods, procedures]

Organized information:"""

    GENERATE_ANSWER_1B = """Your task is to process the information below. You MUST NOT use any prior knowledge. Your output must be derived STRICTLY and SOLELY from the provided text.

Based on the organized knowledge, provide a direct answer to the question:

Question: {question}
Organized Knowledge: {organized_knowledge}

Requirements:
1. Answer based strictly on the provided knowledge
2. Include specific details (numbers, names, dates)
3. Keep language natural and fluent
4. Don't reorganize the knowledge format

Answer:"""



    ORGANIZE_KNOWLEDGE_3B = """Your task is to process the information below. You MUST NOT use any prior knowledge. Your output must be derived STRICTLY and SOLELY from the provided text.

Organize the filtered information into a clear knowledge structure:

Filtered Information: {filtered_info}

Organization Requirements:
1. Reorganize by logical relationships
2. Extract key elements
3. Present information in layers
4. Don't delete any filtered information

Output Format:
**Core Knowledge:**
- Key fact 1
- Key fact 2

**Key Elements:**
- Entities: [people/places/products/organizations]
- Quantities: [numbers/amounts/ratios/counts]
- Time: [dates/deadlines/periods/frequency]
- Actions: [steps/methods/processes/procedures]
- Status: [conditions/requirements/results/states]

**Additional Notes:**
- Important details
- Considerations

Organized knowledge:"""

    SYNTHESIZE_EXTRACT_3B = """Objective: Answer the user's question using ONLY the provided "Source Text".

Rules:

    Use ONLY the Source Text. Do not use any outside knowledge or make assumptions.

    Be Exact. Use the same facts, names, numbers, and words from the text.

    Be Thorough. Include all details from the text that help answer the question. Arrange them in a logical order.

User Question: {question}

Source Text:
{retrieved_information}

Task:

1. Detailed Answer:
Based only on the text provided, write a detailed answer to the user's question. Combine the relevant facts into a clear and logical sequence.

2. Supporting Evidence:
List the exact information from the text that you used to build your answer.

    Key Facts:

        [List the key statements from the text.]

    Entities:

        [List all people, places, and organizations mentioned.]

    Numbers/Data:

        [List all numbers, percentages, and quantities.]

    Dates/Time:

        [List all dates and time periods.]
"""


    FINAL_ANSWER_3B = """Your task is to process the information below. You MUST NOT use any prior knowledge. Your output must be derived STRICTLY and SOLELY from the provided text.

Based on the "Comprehensive Draft Answer" and its "Supporting Evidence" provided below, generate a final, high-quality answer to the original user question.

Original Question: {question}

Organized Knowledge:
{organized_knowledge}

---

**Final Answer Requirements:**

1.  **Absolute Faithfulness:** Your final answer must be a polished, fluent version of the `Comprehensive Draft Answer`. Every statement you make MUST be directly traceable to the `Supporting Evidence` list.
2.  **Use Exact Phrasing:** Use the precise terminology, nouns, and phrasing from the provided knowledge. DO NOT introduce any new words, concepts, or interpretations. If a term is not in the `Supporting Evidence`, it cannot be in your answer.
3.  **Complete and Detailed:** Ensure all specific key elements (names, numbers, dates, etc.) listed in the `Supporting Evidence` are included in your final answer. The answer must be complete and not miss any important information.
4.  **Clarity and Flow:** Present the information with clear logic and natural language, but without sacrificing accuracy or adding content.
5.  **No External Knowledge:** Do not add any information, assumptions, or logical inferences that are not explicitly stated in the provided text.

**Answer:**
"""