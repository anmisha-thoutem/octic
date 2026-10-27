_KNOWLEDGE_BASE = {
    "python": "Python is a versatile, high-level programming language known for readability, broad ecosystem support, and use in AI, automation, data science, and web development.",
    "ai": "Artificial Intelligence is the field of building systems that can perceive, reason, learn, and solve problems in ways similar to human cognition.",
    "llm": "Large language models are neural networks trained on massive text corpora to generate language, answer questions, and support many knowledge tasks.",
    "rag": "Retrieval-augmented generation combines search over external documents with generation from a language model to ground answers in evidence.",
    "langgraph": "LangGraph is a graph-based orchestration framework for building reliable, stateful, multi-step agent workflows with tool calls and memory.",
    "agentic ai": "Agentic AI refers to systems that plan actions, invoke tools, maintain state, and iterate across steps to complete tasks autonomously.",
}


def search_knowledge_base(query: str) -> dict:
    if query is None or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    normalized_query = str(query).lower().strip()
    matches = []
    for topic, content in _KNOWLEDGE_BASE.items():
        topic_lower = topic.lower()
        if topic_lower in normalized_query or normalized_query in topic_lower:
            matches.append({"topic": topic, "content": content})
        elif any(keyword in normalized_query for keyword in topic_lower.split()):
            matches.append({"topic": topic, "content": content})

    if not matches:
        return {
            "matches": [],
            "summary": "No relevant information was found in the local knowledge base for that query.",
        }

    summary = "\n\n".join(f"{item['topic']}: {item['content']}" for item in matches)
    return {
        "matches": [item["topic"] for item in matches],
        "summary": summary,
    }
