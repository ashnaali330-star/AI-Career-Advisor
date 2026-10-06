from rag.retriever import RoleRetriever


def retrieve_job_requirements(retriever: RoleRetriever, role: str) -> str:
    """Return grounded role context for prompts and follow-up questions."""
    results = retriever.retrieve_role(role)
    return "\n\n".join(f"[Source: {item['source']}]\n{item['text']}" for item in results)
