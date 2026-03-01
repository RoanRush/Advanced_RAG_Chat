from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def rerank_documents(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[Document]:
    """
    Score retrieved documents against the query using an LLM prompt,
    then return the top_k highest-scoring ones.
    """
    if not documents:
        return []

    if len(documents) <= top_k:
        return documents

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    scored: list[tuple[float, Document]] = []

    for doc in documents:
        prompt = (
            f"Rate how relevant the following document is to the query on a scale of 0 to 10.\n"
            f"Respond with ONLY a single integer (0-10), nothing else.\n\n"
            f"Query: {query}\n\n"
            f"Document:\n{doc.page_content[:500]}"
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            score = float(response.content.strip().split()[0])
        except Exception:
            score = 5.0

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]
