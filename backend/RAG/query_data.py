import argparse
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from RAG.get_embedding_function import get_embedding_function
from google import genai
from google.genai import types

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_rag(args.query_text)


def query_rag(query_text: str):
    # Prepare DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search DB
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Build prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Call Gemini
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        ),
    )

    # Extract just the text
    if response.candidates and response.candidates[0].content.parts:
        response_text = "".join(
            part.text
            for part in response.candidates[0].content.parts
            if hasattr(part, "text")
        )
    else:
        response_text = "(No text returned by model)"

    # Show only answer + sources
    sources = [doc.metadata.get("id", None) for doc, _ in results]
    print(f"Response:\n{response_text}\n")
    print(f"Sources: {sources}")

    return response_text


if __name__ == "__main__":
    main()
