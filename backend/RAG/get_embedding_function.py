import requests
from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    urls_to_try = ["http://localhost:11434", "http://host.docker.internal:11434"]
    for url in urls_to_try:
        try:
            # Ping Ollama server
            r = requests.get(f"{url}/api/tags", timeout=1)
            if r.status_code == 200:
                print(f"Using Ollama server at {url}")
                return OllamaEmbeddings(base_url=url, model="nomic-embed-text")
        except requests.exceptions.RequestException:
            continue
    raise RuntimeError(
        "No reachable Ollama server found on localhost or host.docker.internal"
    )

