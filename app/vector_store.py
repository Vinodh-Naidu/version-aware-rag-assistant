from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_chunks_minilm_v1"


@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def sync_document(document: dict) -> dict:
    collection = get_collection()

    document_id = document["document_id"]
    version = document["current_version"]
    chunks = document["chunks"]

    collection.delete(where={"document_id": document_id})

    if not chunks:
        return {
            "document_id": document_id,
            "version": version,
            "indexed_chunks": 0,
        }

    texts = [chunk["text"] for chunk in chunks]

    raw_embeddings = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if hasattr(raw_embeddings, "tolist"):
        embeddings = raw_embeddings.tolist()
    else:
        embeddings = np.array(raw_embeddings).tolist()

    ids = [chunk["chunk_id"] for chunk in chunks]

    metadatas = [
        {
            "document_id": chunk["document_id"],
            "document_version": chunk["document_version"],
            "filename": chunk["filename"],
            "chunk_index": chunk["chunk_index"],
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return {
        "document_id": document_id,
        "version": version,
        "indexed_chunks": len(chunks),
    }


def search_chunks(
        question: str,
        top_k: int = 3,
) -> list[dict]:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10.")

    collection = get_collection()

    if collection.count() == 0:
        return []

    raw_query_embedding = get_embedding_model().encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    if hasattr(raw_query_embedding, "tolist"):
        query_embedding = raw_query_embedding.tolist()
    else:
        query_embedding = np.array(raw_query_embedding).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, text in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]
        distance = results["distances"][0][i]

        matches.append({
            "text": text,
            "source": metadata["filename"],
            "document_id": metadata["document_id"],
            "document_version": metadata["document_version"],
            "chunk_index": metadata["chunk_index"],
            "start_char": metadata["start_char"],
            "end_char": metadata["end_char"],
            "distance": distance,
        })

    return matches