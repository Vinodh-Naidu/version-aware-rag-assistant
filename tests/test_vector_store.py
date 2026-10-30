from app import vector_store


class FakeEmbeddings:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = sum(ord(c) for c in text) % 100
            vectors.append([float(value), 1.0])
        return vectors


class FakeCollection:
    def __init__(self):
        self.records = {}

    def delete(self, where):
        document_id = where.get("document_id")
        if document_id:
            self.records = {
                key: value
                for key, value in self.records.items()
                if value["metadata"]["document_id"] != document_id
            }

    def add(self, ids, documents, embeddings, metadatas):
        for i, chunk_id in enumerate(ids):
            self.records[chunk_id] = {
                "text": documents[i],
                "embedding": embeddings[i],
                "metadata": metadatas[i],
            }

    def count(self):
        return len(self.records)

    def query(self, query_embeddings, n_results, include):
        records = list(self.records.items())[:n_results]
        return {
            "documents": [[v["text"] for _, v in records]],
            "metadatas": [[v["metadata"] for _, v in records]],
            "distances": [[0.1 for _ in records]],
        }


def test_sync_replaces_old_document_version(monkeypatch):
    collection = FakeCollection()

    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    monkeypatch.setattr(vector_store, "get_embedding_model", lambda: FakeEmbeddings())

    old_document = {
        "document_id": "doc1",
        "current_version": 1,
        "chunks": [{
            "chunk_id": "doc1:v1:0",
            "document_id": "doc1",
            "document_version": 1,
            "filename": "policy.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 15,
            "text": "Limit is $500.",
        }],
    }

    new_document = {
        "document_id": "doc1",
        "current_version": 2,
        "chunks": [{
            **old_document["chunks"][0],
            "chunk_id": "doc1:v2:0",
            "document_version": 2,
            "text": "Limit is $800.",
        }],
    }

    vector_store.sync_document(old_document)
    vector_store.sync_document(new_document)

    assert list(collection.records) == ["doc1:v2:0"]
    assert collection.records["doc1:v2:0"]["metadata"]["document_version"] == 2


def test_search_returns_source_citation(monkeypatch):
    collection = FakeCollection()

    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    monkeypatch.setattr(vector_store, "get_embedding_model", lambda: FakeEmbeddings())
    monkeypatch.setattr(vector_store, "get_document", lambda doc_id: {"current_version": 1})

    document = {
        "document_id": "doc1",
        "current_version": 1,
        "chunks": [{
            "chunk_id": "doc1:v1:0",
            "document_id": "doc1",
            "document_version": 1,
            "filename": "policy.txt",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 20,
            "text": "Limit is $800.",
        }],
    }

    vector_store.sync_document(document)
    results = vector_store.search_chunks("What is the limit?")

    assert len(results) == 1
    assert results[0]["source"] == "policy.txt"
    assert results[0]["document_version"] == 1
    assert results[0]["text"] == "Limit is $800."