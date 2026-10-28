
from fastapi import FastAPI

app = FastAPI(
    title="VersionAware RAG Assistant",
    description=(
        "A RAG knowledge assistant with "
        "version-aware cache invalidation."
    ),
    version="0.1.0",
)


@app.get("/api/health")
def health():
    return {
        "status": "UP",
        "application": "VersionAware RAG Assistant",
    }
