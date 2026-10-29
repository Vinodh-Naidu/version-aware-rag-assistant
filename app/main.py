from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.answer import AnswerGenerationError, generate_answer
from app.ingestion import (
    DocumentError,
    MAX_FILE_SIZE,
    process_document,
)
from app.storage import (
    get_document,
    list_documents,
    upsert_document,
)
from app.vector_store import search_chunks, sync_document

app = FastAPI(
    title="VersionAware RAG Assistant",
    description="Document ingestion and version-aware knowledge management",
    version="0.2.0",
)


@app.get("/api/health")
def health():
    return {
        "status": "UP",
        "application": "VersionAware RAG Assistant",
    }


@app.post("/api/documents/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    try:
        # Read at most one byte above the configured limit.
        content = await file.read(MAX_FILE_SIZE + 1)

        processed = process_document(filename, content)
        result = upsert_document(processed)

        # Always sync, even for identical uploads, so a previously
        # interrupted indexing operation can be retried.
        stored_document = get_document(result["document_id"])

        if stored_document is None:
            raise HTTPException(
                status_code=500,
                detail="Document metadata was saved but could not be read.",
            )

        index_result = sync_document(stored_document)

        return {
            **result,
            "indexed_chunks": index_result["indexed_chunks"],
        }

    except DocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()


@app.get("/api/documents")
def documents():
    return {"documents": list_documents()}


@app.get("/api/documents/{document_id}")
def document_details(document_id: str):
    document = get_document(document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document


class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.post("/api/retrieve")
def retrieve(request: RetrievalRequest):
    try:
        matches = search_chunks(
            question=request.question,
            top_k=request.top_k,
        )

        return {
            "question": request.question,
            "retrieved_count": len(matches),
            "results": matches,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@app.post("/api/answer")
def answer_question(request: AnswerRequest):
    try:
        return generate_answer(
            question=request.question,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except AnswerGenerationError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc