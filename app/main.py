import logging
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field

from app.answer import MODEL_NAME, AnswerGenerationError, generate_answer
from app.answer_cache import (
    create_cache_key,
    get_cached_answer,
    invalidate_document_cache,
    save_cached_answer,
)
from app.ingestion import (
    MAX_FILE_SIZE,
    DocumentError,
    process_document,
)
from app.storage import (
    get_document,
    list_documents,
    upsert_document,
)
from app.vector_store import search_chunks, sync_document

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("version_aware_rag")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="VersionAware RAG Assistant",
    description="Document ingestion and version-aware knowledge management",
    version="0.2.0",
)


# --- Middleware ---
@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    started_at = time.perf_counter()

    try:
        response = await call_next(request)

        duration_ms = round(
            (time.perf_counter() - started_at) * 1000,
            2,
            )

        logger.info(
            "request_completed request_id=%s method=%s path=%s "
            "status=%s duration_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        logger.exception(
            "request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise


# --- Health Route ---
@app.get("/api/health")
def health():
    return {
        "status": "UP",
        "application": "VersionAware RAG Assistant",
    }


# --- Document Ingestion & Storage Routes ---
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
        if result["status"] == "updated":
            invalidate_document_cache(result["document_id"])

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


# --- Retrieval & QA Routes & Models ---
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
def answer_question(payload: AnswerRequest, request: Request):
    try:
        cache_key = create_cache_key(
            question=request.question,
            top_k=request.top_k,
            model_name=MODEL_NAME,
        )

        # 1. Check whether a valid cached answer exists.
        cached_answer = get_cached_answer(cache_key)

        logger.info(
            "answer_cache_lookup request_id=%s cache_hit=%s",
            getattr(request, "request_id", "unknown"),
            cached_answer is not None,
            )

        if cached_answer is not None:
            return cached_answer

        # 2. Generate a fresh answer if the cache misses.
        result = generate_answer(
            question=request.question,
            top_k=request.top_k,
        )

        result["cache_hit"] = False

        # 3. Cache only answers that have valid citations.
        save_cached_answer(
            cache_key=cache_key,
            question=request.question,
            answer=result,
        )

        return result

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