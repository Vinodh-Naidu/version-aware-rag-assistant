import logging
import time
import uuid

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from app.logging_config import configure_logging
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi import Depends
from app.security import require_api_key

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
configure_logging()

logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="VersionAware RAG Assistant",
    description="Document ingestion and version-aware knowledge management",
    version="0.2.0",
)

# --- Health Route ---
@app.get("/api/health")
def health():
    return {
        "status": "UP",
        "application": "VersionAware RAG Assistant",
    }


# --- Document Ingestion & Storage Routes ---
@app.post("/api/documents/upload", dependencies=[Depends(require_api_key)], status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""

    try:
        # Read at most one byte above the configured limit.
        # content = await file.read(MAX_FILE_SIZE + 1)
        # upload supports 5 MB

        MAX_UPLOAD_SIZE = 5 * 1024 * 1024

        if not file.filename:
            raise DocumentError("Filename is required.")

        filename = file.filename

        if filename.lower().endswith((".pdf", ".txt")) is False:
            raise DocumentError("Only PDF and TXT files are supported.")

        content = await file.read(MAX_UPLOAD_SIZE + 1)

        if len(content) > MAX_UPLOAD_SIZE:
            raise DocumentError("File exceeds the 5 MB upload limit.")

        if not content:
            raise DocumentError("Uploaded file is empty.")
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


@app.get("/api/documents",dependencies=[Depends(require_api_key)])
def documents():
    return {"documents": list_documents()}


@app.get("/api/documents/{document_id}",dependencies=[Depends(require_api_key)])
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


@app.post("/api/retrieve", dependencies=[Depends(require_api_key)])
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


@app.post("/api/answer", dependencies=[Depends(require_api_key)])
def answer_question(payload: AnswerRequest, request: Request):
    try:
        cache_key = create_cache_key(
            question=payload.question,
            top_k=payload.top_k,
            model_name=MODEL_NAME,
        )

        # 1. Check whether a valid cached answer exists.
        cached_answer = get_cached_answer(cache_key)

        logger.info(
            "answer_cache_lookup request_id=%s cache_hit=%s",
            getattr(request.state, "request_id", "unknown"),
            cached_answer is not None,
            )

        if cached_answer is not None:
            return cached_answer

        # 2. Generate a fresh answer if the cache misses.
        result = generate_answer(
            question=payload.question,
            top_k=payload.top_k,
        )

        result["cache_hit"] = False

        # 3. Cache only answers that have valid citations.
        save_cached_answer(
            cache_key=cache_key,
            question=payload.question,
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

@app.middleware("http")
async def request_logging_middleware(request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id

    start_time = time.perf_counter()

    response = await call_next(request)

    duration_ms = round(
        (time.perf_counter() - start_time) * 1000,
        2,
        )

    response.headers["X-Request-ID"] = request_id

    logger.info(
        "HTTP request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )

    return response
def create_error_response(
        request: Request,
        status_code: int,
        error_code: str,
        message: str,
) -> JSONResponse:

    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": error_code,
                "message": message,
                "request_id": request_id,
            }
        },
        headers={
            "X-Request-ID": request_id,
        },
    )
@app.exception_handler(DocumentError)
async def document_error_handler(
        request: Request,
        exc: DocumentError,
):
    return create_error_response(
        request,
        400,
        "DOCUMENT_PROCESSING_ERROR",
        str(exc),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
):
    return create_error_response(
        request,
        422,
        "VALIDATION_ERROR",
        "The request contains invalid or missing fields.",
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
):
    error_codes = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
    }

    # Do not expose arbitrary internal exception details.
    messages = {
        400: "The request could not be processed.",
        401: "Authentication is required.",
        403: "Access is forbidden.",
        404: "The requested resource was not found.",
        405: "The requested HTTP method is not allowed.",
    }

    error_code = error_codes.get(
        exc.status_code,
        "HTTP_ERROR",
    )

    message = messages.get(
        exc.status_code,
        "The request could not be completed.",
    )

    return create_error_response(
        request,
        exc.status_code,
        error_code,
        message,
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
        request: Request,
        exc: Exception,
):
    logger.exception(
        "Unexpected application error",
        extra={
            "request_id": getattr(
                request.state,
                "request_id",
                "unknown",
            ),
            "method": request.method,
            "path": request.url.path,
        },
    )

    return create_error_response(
        request,
        500,
        "INTERNAL_SERVER_ERROR",
        "An unexpected server error occurred.",
    )
