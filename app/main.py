
from fastapi import FastAPI, File, HTTPException, UploadFile, status

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

        return result

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
