
import os
import secrets

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
        api_key: str | None = Security(api_key_header),
) -> None:
    expected_api_key = os.getenv("RAG_API_KEY")

    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )

    if (
            not api_key
            or not secrets.compare_digest(
        api_key,
        expected_api_key,
    )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
