import logging
from fastapi import APIRouter, HTTPException, status
from app.models.schemas import HealthResponse, AnalyzeRequest, AnalyzeResponse
from app.services.conversation_loader import load_conversation
from app.services.context_builder import build_context_pipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Returns system health status."""
    return HealthResponse(status="ok")


@router.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
def analyze_conversation(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyzes raw conversation input, extracts domain-independent structured context,
    ranks important information, builds a portable context handoff package,
    and returns compression metrics.
    """
    if not request.content or not request.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Conversation content cannot be empty.",
        )

    try:
        raw_text = load_conversation(request.source_type, request.content)
        result = build_context_pipeline(raw_text)
        return result
    except ValueError as ve:
        logger.warning("Validation error during analysis: %s", ve)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        logger.exception("Unexpected error processing conversation: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing the conversation. Please try again.",
        )
