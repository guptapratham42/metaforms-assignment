from typing import Any, Optional

from pydantic import BaseModel


class ExtractResponse(BaseModel):
    """Response model for extraction endpoint."""

    data: Optional[dict[str, Any]]
    flagged_fields: Optional[dict[str, float]]
    extraction_stats: Optional[dict[str, Any]]


class ComplexityEstimate(BaseModel):
    """Response model for complexity estimation endpoint."""

    complexity: dict[str, int]
    recommended_model: str
    processing_notes: dict[str, bool]
