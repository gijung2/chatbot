"""
Common Pydantic schemas
"""
from pydantic import BaseModel
from typing import Optional

class ErrorResponse(BaseModel):
    """에러 응답"""
    error: str
    detail: Optional[str] = None
    code: Optional[int] = None
