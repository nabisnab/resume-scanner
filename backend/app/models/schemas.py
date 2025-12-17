"""
Pydantic schemas for the Resume Scanner API.

This module defines all request and response models for the resume scanning service,
including validation, serialization, and documentation for API endpoints.
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime


# ============================================================================
# Request Models
# ============================================================================


class ResumeUploadRequest(BaseModel):
    """Schema for resume file upload requests."""
    
    file_name: str = Field(..., description="Name of the resume file")
    file_content: str = Field(..., description="Base64 encoded file content")
    file_type: str = Field(..., description="MIME type of the file (e.g., application/pdf)")
    user_id: Optional[str] = Field(None, description="Optional user ID for tracking")
    
    class Config:
        schema_extra = {
            "example": {
                "file_name": "john_doe_resume.pdf",
                "file_content": "JVBERi0xLjQK...",
                "file_type": "application/pdf",
                "user_id": "user_123"
            }
        }


class AdminWeightsUpdate(BaseModel):
    """Schema for updating scoring weights in admin panel."""
    
    skills_weight: float = Field(..., ge=0, le=1, description="Weight for skills scoring (0-1)")
    experience_weight: float = Field(..., ge=0, le=1, description="Weight for experience scoring (0-1)")
    education_weight: float = Field(..., ge=0, le=1, description="Weight for education scoring (0-1)")
    keywords_weight: float = Field(..., ge=0, le=1, description="Weight for keywords matching (0-1)")
    
    @validator('skills_weight', 'experience_weight', 'education_weight', 'keywords_weight')
    def weights_sum_to_one(cls, v, values):
        """Validate that weights sum to approximately 1.0"""
        if len(values) == 3:  # Last validator call
            total = v + sum(values.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError('All weights must sum to 1.0')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "skills_weight": 0.35,
                "experience_weight": 0.30,
                "education_weight": 0.20,
                "keywords_weight": 0.15
            }
        }


# ============================================================================
# Response Models
# ============================================================================


class ResumeMetadata(BaseModel):
    """Schema containing resume metadata and extracted information."""
    
    resume_id: str = Field(..., description="Unique identifier for the resume")
    file_name: str = Field(..., description="Original file name")
    upload_date: datetime = Field(..., description="Timestamp of resume upload")
    total_pages: Optional[int] = Field(None, description="Number of pages in resume")
    extracted_text_length: int = Field(..., description="Length of extracted text in characters")
    file_type: str = Field(..., description="MIME type of uploaded file")
    
    class Config:
        schema_extra = {
            "example": {
                "resume_id": "res_abc123def456",
                "file_name": "john_doe_resume.pdf",
                "upload_date": "2025-12-17T12:49:26",
                "total_pages": 2,
                "extracted_text_length": 3456,
                "file_type": "application/pdf"
            }
        }


class ScoreResponse(BaseModel):
    """Schema for individual scoring component response."""
    
    component: str = Field(..., description="Name of the scoring component")
    score: float = Field(..., ge=0, le=100, description="Score value from 0 to 100")
    weight: float = Field(..., ge=0, le=1, description="Weight of this component in final score")
    weighted_score: float = Field(..., ge=0, le=100, description="Weighted contribution to final score")
    details: Optional[str] = Field(None, description="Additional details about the score")
    
    class Config:
        schema_extra = {
            "example": {
                "component": "skills_match",
                "score": 85.5,
                "weight": 0.35,
                "weighted_score": 29.925,
                "details": "Found 12 relevant technical skills"
            }
        }


class RecommendationItem(BaseModel):
    """Schema for individual recommendation item."""
    
    title: str = Field(..., description="Title of the recommendation")
    priority: str = Field(..., description="Priority level: high, medium, or low")
    description: str = Field(..., description="Detailed description of the recommendation")
    impact: str = Field(..., description="Potential impact of implementing this recommendation")
    section: Optional[str] = Field(None, description="Resume section this recommendation applies to")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Add Technical Skills Section",
                "priority": "high",
                "description": "Consider adding a dedicated technical skills section listing programming languages and tools",
                "impact": "Could improve skills matching score by 15-20%",
                "section": "skills"
            }
        }


class RecommendationsResponse(BaseModel):
    """Schema for resume recommendations response."""
    
    resume_id: str = Field(..., description="ID of the analyzed resume")
    recommendations: List[RecommendationItem] = Field(..., description="List of improvement recommendations")
    general_notes: Optional[str] = Field(None, description="General observations about the resume")
    
    class Config:
        schema_extra = {
            "example": {
                "resume_id": "res_abc123def456",
                "recommendations": [
                    {
                        "title": "Add Technical Skills Section",
                        "priority": "high",
                        "description": "Consider adding a dedicated technical skills section",
                        "impact": "Could improve score by 15-20%",
                        "section": "skills"
                    }
                ],
                "general_notes": "Resume has strong experience but could benefit from better formatting"
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    error: str = Field(..., description="Error code or type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of error occurrence")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "FILE_PARSE_ERROR",
                "message": "Failed to parse PDF file. The file may be corrupted or encrypted.",
                "details": {
                    "file_name": "resume.pdf",
                    "error_code": "PDF_EXTRACTION_FAILED"
                },
                "timestamp": "2025-12-17T12:49:26"
            }
        }


# ============================================================================
# Combined Response Models
# ============================================================================


class ResumeAnalysisResponse(BaseModel):
    """Combined response for complete resume analysis."""
    
    metadata: ResumeMetadata = Field(..., description="Resume metadata and file information")
    scores: List[ScoreResponse] = Field(..., description="Individual component scores")
    overall_score: float = Field(..., ge=0, le=100, description="Final overall resume score")
    recommendations: RecommendationsResponse = Field(..., description="Recommendations for improvement")
    
    class Config:
        schema_extra = {
            "example": {
                "metadata": {
                    "resume_id": "res_abc123def456",
                    "file_name": "john_doe_resume.pdf",
                    "upload_date": "2025-12-17T12:49:26",
                    "total_pages": 2,
                    "extracted_text_length": 3456,
                    "file_type": "application/pdf"
                },
                "scores": [
                    {
                        "component": "skills_match",
                        "score": 85.5,
                        "weight": 0.35,
                        "weighted_score": 29.925,
                        "details": "Found 12 relevant technical skills"
                    }
                ],
                "overall_score": 78.5,
                "recommendations": {
                    "resume_id": "res_abc123def456",
                    "recommendations": [],
                    "general_notes": "Strong resume overall"
                }
            }
        }
