from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class AnalyzeRequest(BaseModel):
    source_type: str = Field(default="text", description="Source format/type (e.g. 'text')")
    content: str = Field(..., description="Raw conversation text")


class CompressionMetrics(BaseModel):
    original_words: int = 0
    compressed_words: int = 0
    original_tokens_estimate: int = 0
    compressed_tokens_estimate: int = 0
    compression_percentage: float = 0.0
    original_characters: Optional[int] = 0
    compressed_characters: Optional[int] = 0


class ParsedMessage(BaseModel):
    role: str = "unknown"  # "user", "assistant", "unknown"
    content: str
    index: int


class RankedSentence(BaseModel):
    text: str
    message_index: int
    role: str
    score: float
    features: Dict[str, float] = Field(default_factory=dict)


class StructuredContext(BaseModel):
    conversation_type: str = ""
    primary_goal: str = ""
    secondary_goals: List[str] = Field(default_factory=list)
    important_facts: List[str] = Field(default_factory=list)
    user_preferences: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    rejected_ideas: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    important_entities: List[str] = Field(default_factory=list)
    completed_items: List[str] = Field(default_factory=list)
    current_state: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    recent_context: List[str] = Field(default_factory=list)
    last_user_intent: str = ""


class AnalyzeResponse(BaseModel):
    conversation_type: str = ""
    primary_goal: str = ""
    secondary_goals: List[str] = Field(default_factory=list)
    important_facts: List[str] = Field(default_factory=list)
    user_preferences: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list)
    rejected_ideas: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    important_entities: List[str] = Field(default_factory=list)
    completed_items: List[str] = Field(default_factory=list)
    current_state: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    recent_context: List[str] = Field(default_factory=list)
    last_user_intent: str = ""

    portable_context: str = ""
    metrics: CompressionMetrics
