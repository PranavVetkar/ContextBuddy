import logging
from typing import Optional
from app.models.schemas import (
    AnalyzeResponse,
    CompressionMetrics,
    StructuredContext,
    ParsedMessage,
)
from app.services.conversation_parser import parse_conversation
from app.services.text_ranker import rank_sentences, RankingWeights
from app.services.context_extractor import extract_context
from app.services.handoff_generator import generate_handoff_text
from app.utils.text_utils import count_words, estimate_tokens

logger = logging.getLogger(__name__)


def build_context_pipeline(
    raw_text: str,
    weights: Optional[RankingWeights] = None,
) -> AnalyzeResponse:
    """
    Executes the complete ContextBuddy extraction and compression pipeline:
    1. Parses raw conversation text into normalized messages.
    2. Ranks sentences using TF-IDF, TextRank, recency, and role signals.
    3. Extracts structured domain-independent context.
    4. Generates portable handoff package text.
    5. Computes accurate compression and token statistics.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Cannot analyze empty conversation.")

    # 1. Parse conversation
    parsed_messages = parse_conversation(raw_text)
    if not parsed_messages:
        raise ValueError("Could not parse any messages from the provided text.")

    # 2. Rank sentences with configurable weighting
    ranked_sentences = rank_sentences(parsed_messages, weights=weights)

    # 3. Extract domain-independent structured context
    structured_context = extract_context(parsed_messages, ranked_sentences)

    # 4. Generate portable handoff package
    portable_text = generate_handoff_text(structured_context)

    # 5. Calculate compression metrics
    orig_chars = len(raw_text.strip())
    orig_words = count_words(raw_text)
    orig_tokens = estimate_tokens(raw_text)

    comp_chars = len(portable_text.strip())
    comp_words = count_words(portable_text)
    comp_tokens = estimate_tokens(portable_text)

    if orig_words > 0:
        comp_pct = max(0.0, round((1.0 - (comp_words / orig_words)) * 100.0, 1))
    else:
        comp_pct = 0.0

    metrics = CompressionMetrics(
        original_words=orig_words,
        compressed_words=comp_words,
        original_tokens_estimate=orig_tokens,
        compressed_tokens_estimate=comp_tokens,
        compression_percentage=comp_pct,
        original_characters=orig_chars,
        compressed_characters=comp_chars,
    )

    return AnalyzeResponse(
        conversation_type=structured_context.conversation_type,
        primary_goal=structured_context.primary_goal,
        secondary_goals=structured_context.secondary_goals,
        important_facts=structured_context.important_facts,
        user_preferences=structured_context.user_preferences,
        constraints=structured_context.constraints,
        decisions=structured_context.decisions,
        rejected_ideas=structured_context.rejected_ideas,
        assumptions=structured_context.assumptions,
        important_entities=structured_context.important_entities,
        completed_items=structured_context.completed_items,
        current_state=structured_context.current_state,
        open_questions=structured_context.open_questions,
        recent_context=structured_context.recent_context,
        last_user_intent=structured_context.last_user_intent,
        portable_context=portable_text,
        metrics=metrics,
    )
