import pytest
from app.utils.text_utils import (
    split_into_sentences,
    count_words,
    estimate_tokens,
    jaccard_similarity,
)
from app.services.conversation_parser import parse_conversation
from app.services.text_ranker import (
    rank_sentences,
    compute_tfidf_scores,
    compute_textrank_scores,
    deduplicate_sentences,
    RankingWeights,
)
from app.services.context_extractor import extract_context
from app.services.context_builder import build_context_pipeline
from app.services.handoff_generator import generate_handoff_text


SAMPLE_TRAVEL_CONVERSATION = """
USER:
I need help planning an 8-day trip to Japan for two people in October.
Our total budget cannot exceed $4,500 excluding international flights.
We must have high-speed rail access for easy transit between cities.

ASSISTANT:
An 8-day itinerary in October is fantastic. The weather is crisp and pleasant.
You could split your time between Tokyo (4 days) and Kyoto (4 days), taking the Shinkansen bullet train between them.
Would you prefer modern luxury hotels or traditional ryokans?

USER:
I prefer boutique hotels or authentic ryokans with onsen baths.
Earlier I was considering renting a car to drive to Mt. Fuji, but after checking toll costs, I've ruled out car rental completely.
We will strictly use public transit and the JR Pass.

ASSISTANT:
That is a wise decision—the rail network is world-class and avoids driving stress.
For accommodation, Kyoto has wonderful ryokans in Gion.
Should we book 3 nights in Tokyo and 5 nights in Kyoto, or a 4/4 split?

USER:
I've decided on 3 nights in Tokyo and 5 nights in Kyoto.
Let's go with the Gion district for our Kyoto stay.
Should I buy the 7-day JR Pass or individual Shinkansen tickets given the recent price increase?
"""


def test_text_utils():
    text = "Hello world! This is ContextBuddy. How does it work? Let's check."
    sentences = split_into_sentences(text)
    assert len(sentences) >= 3
    assert count_words(text) >= 10
    assert estimate_tokens(text) > 0

    # Similarity
    sim = jaccard_similarity("I prefer option A", "Option A is my preference")
    assert sim > 0.3


def test_conversation_parser_standard():
    raw = """
USER:
I want to buy a new camera.

ASSISTANT:
Are you looking for full frame or APS-C?

USER:
I've decided on full frame.
"""
    messages = parse_conversation(raw)
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert "camera" in messages[0].content
    assert messages[1].role == "assistant"
    assert messages[2].role == "user"
    assert "full frame" in messages[2].content


def test_conversation_parser_variations():
    # Markdown bold headers, lowercase
    raw = """
**User:**
Let's discuss my retirement investment plan.

**Assistant:**
Sure, what is your time horizon?

Me:
I have 15 years until retirement.
"""
    messages = parse_conversation(raw)
    assert len(messages) == 3
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"
    assert messages[2].role == "user"


def test_text_ranker_and_weights():
    messages = parse_conversation(SAMPLE_TRAVEL_CONVERSATION)
    ranked = rank_sentences(messages, weights=RankingWeights())
    assert len(ranked) > 0
    # Top ranked sentences should have higher scores than lower ranked
    assert ranked[0].score >= ranked[-1].score
    assert "score" in ranked[0].model_dump()
    assert "features" in ranked[0].model_dump()


def test_deduplication():
    sentences = [
        "I prefer boutique hotels with onsen baths.",
        "I prefer boutique hotels with onsen baths.",  # exact dup
        "Boutique hotels with onsen baths are my preference.",  # near dup
        "Our total budget cannot exceed $4,500.",
    ]
    deduped = deduplicate_sentences(sentences, similarity_threshold=0.6)
    assert len(deduped) < len(sentences)
    assert any("budget" in s for s in deduped)


def test_context_extraction_rules():
    messages = parse_conversation(SAMPLE_TRAVEL_CONVERSATION)
    ranked = rank_sentences(messages)
    context = extract_context(messages, ranked)

    # Primary goal
    assert "trip" in context.primary_goal.lower() or "japan" in context.primary_goal.lower()

    # Constraints
    assert any("budget" in c.lower() or "4,500" in c or "cannot" in c.lower() for c in context.constraints)

    # Decisions
    assert any("tokyo" in d.lower() or "kyoto" in d.lower() or "gion" in d.lower() for d in context.decisions)

    # Rejected ideas
    assert any("car" in r.lower() or "rental" in r.lower() for r in context.rejected_ideas)

    # Open questions
    assert any("jr pass" in q.lower() or "shinkansen" in q.lower() or "?" in q for q in context.open_questions)

    # Preferences
    assert any("boutique" in p.lower() or "ryokan" in p.lower() for p in context.user_preferences)


def test_recency_and_state_overrides():
    conversation = """
USER:
I am considering hiring an agency for our marketing campaign.

ASSISTANT:
Agencies provide full service but can be expensive.

USER:
After reviewing costs, I've ruled out hiring an agency.
I decided to handle marketing in-house with a freelance designer.
"""
    messages = parse_conversation(conversation)
    ranked = rank_sentences(messages)
    context = extract_context(messages, ranked)

    assert any("agency" in r.lower() for r in context.rejected_ideas)
    assert any("in-house" in d.lower() or "designer" in d.lower() for d in context.decisions)


def test_full_pipeline_compression_metrics():
    res = build_context_pipeline(SAMPLE_TRAVEL_CONVERSATION)
    assert res.metrics.original_words > 0
    assert res.metrics.compressed_words > 0
    assert res.metrics.original_tokens_estimate > 0
    assert res.metrics.compressed_tokens_estimate > 0
    assert res.portable_context != ""
    assert "CONTEXTBUDDY — PORTABLE CONTEXT" in res.portable_context
    assert "INSTRUCTION FOR NEXT AI" in res.portable_context
