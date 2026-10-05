import re
import logging
from typing import List, Dict, Tuple, Optional, Set
from app.models.schemas import ParsedMessage, StructuredContext, RankedSentence
from app.utils.text_utils import (
    split_into_sentences,
    clean_bullet_text,
    jaccard_similarity,
    get_word_set,
)
from app.services.text_ranker import deduplicate_sentences

logger = logging.getLogger(__name__)

# --- Rule-Based Extraction Patterns ---

DECISION_PATTERNS = [
    re.compile(r"\b(i('ve| have)?\s*(?:officially|definitely|finally|now)?\s*decided (to|on|that)?\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('ll| will)?\s*(?:definitely|now)?\s*go with\s+.+)", re.IGNORECASE),
    re.compile(r"\b(let's\s*(?:definitely|now)?\s*(use|go with|proceed with|choose)\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('ll| will)?\s*choose\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('ve| have)?\s*(?:officially|definitely|finally)?\s*selected\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('ve| have)?\s*(?:officially|finally)?\s*settled on\s+.+)", re.IGNORECASE),
    re.compile(r"\b(we('ve| have)?\s*agreed on\s+.+)", re.IGNORECASE),
    re.compile(r"\b(we should (use|go with|proceed with)\s+.+)", re.IGNORECASE),
    re.compile(r"\b(my final choice is\s+.+)", re.IGNORECASE),
]

REJECTION_PATTERNS = [
    re.compile(r"\b(i('ve| have)?\s*(?:officially|definitely|completely|also|now)?\s*ruled out\s+.+)", re.IGNORECASE),
    re.compile(r"\b(let's\s*(rule out|skip|drop)\s+.+)", re.IGNORECASE),
    re.compile(r"\b(decided (against|not to)\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i don't want\s+.+)", re.IGNORECASE),
    re.compile(r"\b(not going (to use|with)\s+.+)", re.IGNORECASE),
    re.compile(r"\b(no longer considering\s+.+)", re.IGNORECASE),
    re.compile(r"\b(we can eliminate\s+.+)", re.IGNORECASE),
    re.compile(r"\b(won't (be using|proceed with|choose)\s+.+)", re.IGNORECASE),
]

CONSTRAINT_PATTERNS = [
    re.compile(r"\b(must (not\s+)?|cannot|can't|only|at least|maximum|minimum|budget|deadline|required|not allowed)\b", re.IGNORECASE),
    re.compile(r"\b(under \$?[\d,]+|less than \$?[\d,]+|cannot exceed|within \d+)\b", re.IGNORECASE),
    re.compile(r"\b(strictly|need it by|has to be|needs to be)\b", re.IGNORECASE),
]

PREFERENCE_PATTERNS = [
    re.compile(r"\b(i prefer\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('d| would) rather\s+.+)", re.IGNORECASE),
    re.compile(r"\b(my preference is\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i('d| would) like\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i love\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i lean towards\s+.+)", re.IGNORECASE),
    re.compile(r"\b(ideally,?\s+.+)", re.IGNORECASE),
    re.compile(r"\b(i'm looking for\s+.+)", re.IGNORECASE),
]

QUESTION_PATTERNS = [
    re.compile(r"^(should i|what about|which is better|how do i|can i|can we|is .+ worth it|what should i|would it be better to)\b", re.IGNORECASE),
    re.compile(r"(\?)$"),
    re.compile(r"\b(do you think|any thoughts on|could you suggest|what are the pros and cons)\b", re.IGNORECASE),
]

GOAL_PATTERNS = [
    re.compile(r"\b(i('m| am) (choosing|deciding|evaluating|considering|planning|comparing)\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(i('m| am) trying to\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(i want to\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(i need (help\s+)?(to|with|planning)\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(my goal is to\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(i('m| am) looking to\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(help me\s+[^.!?]+)", re.IGNORECASE),
    re.compile(r"\b(i plan to\s+[^.!?]+)", re.IGNORECASE),
]

COMPLETED_PATTERNS = [
    re.compile(r"\b(already (done|finished|booked|bought|completed|set up|implemented))\b", re.IGNORECASE),
    re.compile(r"\b(i('ve| have) already\s+.+)", re.IGNORECASE),
    re.compile(r"\b(we('ve| have) (already\s+)?(booked|completed|finished|bought)\s+.+)", re.IGNORECASE),
]

ASSUMPTION_PATTERNS = [
    re.compile(r"\b(assuming (that)?|i assume|presumably|given that|if we assume)\b", re.IGNORECASE),
]



def classify_conversation_type(messages: List[ParsedMessage], full_text: str) -> str:
    """
    Algorithmic domain-independent conversation type classification.
    Analyzes intent indicators, keywords, and structural patterns.
    """
    text_lower = full_text.lower()

    scores = {
        "Decision Making & Comparison": 0,
        "Planning & Strategy": 0,
        "Research & Exploration": 0,
        "Problem Solving & Troubleshooting": 0,
        "Creative & Writing": 0,
        "Advisory & Consultation": 0,
    }

    # Keyword and regex tallies
    if re.search(r"\b(compare|versus|vs|pros and cons|which is better|decide|choice|option a|option b)\b", text_lower):
        scores["Decision Making & Comparison"] += 3
    if re.search(r"\b(plan|itinerary|schedule|steps|roadmap|timeline|strategy|milestones)\b", text_lower):
        scores["Planning & Strategy"] += 3
    if re.search(r"\b(how to fix|error|issue|problem|bug|troubleshoot|solve|broken|fails)\b", text_lower):
        scores["Problem Solving & Troubleshooting"] += 3
    if re.search(r"\b(write|draft|rewrite|essay|story|copy|post|script|headline)\b", text_lower):
        scores["Creative & Writing"] += 3
    if re.search(r"\b(research|learn|explain|overview|how does|history|types of|curious about)\b", text_lower):
        scores["Research & Exploration"] += 2
    if re.search(r"\b(recommend|advice|suggest|tips|career|budget|guide me)\b", text_lower):
        scores["Advisory & Consultation"] += 2

    # Check question density
    question_count = len(re.findall(r"\?", full_text))
    if question_count >= 3:
        scores["Advisory & Consultation"] += 1
        scores["Research & Exploration"] += 1

    best_type = max(scores, key=scores.get)
    if scores[best_type] == 0:
        return "General Inquiry & Planning"
    return best_type


def extract_primary_and_secondary_goals(messages: List[ParsedMessage]) -> Tuple[str, List[str]]:
    """Identifies primary user goal from early user messages and secondary goals."""
    primary_goal = ""
    secondary_goals: List[str] = []

    user_messages = [m for m in messages if m.role == "user"]
    if not user_messages:
        user_messages = messages  # fallback

    # Inspect first user message first for the primary conversational intent/goal
    if user_messages:
        first_sents = split_into_sentences(user_messages[0].content)
        for sent in first_sents:
            for pattern in GOAL_PATTERNS:
                m = pattern.search(sent)
                if m:
                    primary_goal = clean_bullet_text(m.group(0))
                    break
            if primary_goal:
                break
        if not primary_goal and first_sents:
            primary_goal = clean_bullet_text(first_sents[0])

    # Check remaining sentences across all user messages for secondary goals
    for msg in user_messages:
        sentences = split_into_sentences(msg.content)
        for sent in sentences:
            for pattern in GOAL_PATTERNS:
                m = pattern.search(sent)
                if m:
                    goal_text = clean_bullet_text(m.group(0))
                    if goal_text and jaccard_similarity(goal_text, primary_goal) < 0.45:
                        if not any(jaccard_similarity(goal_text, g) > 0.45 for g in secondary_goals):
                            secondary_goals.append(goal_text)

    return primary_goal, deduplicate_sentences(secondary_goals)



def extract_entities(messages: List[ParsedMessage]) -> List[str]:
    """
    Extracts important domain-independent named entities or key subjects mentioned repeatedly.
    Looks for capitalized multi-word phrases, quoted phrases, and alphanumeric codes.
    """
    full_text = " ".join([m.content for m in messages])
    
    # 1. Quoted terms (e.g. "Option A", "Model X", 'Kyoto')
    quoted = re.findall(r'["\']([A-Za-z0-9\s\-_]{2,30})["\']', full_text)
    
    # 2. Capitalized phrases (2-4 words, e.g. "New York", "Option A", "Sony A7IV")
    capitalized = re.findall(r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){1,3}\b', full_text)
    
    # Filter out common sentence start phrases
    common_starters = {
        "I Am", "I Have", "We Are", "We Have", "You Can", "Let Us", "Let's Go",
        "It Is", "In Addition", "On The", "For Example", "As Well", "First Of",
    }
    
    candidates = []
    for entity in quoted + capitalized:
        entity = entity.strip()
        if entity not in common_starters and len(entity) > 2:
            candidates.append(entity)

    # Count frequencies
    freq: Dict[str, int] = {}
    for c in candidates:
        norm = c.strip()
        freq[norm] = freq.get(norm, 0) + 1

    # Keep items mentioned or unique candidates
    sorted_entities = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)
    return deduplicate_sentences(sorted_entities[:8])


def resolve_recency_and_state(
    decisions: List[Tuple[str, int]],
    rejections: List[Tuple[str, int]],
    preferences: List[Tuple[str, int]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Distinguishes old/transient statements from final decisions and updates.
    Handles recency: if an earlier preference/choice was superseded by a later rejection or decision,
    the later statement takes priority.
    
    Returns:
    - final_decisions: List[str]
    - final_rejections: List[str]
    - final_preferences: List[str]
    - current_state_items: List[str]
    """
    # Sort all by message index (recency)
    # If a decision is later than a preference with high similarity, the decision rules.
    final_decisions: List[str] = []
    final_rejections: List[str] = []
    final_preferences: List[str] = []
    current_state_items: List[str] = []

    # Sort each list by recency index descending
    decisions_sorted = sorted(decisions, key=lambda x: x[1], reverse=True)
    rejections_sorted = sorted(rejections, key=lambda x: x[1], reverse=True)
    preferences_sorted = sorted(preferences, key=lambda x: x[1], reverse=True)

    # 1. Process rejections first (explicit negatives)
    for rej_text, rej_idx in rejections_sorted:
        final_rejections.append(rej_text)

    # 2. Process decisions: check if a rejection happened AFTER this decision regarding the same topic
    for dec_text, dec_idx in decisions_sorted:
        superseded = False
        for rej_text, rej_idx in rejections_sorted:
            if rej_idx > dec_idx and jaccard_similarity(dec_text, rej_text) > 0.35:
                superseded = True
                break
        if not superseded:
            final_decisions.append(dec_text)

    # 3. Process preferences: check if overridden by a later decision or rejection
    for pref_text, pref_idx in preferences_sorted:
        superseded = False
        for dec_text, dec_idx in decisions_sorted:
            if dec_idx >= pref_idx and jaccard_similarity(pref_text, dec_text) > 0.4:
                superseded = True
                break
        for rej_text, rej_idx in rejections_sorted:
            if rej_idx >= pref_idx and jaccard_similarity(pref_text, rej_text) > 0.35:
                superseded = True
                break
        if not superseded:
            final_preferences.append(pref_text)

    # Synthesize current state statements
    if final_decisions:
        current_state_items.append(f"Confirmed decision: {final_decisions[0]}")
    if final_rejections:
        current_state_items.append(f"Ruled out options: {final_rejections[0]}")
    if final_preferences:
        current_state_items.append(f"Active preference: {final_preferences[0]}")

    return (
        deduplicate_sentences(final_decisions),
        deduplicate_sentences(final_rejections),
        deduplicate_sentences(final_preferences),
        deduplicate_sentences(current_state_items),
    )


def extract_context(
    messages: List[ParsedMessage],
    ranked_sentences: List[RankedSentence],
) -> StructuredContext:
    """
    Extracts structured, domain-independent context representation from conversation messages.
    """
    full_text = "\n".join([m.content for m in messages])

    # 1. Classify conversation type
    conv_type = classify_conversation_type(messages, full_text)

    # 2. Primary and secondary goals
    primary_goal, secondary_goals = extract_primary_and_secondary_goals(messages)

    # Collectors for pattern-based extractions with message index
    raw_decisions: List[Tuple[str, int]] = []
    raw_rejections: List[Tuple[str, int]] = []
    raw_preferences: List[Tuple[str, int]] = []
    raw_constraints: List[str] = []
    raw_open_questions: List[str] = []
    raw_assumptions: List[str] = []
    raw_completed_items: List[str] = []
    raw_facts: List[str] = []

    for msg in messages:
        sentences = split_into_sentences(msg.content)
        for sent in sentences:
            clean_s = clean_bullet_text(sent)
            if len(clean_s) < 6:
                continue

            # Check Rejections
            rej_clause = None
            for p in REJECTION_PATTERNS:
                m = p.search(clean_s)
                if m:
                    rej_clause = clean_bullet_text(m.group(0))
                    break
            if rej_clause:
                raw_rejections.append((rej_clause, msg.index))
                continue

            # Check Decisions
            dec_clause = None
            for p in DECISION_PATTERNS:
                m = p.search(clean_s)
                if m:
                    dec_clause = clean_bullet_text(m.group(0))
                    break
            if dec_clause:
                raw_decisions.append((dec_clause, msg.index))

            # Check Preferences
            pref_clause = None
            for p in PREFERENCE_PATTERNS:
                m = p.search(clean_s)
                if m:
                    pref_clause = clean_bullet_text(m.group(0))
                    break
            if pref_clause and not dec_clause:
                raw_preferences.append((pref_clause, msg.index))

            # Check Constraints
            if any(p.search(clean_s) for p in CONSTRAINT_PATTERNS):
                raw_constraints.append(clean_s)

            # Check Questions
            if any(p.search(clean_s) for p in QUESTION_PATTERNS) or clean_s.endswith("?"):
                raw_open_questions.append(clean_s)

            # Check Completed Items
            if any(p.search(clean_s) for p in COMPLETED_PATTERNS):
                raw_completed_items.append(clean_s)

            # Check Assumptions
            if any(p.search(clean_s) for p in ASSUMPTION_PATTERNS):
                raw_assumptions.append(clean_s)

            # Extract declarative facts (numbers, metrics, state declarations)
            if re.search(r"\b(\d+\s*(days|weeks|months|years|people|dollars|usd|eur|gbp|users|hours|percent|%))\b", clean_s, re.I):
                raw_facts.append(clean_s)

    # 3. Resolve recency and state conflicts
    (
        decisions,
        rejected_ideas,
        user_preferences,
        current_state_items,
    ) = resolve_recency_and_state(raw_decisions, raw_rejections, raw_preferences)

    # 4. Filter facts to remove duplicates already captured in goals, decisions, constraints, or preferences
    filtered_facts: List[str] = []
    already_captured = decisions + rejected_ideas + user_preferences + raw_constraints + [primary_goal]
    for fact in raw_facts:
        if not any(jaccard_similarity(fact, existing) > 0.4 for existing in already_captured):
            filtered_facts.append(fact)

    # Top ranked sentences from importance scoring also contribute to facts if unique and informative
    for ranked in ranked_sentences[:4]:
        s = clean_bullet_text(ranked.text)
        if not s.endswith("?") and not any(p.search(s) for p in REJECTION_PATTERNS):
            if not any(jaccard_similarity(s, existing) > 0.4 for existing in already_captured + filtered_facts):
                filtered_facts.append(s)

    important_facts = deduplicate_sentences(filtered_facts)[:4]

    # 5. Completed items
    completed_items = deduplicate_sentences(raw_completed_items)
    for c in completed_items:
        current_state_items.append(f"Completed: {c}")

    # 6. Constraints
    constraints = deduplicate_sentences(raw_constraints)

    # 7. Assumptions
    assumptions = deduplicate_sentences(raw_assumptions)

    # 8. Important entities
    important_entities = extract_entities(messages)

    # 9. Open questions (filter to most recent / important)
    open_questions = deduplicate_sentences(raw_open_questions)[-3:]

    # 10. Recent context: last exchange
    recent_context: List[str] = []
    for msg in messages[-2:]:
        sents = split_into_sentences(msg.content)
        prefix = f"[{msg.role.upper()}]: "
        for s in sents[:1]:
            s_clean = clean_bullet_text(s)
            if not any(jaccard_similarity(s_clean, item) > 0.5 for item in decisions + open_questions):
                recent_context.append(f"{prefix}{s_clean}")


    # 11. Last user intent: examine the last user message
    last_user_intent = ""
    user_msgs = [m for m in messages if m.role == "user"]
    if user_msgs:
        last_user_sents = split_into_sentences(user_msgs[-1].content)
        if last_user_sents:
            last_user_intent = clean_bullet_text(last_user_sents[-1])
    elif messages:
        last_sents = split_into_sentences(messages[-1].content)
        if last_sents:
            last_user_intent = clean_bullet_text(last_sents[-1])

    return StructuredContext(
        conversation_type=conv_type,
        primary_goal=primary_goal,
        secondary_goals=secondary_goals,
        important_facts=important_facts,
        user_preferences=user_preferences,
        constraints=constraints,
        decisions=decisions,
        rejected_ideas=rejected_ideas,
        assumptions=assumptions,
        important_entities=important_entities,
        completed_items=completed_items,
        current_state=deduplicate_sentences(current_state_items),
        open_questions=open_questions,
        recent_context=recent_context,
        last_user_intent=last_user_intent,
    )
