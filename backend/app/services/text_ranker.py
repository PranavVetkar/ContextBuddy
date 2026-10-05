import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from app.models.schemas import ParsedMessage, RankedSentence
from app.utils.text_utils import split_into_sentences, jaccard_similarity

logger = logging.getLogger(__name__)


@dataclass
class RankingWeights:
    """Configurable coefficients for sentence importance ranking."""
    tfidf: float = 0.20
    textrank: float = 0.20
    user: float = 0.25
    decision: float = 0.30
    constraint: float = 0.30
    question: float = 0.20
    recency: float = 0.25


# Basic signal patterns for importance feature detection
DECISION_REGEX = re.compile(
    r"\b(i('ve| have)? decided|decided to|let's go with|i('ll| will)? go with|let's use|we should use|"
    r"i prefer|i choose|i('ll| will)? choose|i('ve| have)? selected|i('ve| have)? settled on|let's proceed with)\b",
    re.IGNORECASE,
)

CONSTRAINT_REGEX = re.compile(
    r"\b(must|cannot|can't|only|at least|maximum|minimum|budget|deadline|required|not allowed|don't want|cannot exceed|within)\b",
    re.IGNORECASE,
)

QUESTION_REGEX = re.compile(
    r"(\?|(^|\b)(should i|what about|which is better|how do i|can i|can we|is .+ worth it|what should i|would it be better to)\b)",
    re.IGNORECASE,
)

REJECTED_REGEX = re.compile(
    r"\b(ruled out|rule out|decided against|not going to use|don't want|skip|won't use|no longer considering|passed on)\b",
    re.IGNORECASE,
)


def compute_tfidf_scores(sentences: List[str]) -> List[float]:
    """Calculates normalized TF-IDF importance scores for a list of sentences."""
    if not sentences:
        return []
    if len(sentences) == 1:
        return [1.0]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Average non-zero TF-IDF weight per sentence
        scores = []
        for i in range(tfidf_matrix.shape[0]):
            row = tfidf_matrix.getrow(i).data
            if len(row) > 0:
                scores.append(float(np.mean(row)))
            else:
                scores.append(0.0)

        max_val = max(scores) if scores else 0.0
        if max_val > 0:
            scores = [s / max_val for s in scores]
        return scores
    except Exception as e:
        logger.warning("TF-IDF calculation fallback due to: %s", e)
        return [0.5] * len(sentences)


def compute_textrank_scores(sentences: List[str]) -> List[float]:
    """Computes TextRank graph-based centrality scores for sentences."""
    n = len(sentences)
    if n <= 1:
        return [1.0] * n

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Set diagonal to zero to prevent self-loops
        np.fill_diagonal(sim_matrix, 0)

        # Construct graph
        graph = nx.from_numpy_array(sim_matrix)
        pagerank_scores = nx.pagerank(graph, alpha=0.85, max_iter=200, weight="weight")

        raw_scores = [pagerank_scores.get(i, 0.0) for i in range(n)]
        max_val = max(raw_scores) if raw_scores else 0.0
        if max_val > 0:
            return [s / max_val for s in raw_scores]
        return [1.0 / n] * n
    except Exception as e:
        logger.warning("TextRank calculation fallback due to: %s", e)
        return [0.5] * n


def rank_sentences(
    messages: List[ParsedMessage],
    weights: Optional[RankingWeights] = None,
) -> List[RankedSentence]:
    """
    Splits messages into individual sentences and computes importance scores
    incorporating TF-IDF, TextRank, user role weighting, linguistic signals, and recency.
    """
    if weights is None:
        weights = RankingWeights()

    # Flatten messages to sentences
    sentence_meta: List[Tuple[str, int, str]] = []  # (text, msg_index, role)
    for msg in messages:
        sents = split_into_sentences(msg.content)
        for s in sents:
            if len(s.strip()) > 5:
                sentence_meta.append((s, msg.index, msg.role))

    if not sentence_meta:
        return []

    sentence_texts = [sm[0] for sm in sentence_meta]
    total_sentences = len(sentence_texts)

    tfidf_scores = compute_tfidf_scores(sentence_texts)
    textrank_scores = compute_textrank_scores(sentence_texts)

    ranked_items: List[RankedSentence] = []

    for i, (text, msg_idx, role) in enumerate(sentence_meta):
        is_user = 1.0 if role == "user" else 0.3
        has_decision = 1.0 if DECISION_REGEX.search(text) else 0.0
        has_constraint = 1.0 if CONSTRAINT_REGEX.search(text) else 0.0
        has_question = 1.0 if QUESTION_REGEX.search(text) else 0.0
        has_rejection = 1.0 if REJECTED_REGEX.search(text) else 0.0

        # Recency score from 0.1 to 1.0
        recency = (i + 1) / total_sentences

        features = {
            "tfidf": tfidf_scores[i],
            "textrank": textrank_scores[i],
            "user_weight": is_user,
            "decision": max(has_decision, has_rejection),
            "constraint": has_constraint,
            "question": has_question,
            "recency": recency,
        }

        importance_score = (
            features["tfidf"] * weights.tfidf
            + features["textrank"] * weights.textrank
            + features["user_weight"] * weights.user
            + features["decision"] * weights.decision
            + features["constraint"] * weights.constraint
            + features["question"] * weights.question
            + features["recency"] * weights.recency
        )

        ranked_items.append(
            RankedSentence(
                text=text,
                message_index=msg_idx,
                role=role,
                score=round(importance_score, 4),
                features=features,
            )
        )

    # Sort descending by score
    ranked_items.sort(key=lambda s: s.score, reverse=True)
    return ranked_items


def deduplicate_sentences(
    sentences: List[str],
    similarity_threshold: float = 0.65,
) -> List[str]:
    """
    Filters redundant sentences while preserving the most informative/recent versions.
    Uses Jaccard word-set similarity to detect duplicates.
    """
    unique_sentences: List[str] = []

    for sentence in sentences:
        is_duplicate = False
        for existing in unique_sentences:
            sim = jaccard_similarity(sentence, existing)
            if sim >= similarity_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sentences.append(sentence)

    return unique_sentences
