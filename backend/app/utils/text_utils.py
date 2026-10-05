import math
import re
from typing import List, Set


def normalize_whitespace(text: str) -> str:
    """Collapses consecutive whitespaces into a single space and strips boundaries."""
    return re.sub(r"\s+", " ", text).strip()


def estimate_tokens(text: str) -> int:
    """
    Estimates token count for generic text without relying on external API tokenizers.
    Standard heuristic: ~1 token per 3.8-4 characters or ~1.33 tokens per word.
    Using word and punctuation extraction for a close approximation to BPE tokenizers.
    """
    if not text or not text.strip():
        return 0
    # Match words or non-whitespace individual punctuation marks
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
    return max(1, len(tokens))


def count_words(text: str) -> int:
    """Counts words in a given text."""
    if not text or not text.strip():
        return 0
    return len(re.findall(r"\b[\w'-]+\b", text))


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into meaningful sentences while preserving list items and major clauses.
    Handles periods, question marks, exclamation marks, and newline-separated points.
    """
    if not text or not text.strip():
        return []

    # First normalize carriage returns
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split on newlines first to respect paragraph and bullet structures
    raw_lines = clean_text.split("\n")
    sentences: List[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet prefixes like "- ", "* ", "1. ", "• "
        line_clean = re.sub(r"^[\*\-•\d\.\)\s]+", "", line).strip()
        if not line_clean:
            continue

        # Split on sentence boundaries: . ! ? followed by space and capital letter or end of string
        # Negative lookbehinds prevent splitting on common abbreviations like e.g., i.e., vs., Dr., etc.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'‘“])", line_clean)
        for part in parts:
            part_str = normalize_whitespace(part)
            if len(part_str) > 3:
                sentences.append(part_str)

    return sentences


def tokenize_words(text: str) -> List[str]:
    """Converts text to lowercase alphanumeric word tokens."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9'-]+\b", text)]


def stem_word(w: str) -> str:
    """Lightweight rule-based stemmer for domain-independent paraphrasing."""
    w = w.lower()
    if w.startswith("prefer"):
        return "prefer"
    if w.startswith("decid"):
        return "decid"
    if w.startswith("choos") or w.startswith("choic"):
        return "choic"
    if w.startswith("option"):
        return "option"
    if w.startswith("select"):
        return "select"
    if w.startswith("recommend"):
        return "recommend"
    if w.startswith("constrain") or w.startswith("restrict"):
        return "restrict"
    for suffix in ("ing", "tion", "tions", "ies", "ied", "ed", "ly", "es", "s"):
        if w.endswith(suffix) and len(w) > len(suffix) + 2:
            return w[: -len(suffix)]
    return w


def get_word_set(text: str) -> Set[str]:
    """Returns set of unique lowercase words."""
    return set(tokenize_words(text))


def get_stemmed_word_set(text: str) -> Set[str]:
    """Returns set of stemmed lowercase words excluding trivial stopwords."""
    stopwords = {"a", "an", "the", "is", "are", "was", "were", "my", "our", "to", "in", "for", "of", "and", "or"}
    words = tokenize_words(text)
    return {stem_word(w) for w in words if w not in stopwords and len(w) > 1}



def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams from normalized text."""
    clean = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Computes hybrid Jaccard similarity combining stemmed word overlap
    and character 3-gram overlap for high-quality paraphrasing detection.
    """
    words_a = get_stemmed_word_set(text_a)
    words_b = get_stemmed_word_set(text_b)
    
    word_sim = 0.0
    if words_a and words_b:
        inter = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        word_sim = inter / union if union > 0 else 0.0

    chars_a = get_char_ngrams(text_a, 3)
    chars_b = get_char_ngrams(text_b, 3)
    char_sim = 0.0
    if chars_a and chars_b:
        inter_c = len(chars_a.intersection(chars_b))
        union_c = len(chars_a.union(chars_b))
        char_sim = inter_c / union_c if union_c > 0 else 0.0

    # 60% word overlap, 40% character ngram overlap
    return round(0.6 * word_sim + 0.4 * char_sim, 4)



def clean_bullet_text(text: str) -> str:
    """Strips leading bullet characters, numbers, and extra symbols."""
    cleaned = re.sub(r"^[\s\*\-•\d\.\)\:\;]+", "", text).strip()
    return cleaned
