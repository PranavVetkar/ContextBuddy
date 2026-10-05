import re
import logging
from typing import List
from app.models.schemas import ParsedMessage

logger = logging.getLogger(__name__)

# Patterns matching role speaker labels at the beginning of a line or paragraph
USER_PATTERN = re.compile(
    r"^(?:\*{0,2})(?:USER|User|HUMAN|Human|ME|Me|CLIENT|Client|PROMPT|Prompt)(?:\*{0,2})\s*:\s*",
    re.IGNORECASE,
)
ASSISTANT_PATTERN = re.compile(
    r"^(?:\*{0,2})(?:ASSISTANT|Assistant|AI|Ai|BOT|Bot|CHATGPT|ChatGPT|CLAUDE|Claude|GEMINI|Gemini|SYSTEM|System|COPILOT|Copilot)(?:\*{0,2})\s*:\s*",
    re.IGNORECASE,
)

# Combined boundary regex to find any speaker prefix line
ROLE_LINE_REGEX = re.compile(
    r"(?m)^(?:\*{0,2})(USER|HUMAN|ME|CLIENT|PROMPT|ASSISTANT|AI|BOT|CHATGPT|CLAUDE|GEMINI|SYSTEM|COPILOT)(?:\*{0,2})\s*:\s*",
    re.IGNORECASE,
)


def parse_conversation(raw_text: str) -> List[ParsedMessage]:
    """
    Parses raw conversation text into a list of ParsedMessage objects.
    Tolerates varied speaker headers, formatting deviations, markdown bolding,
    and unstructured single-block or dialogue texts.
    """
    if not raw_text or not raw_text.strip():
        return []

    text = raw_text.strip()

    # Find all matches where a line starts with a speaker indicator
    matches = list(ROLE_LINE_REGEX.finditer(text))

    messages: List[ParsedMessage] = []

    if matches:
        # Check if there is text before the first match
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                messages.append(
                    ParsedMessage(
                        role="unknown",
                        content=preamble,
                        index=len(messages),
                    )
                )

        for i, match in enumerate(matches):
            role_token = match.group(1).upper()
            if role_token in {"USER", "HUMAN", "ME", "CLIENT", "PROMPT"}:
                role = "user"
            else:
                role = "assistant"

            start_content = match.end()
            end_content = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start_content:end_content].strip()

            if content:
                messages.append(
                    ParsedMessage(
                        role=role,
                        content=content,
                        index=len(messages),
                    )
                )
    else:
        # Fallback: No explicit USER:/ASSISTANT: markers found.
        # Try checking for alternating double-newline separated paragraphs or markdown quote blocks
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) > 1:
            # Assume starting with user, alternating dialogue if multiple paragraphs
            for idx, p in enumerate(paragraphs):
                # Check inline headers within paragraph
                user_m = USER_PATTERN.match(p)
                asst_m = ASSISTANT_PATTERN.match(p)
                if user_m:
                    role = "user"
                    c = p[user_m.end() :].strip()
                elif asst_m:
                    role = "assistant"
                    c = p[asst_m.end() :].strip()
                else:
                    role = "user" if idx % 2 == 0 else "assistant"
                    c = p
                messages.append(ParsedMessage(role=role, content=c, index=len(messages)))
        else:
            # Single block of conversation text
            messages.append(
                ParsedMessage(
                    role="user",
                    content=text,
                    index=0,
                )
            )

    return messages
