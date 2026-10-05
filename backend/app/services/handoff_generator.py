from app.models.schemas import StructuredContext


def generate_handoff_text(context: StructuredContext) -> str:
    """
    Renders structured context into a clean, portable context package
    designed to be pasted directly into any downstream AI assistant.
    """
    sections = []

    sections.append("-----------------------------------")
    sections.append("CONTEXTBUDDY — PORTABLE CONTEXT")
    sections.append("-----------------------------------\n")

    if context.conversation_type:
        sections.append("CONVERSATION TYPE")
        sections.append(f"{context.conversation_type}\n")

    if context.primary_goal:
        sections.append("PRIMARY GOAL")
        sections.append(f"{context.primary_goal}\n")

    if context.secondary_goals:
        sections.append("SECONDARY GOALS")
        for g in context.secondary_goals:
            sections.append(f"- {g}")
        sections.append("")

    if context.important_facts:
        sections.append("IMPORTANT FACTS")
        for f in context.important_facts:
            sections.append(f"- {f}")
        sections.append("")

    if context.user_preferences:
        sections.append("USER PREFERENCES")
        for p in context.user_preferences:
            sections.append(f"- {p}")
        sections.append("")

    if context.constraints:
        sections.append("CONSTRAINTS")
        for c in context.constraints:
            sections.append(f"- {c}")
        sections.append("")

    if context.decisions:
        sections.append("DECISIONS")
        for d in context.decisions:
            sections.append(f"- {d}")
        sections.append("")

    if context.rejected_ideas:
        sections.append("REJECTED / RULED-OUT OPTIONS")
        for r in context.rejected_ideas:
            sections.append(f"- {r}")
        sections.append("")

    if context.assumptions:
        sections.append("ASSUMPTIONS")
        for a in context.assumptions:
            sections.append(f"- {a}")
        sections.append("")

    if context.important_entities:
        sections.append("KEY ENTITIES")
        for e in context.important_entities:
            sections.append(f"- {e}")
        sections.append("")

    if context.current_state:
        sections.append("CURRENT STATE")
        for s in context.current_state:
            sections.append(f"- {s}")
        sections.append("")

    if context.open_questions:
        sections.append("OPEN QUESTIONS")
        for q in context.open_questions:
            sections.append(f"- {q}")
        sections.append("")

    if context.recent_context:
        sections.append("RECENT CONTEXT")
        for rc in context.recent_context:
            sections.append(f"- {rc}")
        sections.append("")

    if context.last_user_intent:
        sections.append("LAST USER INTENT")
        sections.append(f"{context.last_user_intent}\n")

    sections.append("INSTRUCTION FOR NEXT AI")
    sections.append(
        "Continue seamlessly from the context above. Do not ask the user to repeat "
        "information already provided. Respect the documented decisions, constraints, "
        "and current state when formulating your response.\n"
    )
    sections.append("-----------------------------------")

    return "\n".join(sections)
