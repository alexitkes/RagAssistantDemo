"""Шаблоны промптов для LLM."""
from __future__ import annotations

from typing import Iterable

from .storage import RetrievedChunk

SYSTEM_PROMPT = (
    "Ты — RAG-ассистент, который отвечает на вопросы пользователя исключительно "
    "на основе предоставленного контекста из базы знаний. Отвечай на русском языке. "
    "Если в контексте нет ответа на вопрос, честно сообщи об этом и не выдумывай факты. "
    "При ответе можно упоминать источники фрагментов, если это уместно."
)


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    history: Iterable[tuple[str, str]] | None = None,
) -> str:
    """Собирает пользовательский промпт из истории диалога, вопроса и найденных чанков."""
    sections: list[str] = []

    history = list(history) if history else []
    if history:
        history_text = "\n\n".join(
            f"Пользователь: {q}\nАссистент: {a}" for q, a in history
        )
        sections.append("История диалога (предыдущие вопросы и ответы):\n" + history_text)

    context_parts = [
        f"[Фрагмент {i + 1}, источник: {chunk.source}]\n{chunk.text}"
        for i, chunk in enumerate(chunks)
    ]
    context = "\n\n".join(context_parts)
    sections.append(
        "Ниже приведены фрагменты из базы знаний. Используй только их для ответа на вопрос.\n\n"
        "--- КОНТЕКСТ ---\n"
        f"{context}\n"
        "--- КОНЕЦ КОНТЕКСТА ---"
    )
    sections.append(f"Вопрос: {question}\n\nОтвет:")
    return "\n\n".join(sections)
