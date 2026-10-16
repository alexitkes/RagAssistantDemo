"""
Совместимость ragas с новыми версиями langchain-community.

ragas 0.4.x жёстко импортирует `langchain_community.chat_models.vertexai`
и `langchain_community.llms.VertexAI`, которые были удалены в
langchain-community 0.4.x (Vertex AI переехал в отдельный пакет).
Мы не используем Vertex AI, поэтому подставляем заглушки, чтобы
`import ragas` не падал.

Вызывайте `install_ragas_compat()` ПЕРЕД импортом ragas.
"""
from __future__ import annotations

import sys
import types


def install_ragas_compat() -> None:
    """Устанавливает заглушки удалённых модулей Vertex AI для совместимости с ragas."""
    try:
        from langchain_community import chat_models, llms
    except ImportError:
        return

    if not hasattr(chat_models, "vertexai"):
        module = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # заглушка: Vertex AI не используется
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("VertexAI не поддерживается в этом проекте")

        module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module
        chat_models.vertexai = module

    if not hasattr(llms, "VertexAI"):
        class VertexAI:  # заглушка: Vertex AI не используется
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("VertexAI не поддерживается в этом проекте")

        llms.VertexAI = VertexAI
