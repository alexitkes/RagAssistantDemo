"""
Клиенты LLM: GigaChat (Sber) и ProxyAPI (OpenAI-совместимый прокси).

Выбор провайдера определяется настройкой LLM_PROVIDER в .env.
"""
from __future__ import annotations

from config import Config

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .storage import RetrievedChunk


class LLMError(RuntimeError):
    """Ошибка взаимодействия с LLM."""


def ask_gigachat(config: Config, user_prompt: str, history: list[tuple[str, str]] | None = None) -> str:
    """Отправляет промпт в GigaChat через официальный пакет gigachat."""
    try:
        from gigachat import GigaChat
        from gigachat.models import Chat, Messages, MessagesRole
    except ImportError as exc:
        raise LLMError("Не установлен пакет gigachat. Выполните: pip install gigachat") from exc

    messages = [Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT)]

    history = list(history) if history else []
    for q, a in history[-config.history_size:]:
        messages.append(Messages(role=MessagesRole.USER, content=q))
        messages.append(Messages(role=MessagesRole.ASSISTANT, content=a))

    messages.append(Messages(role=MessagesRole.USER, content=user_prompt))

    try:
        with GigaChat(
            credentials=config.gigachat_credentials,
            scope=config.gigachat_scope,
            model=config.gigachat_model,
            verify_ssl_certs=config.gigachat_verify_ssl,
        ) as giga:
            response = giga.chat(Chat(messages=messages, temperature=config.temperature))
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Ошибка обращения к GigaChat: {exc}") from exc

    return response.choices[0].message.content


def ask_proxyapi(config: Config, user_prompt: str, history: list[tuple[str, str]] | None = None) -> str:
    """Отправляет промпт через OpenAI-совместимый API (ProxyAPI)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMError("Не установлен пакет openai. Выполните: pip install openai") from exc

    client = OpenAI(api_key=config.proxyapi_key, base_url=config.proxyapi_url)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = list(history) if history else []
    for q, a in history[-config.history_size:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=config.proxyapi_model,
            temperature=config.temperature,
            messages=messages,
        )
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Ошибка обращения к ProxyAPI: {exc}") from exc

    return response.choices[0].message.content


def ask(
    config: Config,
    question: str,
    chunks: list[RetrievedChunk],
    history: list[tuple[str, str]] | None = None,
) -> str:
    """Основная точка входа: формирует промпт и вызывает провайдера.

    Передает провайдеру LLM промпт и историю диалога.
    """
    user_prompt = build_user_prompt(question, chunks)

    if config.llm_provider == "gigachat":
        return ask_gigachat(config, user_prompt, history)

    if config.llm_provider == "proxyapi":
        return ask_proxyapi(config, user_prompt, history)

    raise LLMError(f"Неизвестный LLM-провайдер: {config.llm_provider}")
