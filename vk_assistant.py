#!/usr/bin/env python3
"""
RAG-ассистент для сообщества ВКонтакте (Longpoll API).

Скрипт:
  1. создаёт объект Longpoll через функцию create_longpoll (по аналогии с vk_example.py):
     сначала пробует VkBotLongPoll (ключ доступа сообщества, groups.getLongPollServer),
     при неудаче переключается на VkLongPoll (ключ доступа пользователя с правами
     messages, messages.getLongPollServer);
  2. слушает новые входящие сообщения в диалогах/беседах сообщества;
  3. для каждого вопроса формирует ответ через RagAssistant.ask_question,
     передавая uid = "vk_{peer_id}" (отдельный кэш ответов на каждый диалог);
  4. отправляет ответ обратно пользователю в тот же диалог через messages.send.

Токен доступа берётся из .env (см. .env.example):
    VK_TOKEN    — токен доступа (сообщества или пользователя с правами messages);
    VK_GROUP_ID — идентификатор сообщества (опционально).

Пример запуска:
    python vk_assistant.py
"""
from __future__ import annotations

import logging
import random
import time

import vk_api
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.longpoll import VkEventType, VkLongPoll

from config import Config, ConfigError
from rag.assistant import RagAssistant

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000       # лимит VK на длину текста сообщения (4096)
RECONNECT_DELAY_SECONDS = 3     # пауза перед переподключением после ошибки


def _split_message(text: str, limit: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает длинный текст на части не длиннее limit (по границам строк)."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def create_longpoll(vk_session, group_id: int | None):
    """Создать объект Longpoll для получения сообщений от VK (по аналогии с vk_example.py).

    Сначала пробуем Longpoll сообщества (VkBotLongPoll, нужен ключ сообщества).
    Если он не создался (например, используется пользовательский токен — VK API
    вернёт ошибку 15 «Access denied»), переключаемся на Longpoll пользователя
    (VkLongPoll, нужен токен пользователя с правами messages).
    """
    if group_id is not None:
        try:
            longpoll = VkBotLongPoll(vk_session, group_id=group_id)
            logger.info("Longpoll запущен, group_id=%s...", group_id)
            return longpoll
        except Exception:  # noqa: BLE001
            logger.warning("Запустить Longpoll для группы не удалось, пробуем для пользователя...")

    longpoll = VkLongPoll(vk_session, group_id=group_id)
    logger.info("Longpoll запущен...")
    return longpoll


def _extract_message(event) -> tuple[int | None, str]:
    """Возвращает (peer_id, text) из события VkLongPoll или VkBotLongPoll."""
    message = getattr(event, "message", None)
    if isinstance(message, dict):
        # Событие VkBotLongPoll: event.message — словарь с полями сообщения
        return message.get("peer_id"), (message.get("text") or "").strip()
    # Событие VkLongPoll: поля лежат на самом событии
    return getattr(event, "peer_id", None), (getattr(event, "text", None) or "").strip()


def _is_incoming_message(event) -> bool:
    """True, если событие — новое входящее сообщение от пользователя."""
    if event.type == VkEventType.MESSAGE_NEW:
        return bool(getattr(event, "to_me", False))
    if event.type == VkBotEventType.MESSAGE_NEW:
        message = getattr(event, "message", None)
        return bool(message and not message.get("out"))
    return False


def send_text(vk_session, peer_id: int, text: str) -> None:
    """Отправляет текст в диалог peer_id (при необходимости — несколькими сообщениями)."""
    for part in _split_message(text):
        vk_session.method(
            "messages.send",
            {
                "peer_id": peer_id,
                "message": part,
                # random_id — 64-битное целое (int64): берём 63 бита, чтобы
                # значение гарантированно не выходило за пределы 2^63-1
                "random_id": random.getrandbits(63),
            },
        )
        time.sleep(0.4)  # не упираться в лимиты частоты запросов VK API


def handle_event(vk_session, event, assistant: RagAssistant) -> None:
    """Обработать сообщение от пользователя VK: сформировать ответ и отправить."""
    if not _is_incoming_message(event):
        return

    peer_id, text = _extract_message(event)
    if not peer_id or not text:
        logger.info("Пропущено входящее сообщение без текста (peer_id=%s)", peer_id)
        return

    # Отдельный uid на каждый диалог: свой кэш и история для каждого собеседника
    uid = f"vk_{peer_id}"
    logger.info("[%s] Получено сообщение от %s: '%s'", uid, peer_id, text[:120])
    try:
        answer = assistant.ask_question(text, uid)
    except Exception:  # noqa: BLE001
        logger.exception("Ошибка при формировании ответа")
        answer = "Извините, не удалось сформировать ответ. Попробуйте позже."

    send_text(vk_session, peer_id, answer)
    logger.info("[%s] Ответ отправлен (%d симв.)", uid, len(answer))


def run_bot(vk_session, assistant: RagAssistant, group_id: int | None) -> None:
    """Основной цикл: слушает сообщения сообщества, отвечает на них, переподключается."""
    while True:
        try:
            longpoll = create_longpoll(vk_session, group_id)
        except Exception:  # noqa: BLE001
            logger.exception("Запустить Longpoll не удалось")
            time.sleep(RECONNECT_DELAY_SECONDS)
            continue

        logger.info("Ожидание сообщений...")
        try:
            for event in longpoll.listen():
                try:
                    handle_event(vk_session, event, assistant)
                except Exception:  # noqa: BLE001
                    logger.exception("Обработать событие не удалось")
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Longpoll оборвался, переподключение...")
            time.sleep(RECONNECT_DELAY_SECONDS)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        raise SystemExit(1)

    if not config.vk_token:
        print("Ошибка: не задан VK_TOKEN — добавьте токен доступа в файл .env (см. .env.example)")
        raise SystemExit(1)

    vk_session = vk_api.VkApi(token=config.vk_token, api_version=config.vk_api_version)

    assistant = RagAssistant(config)
    try:
        run_bot(vk_session, assistant, config.vk_group_id)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
