#!/usr/bin/env python3
"""
Сценарий работы RAG-ассистента.

Принимает вопрос пользователя в интерактивном режиме и выводит ответ,
сформированный классом RagAssistant (кэш → векторный поиск → LLM).
Факт обращения к кэшу выводится через logging.

Идентификатор пользователя (--uid) задаётся из командной строки; для каждого
пользователя используется отдельный кэш ответов. По умолчанию — пустая строка.

Пример запуска:
    python assistant.py
    python assistant.py --uid user-42
"""
from __future__ import annotations

import argparse
import logging

from config import Config, ConfigError
from rag.assistant import RagAssistant

EXIT_COMMANDS = {"exit", "quit", "выход", "q"}


def run_qa_session(assistant: RagAssistant, uid: str = "") -> None:
    config = assistant.config
    print(
        "RAG-ассистент готов.\n"
        f"  Векторное хранилище: {config.vector_db}\n"
        f"  LLM: {config.llm_provider}\n"
        f"  Пользователь (uid): {uid or '(общий кэш)'}\n"
        "Задайте вопрос или введите 'exit' / 'выход' для завершения.\n"
    )

    while True:
        try:
            question = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            break

        try:
            answer = assistant.ask_question(question, uid)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}\n")
            continue

        print(f"\nОтвет: {answer}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="RAG-ассистент")
    parser.add_argument(
        "--uid",
        default="",
        help="Идентификатор пользователя (по умолчанию пустая строка — общий кэш)",
    )
    args = parser.parse_args()

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        raise SystemExit(1)

    assistant = RagAssistant(config)
    try:
        run_qa_session(assistant, args.uid)
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
