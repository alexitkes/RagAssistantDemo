#!/usr/bin/env python3
"""
Сценарий работы RAG-ассистента.

Принимает вопрос пользователя в интерактивном режиме и выводит ответ,
сформированный классом RagAssistant (кэш → векторный поиск → LLM).
Факт обращения к кэшу выводится через logging.

Пример запуска:
    python assistant.py
"""
from __future__ import annotations

import logging

from config import Config, ConfigError
from rag.assistant import RagAssistant

EXIT_COMMANDS = {"exit", "quit", "выход", "q"}


def run_qa_session(assistant: RagAssistant) -> None:
    config = assistant.config
    print(
        "RAG-ассистент готов.\n"
        f"  Векторное хранилище: {config.vector_db}\n"
        f"  LLM: {config.llm_provider}\n"
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
            answer = assistant.ask_question(question)
        except Exception as exc:  # noqa: BLE001
            print(f"Ошибка: {exc}\n")
            continue

        print(f"\nОтвет: {answer}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        raise SystemExit(1)

    assistant = RagAssistant(config)
    try:
        run_qa_session(assistant)
    finally:
        assistant.close()


if __name__ == "__main__":
    main()
