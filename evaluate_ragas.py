#!/usr/bin/env python3
"""
Оценка качества RAG-ассистента метриками Ragas.

Для набора прописанных вопросов по документам из data/ вычисляются метрики:
  * faithfulness      — насколько ответ фактически соответствует найденному контексту;
  * answer_relevancy  — насколько ответ релевантен заданному вопросу;
  * context_precision — насколько найденный контекст релевантен вопросу.

Для вычисления метрик Ragas использует LLM и эмбеддинги через ProxyAPI
(OpenAI-совместимый API), как и сам ассистент.

Пример запуска:
    python evaluate_ragas.py
"""
from __future__ import annotations

import logging

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision

from config import Config, ConfigError
from rag.assistant import RagAssistant
from rag.ragas_compat import install_ragas_compat

# Совместимость ragas с новыми версиями langchain-community (до импорта ragas)
install_ragas_compat()

# Вопросы для оценки и эталонные ответы (ground truth) по документам из data/
EVALUATION_SET = [
    {
        "question": "Сколько дней составляет ежегодный оплачиваемый отпуск?",
        "ground_truth": "Ежегодный оплачиваемый отпуск составляет 28 календарных дней.",
    },
    {
        "question": "Как сменить пароль на портале «Вестник»?",
        "ground_truth": (
            "Пароль меняется в личном кабинете в разделе «Безопасность»; "
            "новый пароль должен содержать не менее 12 символов, включать "
            "буквы разного регистра и цифры."
        ),
    },
    {
        "question": "В какие форматы экспортируются отчёты в продукте «Атлас»?",
        "ground_truth": "Отчёты экспортируются в форматы PDF и Excel.",
    },
    {
        "question": "Что делать при ошибке аутентификации в «Атласе»?",
        "ground_truth": (
            "Проверить, что системное время рабочей станции синхронизировано "
            "с сервером времени компании."
        ),
    },
    {
        "question": "За сколько рабочих дней до выезда подаётся заявка на командировку?",
        "ground_truth": (
            "Заявка на командировку подаётся не позднее чем за пять рабочих дней до выезда."
        ),
    },
]


def _build_ragas_components(config: Config):
    """Создаёт LLM и эмбеддинги для Ragas на основе ProxyAPI (ragas 0.4+)."""
    try:
        from openai import OpenAI
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
    except ImportError as exc:
        raise RuntimeError(
            "Для оценки Ragas нужны пакеты ragas, datasets, langchain-openai "
            "и langchain-community. Выполните: "
            "pip install ragas datasets langchain-openai langchain-community"
        ) from exc

    client = OpenAI(api_key=config.proxyapi_key, base_url=config.proxyapi_url)
    llm = llm_factory(
        config.proxyapi_model,
        client=client,
        temperature=config.temperature,
    )
    embeddings = embedding_factory(
        client=client,
        model=config.proxyapi_embedding_model,
    )

    return llm, embeddings


def _build_metrics(llm, embeddings):
    """Настраивает метрики Ragas: faithfulness, answer_relevancy, context_precision.

    Используются «классические» метрики-синглтоны ragas (экземпляры Metric),
    которым задаются LLM и эмбеддинги на уровне метрики.
    """

    faithfulness = Faithfulness(llm=llm)
    answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    # ContextPrecision может не принимать embeddings напрямую
    try:
        context_precision = ContextPrecision(llm=llm, embeddings=embeddings)
    except TypeError:
        context_precision = ContextPrecision(llm=llm)

    return [faithfulness, answer_relevancy, context_precision]


def evaluate_ragas(dataset, config):
    """Вычисляет метрики Ragas для датасета"""
    llm, embeddings = _build_ragas_components(config)
    metrics = _build_metrics(llm, embeddings)

    return evaluate(dataset, metrics=metrics)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        config = Config.from_env()
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        raise SystemExit(1)

    assistant = RagAssistant(config)
    try:
        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        ground_truths: list[str] = []

        for item in EVALUATION_SET:
            question = item["question"]
            print(f"\nВопрос: {question}")
            try:
                answer, ctx = assistant.answer_with_contexts(question)
            except Exception as exc:  # noqa: BLE001
                print(f"  Ошибка при формировании ответа: {exc}")
                continue
            print(f"  Ответ: {answer[:200]}")
            print(f"  Контекстов: {len(ctx)}")
            questions.append(question)
            answers.append(answer)
            contexts.append(ctx)
            ground_truths.append(item["ground_truth"])

        if not questions:
            print("Не удалось получить ответы ни на один вопрос.")
            raise SystemExit(1)

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        result = evaluate_ragas(dataset, config)

        print("\n=== Результаты оценки Ragas ===")
        print(result)
        print("\nДетализация по вопросам:")
        for (i, score) in enumerate(result.scores):
            print(f"{i + 1}:\t{questions[i]}\n\t{answers[i]}\n\tfaithfulness={score['faithfulness']}; answer_relevancy={score['answer_relevancy']}; context_precision={score['context_precision']}.\n")

    finally:
        assistant.close()


if __name__ == "__main__":
    main()
