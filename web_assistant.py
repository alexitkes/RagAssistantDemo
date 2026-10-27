#!/usr/bin/env python3
"""
RAG-ассистент: веб-приложение на Flask.

Скрипт:
  1. запускает локальный HTTP-сервер Flask (по умолчанию http://127.0.0.1:5000);
  2. главная страница содержит форму для ввода вопроса;
  3. после отправки формы вопрос передаётся в RagAssistant.ask_question,
     а в качестве идентификатора пользователя передаётся строка "ip_{IP}",
     где IP — адрес пользователя (request.remote_addr). Это даёт каждому
     пользователю отдельный кэш ответов и отдельный лимит частоты вопросов;
  4. полученный ответ выводится на странице вместе с формой для следующего вопроса.

Конфигурация читается классом Config из config.py (переменные — в файле .env,
шаблон — .env.example).

Пример запуска:
    python web_assistant.py

Примечания:
  * разработческий сервер Flask запускается однопоточным (threaded=False):
    RagAssistant держит одно соединение с SQLite и общую историю диалога,
    поэтому запросы обрабатываются последовательно;
  * если публиковать приложение за reverse proxy, адрес пользователя следует
    брать из заголовка X-Forwarded-For (например, через
    werkzeug.middleware.proxy_fix.ProxyFix), иначе request.remote_addr вернёт
    адрес прокси, а не пользователя.
"""
from __future__ import annotations

import atexit
import logging
import threading

from flask import Flask, render_template_string, request

from config import Config, ConfigError
from rag.assistant import RagAssistant
from rag.rate_limit import RateLimitError

logger = logging.getLogger(__name__)

HOST = "127.0.0.1"  # HTTP-сервер слушает только локальные подключения
PORT = 5000

app = Flask(__name__)

# Ассистент создаётся лениво (при первом вопросе), чтобы модуль можно было
# импортировать без настроенного .env, а ошибки конфигурации увидеть сразу
# при запуске (см. main(): ассистент создаётся до старта сервера).
_assistant: RagAssistant | None = None
_assistant_lock = threading.Lock()

PAGE_TEMPLATE = """\
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG-ассистент</title>
  <style>
    :root { color-scheme: light dark; }
    body {
      max-width: 760px;
      margin: 2rem auto;
      padding: 0 1rem;
      font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
      line-height: 1.55;
    }
    h1 { margin-bottom: 0.25rem; }
    .subtitle { color: #667; margin-top: 0; }
    .answer, .error {
      border-radius: 8px;
      padding: 1rem 1.25rem;
      margin: 1.5rem 0;
    }
    .answer { background: #eef6ff; border: 1px solid #bcd9f5; }
    .error { background: #fff0f0; border: 1px solid #f0b9b9; color: #8a1f1f; }
    .answer h2 { margin-top: 0; }
    .answer pre { white-space: pre-wrap; font-family: inherit; margin: 0.25rem 0 0; }
    label { font-weight: 600; display: block; margin: 1rem 0 0.35rem; }
    textarea {
      width: 100%;
      box-sizing: border-box;
      padding: 0.6rem 0.75rem;
      font: inherit;
      border: 1px solid #ccc;
      border-radius: 6px;
    }
    button {
      margin-top: 0.75rem;
      padding: 0.55rem 1.4rem;
      font: inherit;
      font-weight: 600;
      color: #fff;
      background: #1a6fb5;
      border: 0;
      border-radius: 6px;
      cursor: pointer;
    }
    button:hover { background: #155c96; }
  </style>
</head>
<body>
  <h1>RAG-ассистент</h1>
  <p class="subtitle">Задайте вопрос по базе знаний — ответ будет сформирован на основе найденных документов.</p>

  {% if error %}
  <div class="error">{{ error|e }}</div>
  {% endif %}

  {% if answer is not none %}
  <div class="answer">
    <h2>Ответ</h2>
    <p class="subtitle" style="margin-top: 0"><strong>Вопрос:</strong> {{ question|e }}</p>
    <pre>{{ answer|e }}</pre>
  </div>
  {% endif %}

  <form method="post" autocomplete="off">
    <label for="question">Ваш вопрос:</label>
    <textarea id="question" name="question" rows="5" required autofocus
              placeholder="Например: Какие требования предъявляются к ...?"></textarea>
    <button type="submit">Задать вопрос</button>
  </form>
</body>
</html>
"""


def get_assistant() -> RagAssistant:
    """Возвращает RagAssistant, создавая его при первом обращении."""
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                try:
                    _assistant = RagAssistant(Config.from_env())
                except ConfigError as exc:
                    raise RuntimeError(f"Ошибка конфигурации: {exc}") from exc
    return _assistant


def _close_assistant() -> None:
    """Закрывает соединения ассистента при завершении процесса."""
    global _assistant
    if _assistant is not None:
        _assistant.close()
        _assistant = None


atexit.register(_close_assistant)


@app.route("/", methods=["GET", "POST"])
def index() -> str:
    """Главная страница: форма вопроса, а после отправки — ответ и новая форма."""
    question = ""
    answer: str | None = None
    error: str | None = None

    if request.method == "POST":
        question = (request.form.get("question") or "").strip()
        if not question:
            error = "Пожалуйста, введите вопрос."
        else:
            # Идентификатор пользователя: отдельный кэш и лимиты для каждого IP
            uid = f"ip_{request.remote_addr or 'unknown'}"
            logger.info("[%s] Вопрос: '%s'", uid, question[:120])
            try:
                assistant = get_assistant()
            except RuntimeError as exc:
                logger.exception("Не удалось создать ассистента")
                error = str(exc)
            else:
                try:
                    answer = assistant.ask_question(question, uid)
                    logger.info("[%s] Ответ получен (%d симв.)", uid, len(answer))
                except RateLimitError as exc:
                    logger.warning("[%s] Превышен лимит вопросов: %s", uid, exc)
                    error = str(exc)
                except Exception:  # noqa: BLE001
                    logger.exception("[%s] Ошибка при формировании ответа", uid)
                    error = "Не удалось сформировать ответ. Попробуйте позже."

    return render_template_string(PAGE_TEMPLATE, question=question, answer=answer, error=error)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Проверяем конфигурацию и создаём ассистента до запуска сервера,
    # чтобы ошибки настроек были видны сразу, а не при первом вопросе.
    try:
        get_assistant()
    except RuntimeError as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1)

    print(f"Веб-ассистент запущен: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False, threaded=False)


if __name__ == "__main__":
    main()
