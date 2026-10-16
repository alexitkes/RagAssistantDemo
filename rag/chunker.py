"""
Загрузка документов базы знаний и разбиение на чанки.

Правило деления на чанки: один чанк = один абзац, то есть фрагмент текста,
ограниченный разделителем из двух идущих подряд символов новой строки ("\n\n").
Поддерживаются файлы форматов .txt и .html (из HTML извлекается текст).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PARAGRAPH_SEPARATOR = "\n\n"

_TEXT_EXTENSIONS = {".txt"}
_HTML_EXTENSIONS = {".html", ".htm"}
SUPPORTED_EXTENSIONS = _TEXT_EXTENSIONS | _HTML_EXTENSIONS

# Блочные HTML-элементы: после них вставляется пустая строка, чтобы
# абзацы в итоговом тексте были разделены символами "\n\n".
_BLOCK_TAGS = [
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "div", "section", "article", "blockquote", "table", "tr", "ul", "ol",
]


@dataclass(frozen=True)
class Document:
    """Документ базы знаний: текст и имя файла-источника."""

    source: str
    text: str


def _html_to_text_stdlib(html: str) -> str:
    """Запасной вариант извлечения текста из HTML средствами стандартной библиотеки."""

    from html.parser import HTMLParser

    class _TextExtractor(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self._skip = 0
            self._skip_tags = {"script", "style", "noscript", "template"}

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            tag = tag.lower()
            if tag in self._skip_tags:
                self._skip += 1
            elif tag == "br":
                self.parts.append("\n")
            elif tag in _BLOCK_TAGS:
                self.parts.append("\n\n")

        def handle_endtag(self, tag: str) -> None:
            tag = tag.lower()
            if tag in self._skip_tags and self._skip > 0:
                self._skip -= 1
            elif tag in _BLOCK_TAGS:
                self.parts.append("\n\n")

        def handle_data(self, data: str) -> None:
            if self._skip == 0:
                self.parts.append(data)

    parser = _TextExtractor()
    parser.feed(html)
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)   # убираем хвостовые пробелы перед переносами строк
    text = re.sub(r"\n{3,}", "\n\n", text)   # нормализуем несколько пустых строк подряд
    return text


def html_to_text(html: str) -> str:
    """Извлекает текст из HTML-разметки, между блоками оставляет пустые строки.

    Предпочтительно используется BeautifulSoup (если пакет установлен);
    иначе — парсер стандартной библиотеки.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return _html_to_text_stdlib(html)

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append(soup.new_string("\n\n"))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    text = soup.get_text()
    text = re.sub(r"[ \t]+\n", "\n", text)   # убираем хвостовые пробелы перед переносами строк
    text = re.sub(r"\n{3,}", "\n\n", text)   # нормализуем несколько пустых строк подряд
    return text


def _normalize_newlines(text: str) -> str:
    """Приводит переводы строк к единому виду "\n" (в т.ч. Windows "\r\n")."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def load_documents(data_dir: Path) -> list[Document]:
    """
    Считывает все файлы .txt и .html из data_dir (включая подкаталоги).

    Возвращает список документов. Если в каталоге нет файлов поддерживаемых
    форматов, список будет пустым.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Каталог с базой знаний не найден: {data_dir}")

    documents: list[Document] = []
    for path in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        content = html_to_text(raw_text) if suffix in _HTML_EXTENSIONS else raw_text
        documents.append(
            Document(source=str(path.relative_to(data_dir)), text=_normalize_newlines(content))
        )
    return documents


def chunk_by_paragraphs(text: str) -> list[str]:
    """
    Разбивает текст на чанки-абзацы по разделителю "\n\n".

    Пустые абзацы отбрасываются, текст каждого абзаца обрезается
    (strip) и нормализуется.
    """
    paragraphs = (p.strip() for p in text.split(PARAGRAPH_SEPARATOR))
    return [p for p in paragraphs if p]
