"""
# src/documents_loader.py

Полный загрузчик и конвертер документов для RAG.
- Поддерживает docx, pdf, txt, csv, xls, xlsx и таблицы через docling.
- Нормализует Markdown текст.
- Сохраняет расширенные метаданные.
"""

from docling.document_converter import DocumentConverter
from pathlib import Path
import datetime
from utils import SOURCE_DIR, OUT_DIR, logger
from tqdm import tqdm
import chardet
import re
import json

# -------------------------
# Вспомогательные функции
# -------------------------


def normalize_markdown(text: str) -> str:
    """
    Нормализует текст для Markdown:
    - удаляет лишние пробелы и табуляции
    - заменяет >2 переносов строк на 2
    - сохраняет Markdown-символы (#, *, -, |)
    """
    # Убираем ненужные пробелы в начале/конце строк
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)

    # Убираем двойные-тройные пробелы
    text = re.sub(r"[ \t]+", " ", text)

    # Заменяем 3+ переносов строк на 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def read_text_auto_encoding(path: Path) -> str:
    raw = path.read_bytes()  # читаем как байты
    result = chardet.detect(raw)
    encoding = result["encoding"] or "utf-8"
    print(f"Detected {path} encoding: {encoding}")
    return raw.decode(encoding, errors="ignore")  # игнорируем неподдерживаемые символы


def save_text(path: Path, text: str):
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def save_metadata(path: Path, meta: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# -------------------------
# Обработка файлов
# -------------------------


def process_file(src_path: Path, out_dir):
    logger.info(f"Обработка: {src_path}")
    converter = DocumentConverter()
    doc = None
    try:
        if src_path.suffix.lower() == ".txt":
            text = read_text_auto_encoding(src_path)
        else:
            doc = converter.convert(str(src_path)).document
            text = doc.export_to_markdown()
    except Exception as exp:
        logger.exception(f"Ошибка конвертации {src_path}: {exp}")
        return None

    # нормализация Markdown
    text = normalize_markdown(text)

    # сохранение текста
    out_text_path = out_dir / (src_path.stem + ".txt")
    save_text(out_text_path, text)

    # метаданные
    out_meta_path = out_dir / (src_path.stem + ".json")
    meta = meta = {
        "source_path": str(src_path),
        "filename": src_path.name,
        "file_type": src_path.suffix.lower(),
        "saved_text": str(out_text_path),
        "saved_meta": str(out_meta_path),
        "n_chars": len(text),
        "n_words": len(text.split()),
        "converted_at": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        "author": getattr(doc, "author", None),
        "created_at": getattr(doc, "created_at", None),
        "language": getattr(doc, "language", None),
        "n_tables": (
            len(doc.tables) if hasattr(doc, "tables") else 0
        ),  # количество таблиц
    }

    save_metadata(out_meta_path, meta)

    return meta


# -------------------------
# Главная функция
# -------------------------


def load_documents():
    files = list(SOURCE_DIR.rglob("*.*"))

    for f in tqdm(files, desc="Files"):
        process_file(f, OUT_DIR)


if __name__ == "__main__":
    load_documents()
