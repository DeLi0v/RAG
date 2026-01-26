import json
import numpy as np
import hnswlib
import lmstudio as lms
from pathlib import Path
import datetime
from src.utils import (
    CHUNKS_JSONL,
    EMBEDDINGS_OUT,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
    MAX_CONTEXT_TOKENS,
    TOKENS_QUESTION,
    SIM_THRESHOLD,
    TOKEN_COEFF,
    logger,
)


# ----------------------------
# 1. Загрузка hnswlib индекса
# ----------------------------
def load_embeddings_hnsw():
    logger.info("Начало загрузки HNSW индекса...")
    try:
        data = np.load(EMBEDDINGS_OUT)
        embeddings = data["embeddings"].astype("float32")
        dim = embeddings.shape[1]
        total = embeddings.shape[0]

        logger.debug(f"Размерность эмбеддингов: {dim}, всего элементов: {total}")

        # Инициализация HNSW индекса
        logger.info("Инициализация HNSW индекса...")
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=total, ef_construction=400, M=64)  # увеличили
        index.add_items(embeddings, np.arange(total))
        index.set_ef(256)  # увеличили

        logger.debug("Загрузка данных чанков...")
        chunks_data = [json.loads(l) for l in CHUNKS_JSONL.open("r", encoding="utf-8")]

        logger.info(
            f"HNSW индекс успешно загружен: {total} эмбеддингов, {len(chunks_data)} чанков"
        )
        return index, chunks_data
    except FileNotFoundError as e:
        logger.error(f"Файл не найден: {e}")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке HNSW индекса: {e}")
        raise


# ----------------------------
# 2. Поиск релевантных чанков
# ----------------------------
def select_relevant_chunks(query_vec, index, chunks_data):
    logger.info("Поиск релевантных чанков...")
    k = min(50, len(chunks_data))  # чтобы не запрашивать больше, чем есть

    logger.debug(f"Ищем {k} ближайших соседей из {len(chunks_data)} доступных")
    labels, distances = index.knn_query(query_vec, k=k)

    selected = []
    total_tokens = 0
    skipped_low_score = 0
    skipped_language = 0

    logger.debug(f"Начало обработки {len(labels[0])} кандидатов")
    for idx, dist in zip(labels[0], distances[0]):
        score = 1 - dist  # cosine → similarity

        if score < SIM_THRESHOLD:
            skipped_low_score += 1
            logger.debug(
                f"Чанк {idx} пропущен: сходство {score:.3f} < порога {SIM_THRESHOLD}"
            )
            continue

        chunk = chunks_data[idx]

        if chunk.get("language") and chunk["language"] != "ru":
            skipped_language += 1
            logger.debug(
                f"Чанк {idx} пропущен: язык {chunk.get('language')} (требуется 'ru')"
            )
            continue

        chunk_tokens = int(len(chunk["text"].split()) * TOKEN_COEFF)
        if total_tokens + chunk_tokens > MAX_CONTEXT_TOKENS - TOKENS_QUESTION:
            logger.debug(
                f"Превышен лимит токенов: текущие {total_tokens} + {chunk_tokens} > {MAX_CONTEXT_TOKENS - TOKENS_QUESTION}"
            )
            break

        selected.append(chunk)
        total_tokens += chunk_tokens

        logger.debug(
            f"Чанк {idx} добавлен: файл={chunk['source_file']}, токены={chunk_tokens}, сумма={total_tokens}"
        )

        if len(selected) >= TOP_K:
            logger.debug(f"Достигнут максимальный TOP_K={TOP_K}")
            break

    logger.info(f"Отобрано чанков: {len(selected)} из {len(labels[0])} кандидатов")
    logger.debug(
        f"Статистика: пропущено по сходству={skipped_low_score}, по языку={skipped_language}"
    )
    logger.debug(f"Общий объем токенов: {total_tokens}")

    return selected


# ----------------------------
# 3. Построение промпта
# ----------------------------
def build_prompt(question, chunks):
    logger.info("Формирование промпта...")
    ctx = ""
    for c in chunks:
        ctx += f"[Источник: {c['source_file']}, символы {c['start_char']}-{c['end_char']}]\n"
        ctx += f"{c['text']}\n\n"
        logger.debug(f"Добавлен чанк {i+1}: {c['source_file']}")

    prompt = f"Используй контекст ниже, обязательно указывая источники, откуда была взята информация.\n\n{ctx}\nВопрос: {question}"

    logger.debug(f"Длина промпта: {len(prompt)} символов, {len(prompt.split())} слов")
    logger.info(f"Промпт сформирован с {len(chunks)} чанками")

    return prompt


# ----------------------------
# Генерация похожих вопросов
# ----------------------------
def generate_similar_questions(question, model_llm, n=5):
    logger.info(f"Генерация {n} похожих вопросов...")
    prompt = (
        f"Сделай {n} переформулировок или похожих вопросов к следующему:\n"
        f'"{question}"\n\n'
        f"Выводи один вопрос на строку, без нумерации."
    )

    logger.debug(f"Промпт для генерации вопросов: {prompt[:100]}...")

    try:
        resp = model_llm.respond(
            prompt,
            on_prompt_processing_progress=(
                lambda progress: logger.debug(
                    f"Прогресс генерации вопросов: {round(progress*100)}%"
                )
            ),
        )
        variants = [
            l.strip("-• ").strip() for l in resp.content.split("\n") if l.strip()
        ]
        valid_variants = [
            v for v in variants if v and len(v) > 5
        ]  # Фильтр коротких строк

        logger.info(f"Сгенерировано похожих вопросов: {len(valid_variants)}")
        for i, v in enumerate(valid_variants[:3]):  # Логируем первые 3
            logger.debug(f"Вариант {i+1}: {v[:50]}...")

        return valid_variants[:n]

    except Exception as e:
        logger.error(f"Ошибка при генерации похожих вопросов: {e}")
        return []


# ----------------------------
# 4. Основная функция RAG
# ----------------------------
def query_rag(question):

    logger.info("=" * 60)
    logger.info(f"ЗАПУСК RAG ЗАПРОСА: {question}")
    logger.info("=" * 60)

    start_time = datetime.datetime.now()

    try:
        logger.info("Инициализация моделей...")
        model_embed = lms.embedding_model(EMBEDDING_MODEL)
        model_llm = lms.llm(LLM_MODEL)
        lms.set_sync_api_timeout(600)
        logger.info("Модели инициализированы")

        # Загрузка индекса и данных
        logger.info("Загрузка данных и индекса...")
        index, chunks_data = load_embeddings_hnsw()

        # Генерация вариантов вопросов
        logger.info("Генерация вариантов запросов...")
        similar = generate_similar_questions(question, model_llm)
        all_questions = [question] + similar

        logger.info(f"Всего вопросов для обработки: {len(all_questions)}")
        logger.debug("Список всех вопросов:")
        for i, q in enumerate(all_questions):
            logger.debug(f"  {i+1}. {q}")

        # Поиск чанков для всех вопросов
        all_chunks = {}
        logger.info("Поиск релевантных чанков для всех вопросов...")

        for i, q in enumerate(all_questions):
            logger.info(f"Обработка вопроса {i+1}/{len(all_questions)}: {q[:50]}...")
            vec = np.array(model_embed.embed(q), dtype="float32").reshape(1, -1)
            found = select_relevant_chunks(vec, index, chunks_data)

            logger.debug(f"Найдено чанков для вопроса {i+1}: {len(found)}")

            for c in found:
                key = f"{c['source_file']}:{c['chunk_id']}"
                all_chunks[key] = c

        # Выбор топ чанков
        top_chunks = list(all_chunks.values())[:TOP_K]
        logger.info(
            f"Итоговая статистика: уникальных чанков={len(all_chunks)}, отобрано={len(top_chunks)}"
        )

        # Формирование промпта
        logger.info("Формирование финального промпта...")
        prompt = build_prompt(question, top_chunks)

        # Генерация ответа
        logger.info("Начало генерации ответа LLM...")
        logger.debug(f"Параметры генерации: temperature=0.0")

        answer = model_llm.respond(
            prompt,
            config={"temperature": 0.0},
            on_prompt_processing_progress=(
                lambda progress: logger.info(
                    f"Прогресс генерации ответа: {round(progress*100)}%"
                )
            ),
        )

        # Завершение
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        logger.info(f"ОТВЕТ СГЕНЕРИРОВАН за {duration:.2f} секунд")
        logger.info(f"Длина ответа: {len(answer.content)} символов")
        logger.debug(f"Первые 200 символов ответа: {answer.content[:200]}...")

        if len(answer.content) < 50:
            logger.warning("Ответ очень короткий, возможно проблема с генерацией")

        logger.info("=" * 60)
        logger.info("RAG ЗАПРОС УСПЕШНО ЗАВЕРШЕН")
        logger.info("=" * 60)

        return answer

    except Exception as e:
        logger.error("КРИТИЧЕСКАЯ ОШИБКА В RAG ПРОЦЕССЕ")
        logger.error(f"Ошибка: {str(e)}", exc_info=True)
        logger.error(
            f"Время выполнения до ошибки: {(datetime.datetime.now() - start_time).total_seconds():.2f} сек"
        )

        # Создаем fallback ответ
        fallback_answer = type(
            "obj",
            (object,),
            {"content": f"Извините, произошла ошибка при обработке запроса: {str(e)}"},
        )
        return fallback_answer
