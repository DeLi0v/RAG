"""
advanced_rag.py
RAG с:
- FAISS для быстрого поиска
- фильтрацией по метаданным
- управлением токенами
- добавлением источников для LLM
"""

import json
import numpy as np
import faiss
import lmstudio as lms
from pathlib import Path

# ----------------------------
# Настройки
# ----------------------------
CHUNKS_JSONL = Path("chunks.jsonl")
EMBEDDINGS_OUT = Path("embeddings.npz")
EMBEDDING_MODEL_NAME = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL_NAME = "mistral-7b-instruct"
TOP_K = 10
MAX_TOKENS = 1024
TOKENS_QUESTION = 50  # оценка токенов вопроса
SIM_THRESHOLD = 0.6  # минимальное сходство
TOKEN_COEFF = 1.3  # 1 слово ≈ 1.3 токена


# ----------------------------
# 1️⃣ Загрузка эмбеддингов и создание FAISS индекса
# ----------------------------
def load_embeddings_faiss():
    # эмбеддинги
    lms.set_sync_api_timeout(600)
    data = np.load(EMBEDDINGS_OUT)
    embeddings = data["embeddings"].astype("float32")
    embedding_dim = embeddings.shape[1]

    # FAISS индекс
    index = faiss.IndexFlatIP(embedding_dim)
    faiss.normalize_L2(embeddings)  # для косинусного сходства
    index.add(embeddings)

    # чанки
    chunks_data = [
        json.loads(line) for line in CHUNKS_JSONL.open("r", encoding="utf-8")
    ]

    return index, embeddings, chunks_data


# ----------------------------
# 2️⃣ Выборка топ-K чанков с фильтрацией
# ----------------------------
def select_relevant_chunks(
    query_vector,
    index,
    chunks_data,
    top_k=TOP_K,
    max_tokens=MAX_TOKENS - TOKENS_QUESTION,
    threshold=SIM_THRESHOLD,
):
    # нормализуем вектор для косинусного поиска
    faiss.normalize_L2(query_vector)

    # поиск top 50, потом отфильтруем по порогу
    D, I = index.search(query_vector, 50)
    selected_chunks = []
    total_tokens = 0

    for score, idx in zip(D[0], I[0]):
        if score < threshold:
            continue
        chunk = chunks_data[idx]

        # фильтрация по метаданным (пример: язык русский)
        if chunk.get("language") and chunk["language"] != "ru":
            continue

        chunk_tokens = int(len(chunk["text"].split()) * TOKEN_COEFF)
        if total_tokens + chunk_tokens > max_tokens:
            break

        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
        if len(selected_chunks) >= top_k:
            break

    return selected_chunks


# ----------------------------
# 3️⃣ Формирование контекста для LLM
# ----------------------------
def build_prompt(question, chunks):
    context = ""
    for c in chunks:
        context += f"[Источник: {c['source_file']}, символы {c['start_char']}-{c['end_char']}]\n{c['text']}\n\n"

    prompt = (
        f"Используй контекст (с указанием источника):\n{context}\n\nВопрос: {question}"
    )
    return prompt


# ----------------------------
# 4️⃣ Основная функция запроса
# ----------------------------
def query_rag(question: str):
    # инициализация моделей
    model_embed = lms.embedding_model(EMBEDDING_MODEL_NAME)
    model_llm = lms.llm(LLM_MODEL_NAME)

    # загрузка индекса и чанков
    index, embeddings, chunks_data = load_embeddings_faiss()

    # эмбеддинг запроса
    raw_vec = model_embed.embed(question)
    query_vec = np.array(raw_vec, dtype="float32").reshape(1, -1)

    # выбор релевантных чанков
    top_chunks = select_relevant_chunks(query_vec, index, chunks_data)

    # формируем промпт
    prompt = build_prompt(question, top_chunks)

    # генерация ответа
    response = model_llm.respond(
        prompt,
        config={"temperature": 0.0},
        on_prompt_processing_progress=(
            lambda progress: print(f"{round(progress*100)}% complete")
        ),
    )
    return response


# ----------------------------
# Пример запуска
# ----------------------------
if __name__ == "__main__":
    while True:
        print("Вопрос (выйти: q или й):")
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in ["q", "й", "exit", "выход"]:
            break
        ans = query_rag(q)
        print("Ответ: ", ans)
        print("\n---\n")
