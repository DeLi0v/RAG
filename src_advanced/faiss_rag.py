"""
advanced_rag.py — RAG-система c:
- hnswlib (кроссплатформенный быстрый поиск)
- фильтрацией по метаданным
- управлением токенами
- расширением вариантов вопроса
- LM Studio как LLM + embeddings
"""

import json
import numpy as np
import hnswlib
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
TOKENS_QUESTION = 50
SIM_THRESHOLD = 0.6
TOKEN_COEFF = 1.3


# ----------------------------
# 1. Загрузка hnswlib индекса
# ----------------------------
def load_embeddings_hnsw():
    data = np.load(EMBEDDINGS_OUT)
    embeddings = data["embeddings"].astype("float32")
    dim = embeddings.shape[1]
    total = embeddings.shape[0]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=total, ef_construction=200, M=48)
    index.add_items(embeddings, np.arange(total))
    index.set_ef(128)

    chunks_data = [json.loads(l) for l in CHUNKS_JSONL.open("r", encoding="utf-8")]

    return index, chunks_data


# ----------------------------
# 2. Поиск релевантных чанков
# ----------------------------
def select_relevant_chunks(query_vec, index, chunks_data):
    labels, distances = index.knn_query(query_vec, k=50)
    selected = []
    total_tokens = 0

    for idx, dist in zip(labels[0], distances[0]):
        score = 1 - dist  # cosine → similarity

        if score < SIM_THRESHOLD:
            continue

        chunk = chunks_data[idx]

        if chunk.get("language") and chunk["language"] != "ru":
            continue

        chunk_tokens = int(len(chunk["text"].split()) * TOKEN_COEFF)
        if total_tokens + chunk_tokens > MAX_TOKENS - TOKENS_QUESTION:
            break

        selected.append(chunk)
        total_tokens += chunk_tokens

        if len(selected) >= TOP_K:
            break

    return selected


# ----------------------------
# 3. Построение промпта
# ----------------------------
def build_prompt(question, chunks):
    ctx = ""
    for c in chunks:
        ctx += f"[Источник: {c['source_file']}, символы {c['start_char']}-{c['end_char']}]\n"
        ctx += f"{c['text']}\n\n"

    return f"Используй контекст ниже, обязательно указывая источники.\n\n{ctx}\nВопрос: {question}"


# ----------------------------
# Генерация похожих вопросов
# ----------------------------
def generate_similar_questions(question, model_llm, n=5):
    prompt = (
        f"Сделай {n} переформулировок или похожих вопросов к следующему:\n"
        f'"{question}"\n\n'
        f"Выводи один вопрос на строку, без нумерации."
    )

    resp = model_llm.respond(prompt)
    variants = [l.strip("-• ").strip() for l in resp.split("\n") if l.strip()]
    return variants[:n]


# ----------------------------
# 4. Основная функция RAG
# ----------------------------
def query_rag(question):
    model_embed = lms.embedding_model(EMBEDDING_MODEL_NAME)
    model_llm = lms.llm(LLM_MODEL_NAME)

    index, chunks_data = load_embeddings_hnsw()

    similar = generate_similar_questions(question, model_llm)
    all_questions = [question] + similar

    print("\nСгенерированные варианты запроса:")
    for v in all_questions:
        print(" •", v)

    all_chunks = {}

    for q in all_questions:
        vec = np.array(model_embed.embed(q), dtype="float32").reshape(1, -1)
        found = select_relevant_chunks(vec, index, chunks_data)

        for c in found:
            key = f"{c['source_file']}:{c['chunk_id']}"
            all_chunks[key] = c

    top_chunks = list(all_chunks.values())[:TOP_K]

    prompt = build_prompt(question, top_chunks)

    answer = model_llm.respond(
        prompt,
        config={"temperature": 0.0},
    )

    return answer


# ----------------------------
# CLI режим
# ----------------------------
if __name__ == "__main__":
    while True:
        print("\nВопрос (q — выход):")
        q = input("> ").strip()

        if q.lower() in ["q", "й", "exit", "выход"]:
            break

        print("\nОтвет:\n")
        print(query_rag(q))
        print("\n---\n")
