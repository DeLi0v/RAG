import lmstudio as lms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from src.utils import (
    EMBEDDING_MODEL,
    EMBEDDINGS_OUT,
    TOP_K,
    LLM_MODEL,
    CHUNKS_JSONL,
    MAX_CONTEXT_TOKENS,
    SIM_THRESHOLD,
)


def limit_chunks_by_tokens(top_chunks, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Ограничиваем контекст суммарным количеством токенов
    """
    total_tokens = 0
    selected_chunks = []

    for chunk in top_chunks:
        # Эвристика: 1 слово ≈ 1.3 токена
        chunk_tokens = int(len(chunk["text"].split()) * 1.3)
        if total_tokens + chunk_tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens

    return selected_chunks


def select_relevant_chunks(
    sims,
    chunks_data,
    top_k=TOP_K,
    max_tokens=MAX_CONTEXT_TOKENS,
    threshold=SIM_THRESHOLD,
):
    sorted_idx = sims.argsort()[::-1]
    total_tokens = 0
    selected_chunks = []

    for i in sorted_idx:
        if sims[i] < threshold:
            continue
        chunk = chunks_data[i]
        chunk_tokens = int(len(chunk["text"].split()) * 1.3)
        if total_tokens + chunk_tokens > max_tokens:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens
        if len(selected_chunks) >= top_k:
            break

    return selected_chunks


def query_rag(query: str, top_k: int = TOP_K):
    lms.set_sync_api_timeout(600)
    model = lms.embedding_model(EMBEDDING_MODEL)
    model_llm = lms.llm(LLM_MODEL)

    # загружаем эмбеддинги и метаданные
    data = np.load(EMBEDDINGS_OUT)
    embeddings = data["embeddings"]
    chunks_data = [
        json.loads(line) for line in CHUNKS_JSONL.open("r", encoding="utf-8")
    ]

    query_vector = model.embed(query)

    # косинусное сходство
    sims = cosine_similarity([query_vector], embeddings)[0]
    top_chunks = select_relevant_chunks(sims, chunks_data, max_tokens=1000)

    # формируем контекст
    context = "\n\n".join([c["text"] for c in top_chunks])
    prompt = f"Используй контекст:\n{context}\n\nВопрос: {query}"

    print("Промпт:", prompt)

    # генерация ответа
    response = model_llm.respond(
        prompt,
        config={"temperature": 0.0},
        on_prompt_processing_progress=(
            lambda progress: print(f"{round(progress*100)}% complete")
        ),
    )
    return response
