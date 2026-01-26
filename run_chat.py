# run_chat.py

from src.pipline_chat import query_rag
from src.utils import logger

print("Вопрос  (q/й — выход):")
while True:
    q = input("> ").strip()
    if not q:
        continue
    if q.lower() in ["q", "й", "exit", "выход"]:
        break
    logger.info("Вопрос получен. Начало генерации ответа...")
    ans = query_rag(q)
    logger.info("Ответ сгенерирован...")
    print("Ответ: ", ans)
    print("\n---\n")
