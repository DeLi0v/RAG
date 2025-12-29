# run_chat.py

from src.pipline_chat import query_rag

print("Вопрос  (q/й — выход):")
while True:
    q = input("> ").strip()
    if not q:
        continue
    if q.lower() in ["q", "й", "exit", "выход"]:
        break
    ans = query_rag(q)
    print("Ответ: ", ans)
    print("\n---\n")
