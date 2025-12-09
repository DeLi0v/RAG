import json
import numpy as np
import lmstudio as lms

from utils import EMBEDDING_MODEL, CHUNKS_JSONL, EMBEDDINGS_OUT, EMBEDDINGS_METADATA_OUT


# инициализация модели эмбеддингов в LM Studio
def embedd(model_name: str = EMBEDDING_MODEL):

    lms.set_sync_api_timeout(600)
    model = lms.embedding_model(model_name)  # замените на актуальное имя модели

    embeddings = []
    metadata = []

    with CHUNKS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record["text"]
            # генерация эмбеддинга
            vector = model.embed(text)
            embeddings.append(vector)
            metadata.append(
                {
                    "chunk_id": record["chunk_id"],
                    "source_file": record["source_file"],
                    "source_path": record["source_path"],
                    "start_char": record["start_char"],
                    "end_char": record["end_char"],
                }
            )

    # сохраняем эмбеддинги и метаданные
    np.savez_compressed(EMBEDDINGS_OUT, embeddings=np.array(embeddings))
    with EMBEDDINGS_METADATA_OUT.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings: {EMBEDDINGS_OUT}")
    print(f"Saved metadata: {EMBEDDINGS_METADATA_OUT}")
