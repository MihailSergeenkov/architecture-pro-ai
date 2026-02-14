from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from llama_cpp import Llama

from pathlib import Path

PERSIST_DIR = Path("chroma.db")
TOP_K = 3
SYSTEM_PROMPT = """Отвечай строго в формате:
Шаг 1: ...
Шаг 2: ...
Итог: <ответ или "Я не знаю">"""

@dataclass
class RetrievedChunk:
    content: str
    source_path: str
    chunk_index: int


class RAGPipeline:
    def __init__(self) -> None:
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=str(PERSIST_DIR),
        )

        self.llm = Llama.from_pretrained(
            repo_id="SpeedyPenguins/Llama-3.2-3B-Instruct-Q4_K_M-GGUF",
            filename="llama-3.2-3b-instruct-q4_k_m.gguf",
            n_ctx=4096,
            n_threads=8,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
        )

    # --- RAG шаги ---

    def retrieve(self, query: str, k: int = TOP_K) -> List[RetrievedChunk]:
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        print(f"Found {len(list(docs_with_scores))} docs") 
        chunks: List[RetrievedChunk] = []
        for d, score in docs_with_scores:
            m = d.metadata
            chunks.append(
                RetrievedChunk(
                    content=d.page_content,
                    source_path=m.get("source_path", ""),
                    chunk_index=m.get("chunk_index", -1),
                )
            )
        return chunks

    def _few_shot_examples(self) -> str:
        return """Пример вопроса: Кто такой Rhade Tan?
Ответ:
Шаг 1: Выполнен поиск в базе знаний.
Шаг 2: Найдена информация о персонаже.
Итог: Rhade Tan — легендарный пилот и контрабандист с планеты Corellia.

Пример вопроса: Кто такой TEST?
Ответ:
Шаг 1: Выполнен поиск в базе знаний.
Шаг 2: Информация не найдена.
Итог: Я не знаю"""

    def build_prompt(self, query: str, chunks: List[RetrievedChunk]) -> str:
        context_blocks = []
        for i, ch in enumerate(chunks, 1):
            context_blocks.append(
                f"[ФРАГМЕНТ {i}]\n{ch.content}"
            )
        context_text = "\n\n".join(context_blocks)

        few_shot = self._few_shot_examples()

        full_prompt = f"""<|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|>

<|start_header_id|>user<|end_header_id|>

Примеры:
{few_shot}

Контекст:
{context_text}

Вопрос: {query}<|eot_id|>

Ответ (строго в формате Шаг 1 / Шаг 2 / Итог)
<|start_header_id|>assistant<|end_header_id|>

"""
        return full_prompt

    def generate_answer(self, prompt: str, max_tokens: int = 1024) -> str:
        out = self.llm(
            prompt,
            max_tokens=max_tokens,
            stop=["<|end_of_text|>"],
            temperature=0.2,
            top_p=0.9,
        )
        text = out["choices"][0]["text"]
        cleaned = text.strip()
        if cleaned.startswith("Шаг"):
            lines = cleaned.split("\n")
            result_lines = []
            for line in lines:
                if line.strip() and not any(word in line for word in ["примечание", "рекомендация", "контекст", "вопрос", "ответ"]):
                    result_lines.append(line)
                elif line.startswith("Шаг") or line.startswith("Итог"):
                    result_lines.append(line)
            cleaned = "\n".join(result_lines[:3])
        return cleaned

    def answer(self, query: str) -> str:
        chunks = self.retrieve(query, k=TOP_K)

        if not chunks:
            return (
                "Шаг 1: Я попытался найти информацию в базе знаний.\n"
                "Шаг 2: Никаких релевантных фрагментов не найдено.\n"
                "Итог: Я не знаю. В текущей базе знаний нет информации по этому вопросу."
            )

        prompt = self.build_prompt(query, chunks)
        answer = self.generate_answer(prompt)
        return answer
