import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


def load_documents(directory: str) -> List[Document]:
    docs = []
    path = Path(directory)
    
    for file_path in path.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        title = file_path.stem.replace("_", " ")
        doc = Document(
            page_content=text,
            metadata={
                "source": str(file_path),
                "title": title,
            }
        )
        docs.append(doc)
    
    return docs


def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda x: len(x.split()),
    )
    
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )


def build_index(docs_directory: str, persist_directory: str = "chroma.db"):
    documents = load_documents(docs_directory)
    print(f"Загружено {len(documents)} документов")
    
    chunks = split_documents(documents)
    print(f"Создано {len(chunks)} чанков")
    
    embeddings = create_embeddings()
    print("Эмбеддинги загружены")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    
    print(f"Индекс сохранён в {persist_directory}")
    return vector_store


def search_examples(vector_store: Chroma, query: str, k: int = 3):
    print(f"Кол-во документов в индексе: {vector_store._collection.count()}")
    results = vector_store.similarity_search(query=query, k=k)
    
    print(f"\nЗапрос: {query}")
    print("-" * 50)
    
    for i, doc in enumerate(results, 1):
        print(f"\n[Результат {i}]")
        print(f"Источник: {doc.metadata['source']}")
        print(f"Заголовок: {doc.metadata['title']}")
        print(f"Chunk ID: {doc.metadata['chunk_id']}")
        print(f"Текст (первые 200 символов): {doc.page_content[:200]}...")
        print("-" * 30)
    
    return results


def load_index(persist_directory: str = "chroma.db") -> Chroma:
    embeddings = create_embeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


if __name__ == "__main__":
    import sys
    
    KB_DIR = "knowledge_base"
    CHROMA_DIR = "chroma.db"
    
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        vector_store = build_index(KB_DIR, CHROMA_DIR)
        search_examples(vector_store, "Кто такой Зенон?")
    elif len(sys.argv) > 1 and sys.argv[1] == "--test":
        vector_store = load_index(CHROMA_DIR)
        search_examples(vector_store, "Кто такой Зенон?")
    else:
        print(f"\nИспользуйте флаг --build или --test")
