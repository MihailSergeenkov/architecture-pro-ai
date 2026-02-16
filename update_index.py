import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


METADATA_FILE = "indexed_files.json"


def load_indexed_files(metadata_path: str = METADATA_FILE) -> Dict[str, dict]:
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_indexed_files(indexed_files: Dict[str, dict], metadata_path: str = METADATA_FILE):
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(indexed_files, f, indent=2, ensure_ascii=False)


def get_file_info(file_path: Path) -> dict:
    stat = file_path.stat()
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }


def load_documents(directory: str) -> List[Document]:
    docs = []
    path = Path(directory)
    
    if not path.exists():
        logger.warning(f"Директория {directory} не существует")
        return docs
    
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


def get_file_key(file_path: str) -> str:
    return str(Path(file_path).resolve())


def update_index(
    docs_directory: str,
    persist_directory: str = "chroma.db",
    metadata_path: str = METADATA_FILE,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Tuple[int, int, int, str]:
    start_time = datetime.now()
    errors = []
    
    logger.info(f"Начало обновления индекса: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    indexed_files = load_indexed_files(metadata_path)
    current_files: Dict[str, dict] = {}
    
    docs_path = Path(docs_directory)
    if not docs_path.exists():
        errors.append(f"Директория {docs_directory} не существует")
        logger.error(errors[-1])
        return 0, 0, 0, "; ".join(errors)
    
    for file_path in docs_path.glob("*.txt"):
        file_key = get_file_key(str(file_path))
        current_files[file_key] = get_file_info(file_path)
    
    new_or_modified: List[Path] = []
    deleted: Set[str] = set(indexed_files.keys()) - set(current_files.keys())
    
    for file_key in current_files:
        path_obj = Path(file_key)
        
        if file_key not in indexed_files:
            new_or_modified.append(path_obj)
            logger.info(f"Новый файл: {path_obj.name}")
        elif current_files[file_key]["mtime"] != indexed_files[file_key].get("mtime"):
            new_or_modified.append(path_obj)
            logger.info(f"Изменённый файл: {path_obj.name}")
    
    if deleted:
        logger.info(f"Удалённые файлы: {len(deleted)}")
        for file_key in deleted:
            logger.info(f"  - {file_key}")
    
    files_added = len(new_or_modified)
    
    if files_added == 0 and len(deleted) == 0:
        logger.info("Нет новых или изменённых документов")
        embeddings = create_embeddings()
        vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        index_size = vector_store._collection.count()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        final_msg = (
            f"index updated at {end_time.strftime('%Y-%m-%d')}, "
            f"{files_added} files added, "
            f"0 new chunks, "
            f"index size: {index_size}, "
            f"{len(errors)} errors"
        )
        print(f"\n{final_msg}")
        logger.info(f"Длительность: {duration:.2f} сек")
        
        return files_added, 0, index_size, "; ".join(errors) if errors else ""
    
    new_documents: List[Document] = []
    for file_path in new_or_modified:
        try:
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
            new_documents.append(doc)
            logger.info(f"Загружен документ: {file_path.name}")
        except Exception as e:
            error_msg = f"Ошибка при чтении {file_path}: {e}"
            logger.error(error_msg)
            errors.append(error_msg)
    
    new_chunks = split_documents(new_documents, chunk_size, chunk_overlap)
    chunks_added = len(new_chunks)
    
    logger.info(f"Создано {chunks_added} новых чанков из {len(new_documents)} документов")
    
    vector_store = None
    
    try:
        embeddings = create_embeddings()
        
        if deleted:
            for file_key in deleted:
                source_filter = file_key
                try:
                    vector_store = Chroma(
                        persist_directory=persist_directory,
                        embedding_function=embeddings
                    )
                    existing_docs = vector_store.get(where={"source": source_filter})
                    if existing_docs and existing_docs.get("ids"):
                        vector_store.delete(ids=existing_docs["ids"])
                        logger.info(f"Удалены чанки для: {source_filter}")
                except Exception as e:
                    error_msg = f"Ошибка при удалении чанков для {source_filter}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        if new_chunks:
            vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            vector_store.add_documents(documents=new_chunks)
            logger.info(f"Добавлено {chunks_added} чанков в индекс")
        
        if vector_store is not None:
            index_size = vector_store._collection.count()
        else:
            index_size = 0
        
    except Exception as e:
        error_msg = f"Ошибка при обновлении индекса: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        index_size = 0
    
    for file_path in new_or_modified:
        file_key = get_file_key(str(file_path))
        indexed_files[file_key] = get_file_info(file_path)
    
    for file_key in deleted:
        del indexed_files[file_key]
    
    save_indexed_files(indexed_files, metadata_path)
    logger.info(f"Метаданные индекса сохранены в {metadata_path}")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    final_msg = (
        f"index updated at {end_time.strftime('%Y-%m-%d')}, "
        f"{files_added} files added, "
        f"{chunks_added} new chunks, "
        f"index size: {index_size}, "
        f"{len(errors)} errors"
    )
    print(f"\n{final_msg}")
    logger.info(f"Длительность выполнения: {duration:.2f} сек")
    
    return files_added, chunks_added, index_size, "; ".join(errors) if errors else ""


if __name__ == "__main__":    
    KB_DIR = "docs"
    CHROMA_DIR = "chroma.db"
    
    update_index(KB_DIR, CHROMA_DIR)
