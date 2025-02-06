import os
from pathlib import Path
from loguru import logger
import shutil
from typing import Callable
# Langchain
from langchain_chroma import Chroma
from langchain_core.documents import Document

# retry
from retry import retry


class VectorStore():
    """
    A class to manage a vector store using Chroma.
    Attributes:
    -----------
    DEFAULT_FN : str
        Default file path for persisting the Chroma database.
    Methods:
    --------
    __init__(embedding_function: Callable | None = None):
        Initializes the VectorStore with a specified embedding function.
    add(documents: list[Document]):
        Adds a list of documents to the vector store.
    clean() -> bool:
        Cleans the vector store by removing the persisted database directory.
    """
    
    DEFAULT_FN = "./chroma_db"
    
    def __init__(self, embedding_function:Callable|None=None):
        """
        Initializes the Ragstore instance.
        Args:
            embedding_function (Callable, optional): A function to generate embeddings. Defaults to None.
        Raises:
            ValueError: If the environment variable 'CHROMA_COLLECTION_NAME' is not found.
        Environment Variables:
            CHROMA_COLLECTION_NAME: The name of the Chroma collection.
            CHROMA_PERSIST_FN: The file path for persisting the Chroma collection. If not found, a default file path is used.
        """
        
        if (cn := os.getenv("CHROMA_COLLECTION_NAME")) is None:
            logger.error("Chroma collection name not found")
            raise ValueError("Chroma collection name not found")
        
        if (fn := os.getenv("CHROMA_PERSIST_FN")) is None:
            logger.error("Chroma persist file not found. Using {DEFAULT_FN}")
            fn = self.DEFAULT_FN
        
        self._db = Chroma(
            collection_name=cn,
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=fn
        )
        
    @retry(exceptions=(Exception,), delay=1, retries=3)
    def add(self,documents:list[Document]):
        """
        Adds a list of documents to the database.
        Args:
            documents (list[Document]): A list of Document objects to be added to the database.
        """
        
        self._db.add_documents(documents)

    def clean(self)->bool:
        """
        Cleans the vector store by removing the directory specified by the 
        environment variable 'CHROMA_PERSIST_FN'.
        Logs a warning before cleaning and logs an info message after 
        successfully clearing the directory or if the directory does not exist.
        Returns:
            bool: True if the directory was successfully removed, False if 
            the directory does not exist.
        """
        logger.warning("Cleaning the vector store")
        persist_path = Path(os.getenv("CHROMA_PERSIST_FN"))
        if persist_path.exists() and persist_path.is_dir():
            shutil.rmtree(persist_path)
            logger.info(f"Chroma database at '{persist_path}' has been cleared.")
            return True
        else:
            logger.info(f"Chroma database directory '{persist_path}' does not exist.")
            return False
