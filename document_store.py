from loguru import logger

# REDIS
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
# retry
from retry import retry


class DocumentStore():
    """
    A class to interact with a Redis-based document store.
    Attributes:
    -----------
    _dbc : redis.Redis
        The Redis connection object.
    Methods:
    --------
    __init__(host: str = 'localhost', port: int = 6379, username: str | None = None, password: str | None = None):
        Initializes the DocumentStore with a Redis connection.
    is_in(key: str) -> bool:
        Checks if a given key exists in the document store.
    add(documents: list[dict]):
        Adds a list of documents to the document store.
    get(key: str) -> dict:
        Retrieves a document from the document store by key.
    clean():
        Cleans the entire document store by flushing the database.
    """
    
    def __init__(self, host:str='localhost', port:int=6379, username:str|None=None, password:str|None=None,
                 idx_name:str="RAGGAE_INDEX", idx_prefix:str="RAGGAE"):
        """
        Initializes a connection to a Redis database.
        Args:
            host (str): The hostname of the Redis server. Defaults to 'localhost'.
            port (int): The port number on which the Redis server is listening. Defaults to 6379.
            username (str | None): The username for Redis authentication. Defaults to None.
            password (str | None): The password for Redis authentication. Defaults to None.
        Raises:
            redis.ConnectionError: If there is an error connecting to the Redis server.
            Exception: If there is any other error in establishing the Redis connector.
        """
        self._INDEX = idx_name
        self._PREFIX = idx_prefix
        
        try:
            self._dbc = redis.Redis(host=host, port=port, decode_responses=True, username=username, password=password)
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise e
        except Exception as e:
            logger.error(f"Error in establishing a Redis connector: {e}")
            raise e
        # If not present create an index for metadata search
        self.create_index()
        
    def is_in(self, key:str)->bool:
        """
        Check if a given key exists in the database.
        Args:
            key (str): The key to check for existence in the database.
        Returns:
            bool: True if the key exists, False otherwise.
        """
        # Check if the key exists in the database
        if self._dbc.exists(key):
            return True
        if self._dbc.ft(self._INDEX).search(f"{key}").total > 0:
            return True
        return False

    @retry(exceptions=(Exception,), delay=1, retries=3)    
    def add(self, documents:list[dict]):
        """
        Adds a list of documents to the database.
        Args:
            documents (list[dict]): A list of dictionaries where each dictionary represents a document.
                Each dictionary should have the following structure:
                {
                    'document_id': <unique identifier for the document>,
                    'document': <dictionary containing the document data>
                }
        Returns:
            None
        """
        
        pipe = self._dbc.pipeline()
        for document in documents:
            content_id = document['document_id']
            pipe.hset(content_id, mapping=document['document'])
        pipe.execute()

    def get(self, key:str)->dict:
        """
        Retrieve a value from the database using the provided key.
        Args:
            key (str): The key to look up in the database.
        Returns:
            dict: The value associated with the key in the database.
        """
        return self._dbc.get(key)
    
    def clean(self):
        """
        Cleans the document store by flushing the database.
        This method logs a warning message indicating that the document store
        is being cleaned, and then proceeds to flush the database using the
        `_dbc.flushdb()` method.
        """
        
        logger.warning("Cleaning the document store")
        self._dbc.flushdb()

    def index_exists(self, index_name):
        """
        Checks whether a RediSearch index exists in the Redis database.
        
        :param index_name: The name of the index to check.
        :return: True if the index exists, False otherwise.
        """
        try:
            # Attempt to get index info
            self._dbc.ft(index_name).info()
            return True
        except Exception as e:
            # Exception indicates the index doesn't exist or another issue
            return False

    def create_index(self, force:bool=False):
        """
        Sets up an index for full-text search with metadata fields in Redis.
        """
        # Drop the index if it already exists
        if self.index_exists(self._INDEX):
            if not force:
                return
            try:
                self._dbc.ft(self._INDEX).dropindex(delete_documents=False)
            except Exception as e:
                logger.error(f"Index does not exist or cannot be dropped: {e}")

        # Define fields for the index
        schema = (
            TextField("doc-id"),          # Document hash
            TextField("chunk-id"),        # Element ID
            TextField("type"),            # Element type
            TextField("source"),          # File name
            NumericField("page_number"),  # Page number
            TagField("languages"),        # Languages (comma-separated tags)
            TextField("content"),         # Full-text content
        )

        # Create the index
        self._dbc.ft(self._INDEX).create_index(
            schema,
            definition=IndexDefinition(prefix=[f"{self._PREFIX}:"], index_type=IndexType.HASH)
        )
        logger.info(f"Index '{self._INDEX}' created successfully.")

    def full_text_search(self, query_string:str, filters=None):
        """
        Performs a full-text search on the index, with optional filters.
        
        :param query_string: The text to search for.
        :param filters: A dictionary of metadata filters (e.g., {"type": "text", "languages": "en"}).
        :return: List of matching documents with metadata.
        """
        index_name = self._INDEX
        query = Query(query_string)

        # Apply filters if specified
        if filters:
            for field, value in filters.items():
                if field == "languages":  # For TagField, use curly braces
                    query = f"@{field}:{{{value}}}"
                elif isinstance(value, str):
                    query = value.strip()
                elif isinstance(value, list):
                    query = '('+'|'.join([f"'{val.replace('-',' ').strip()}'" for val in value])+')'

        # Execute the search
        results = self._dbc.ft(index_name).search(query)

        # Format results
        documents = []
        for doc in results.docs:
            documents.append({
                "doc-id": doc['doc-id'],
                "chunk-id": doc['chunk-id'],
                "type": doc.type,
                "source": doc.source,
                "page_number": doc.page_number,
                "languages": doc.languages.split(",") if doc.languages else [],
                "content": doc.content
            })

        return documents
    
