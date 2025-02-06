import os
import zlib
from pathlib import Path
from typing import Tuple, Callable
from loguru import logger
import tiktoken
import shutil
# Hash
import hashlib
import base64
# Unstructured
from unstructured.partition.auto import partition
from unstructured.documents.elements import Text, Image, Title, Table, FigureCaption, NarrativeText
# Concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
# Langchain
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain_community.callbacks.manager import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
# REDIS
import redis
from redis.commands.search.field import TextField, NumericField, TagField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class RAGStore():
    """
    A class to handle the storage and processing of documents using a Retrieval-Augmented Generation (RAG) approach.
    Attributes:
        RAG_PREFIX (str): Prefix used for document identifiers.
        _debug (bool): Flag to enable or disable debug mode.
        _llm_type (str): Type of the language model to use.
        _docstore (DocumentStore): Instance of the DocumentStore class.
        _vectorstore (VectorStore): Instance of the VectorStore class for indexing summaries and their embeddings.
    Methods:
        __init__(llm: str = 'OpenAI', debug: bool = False):
            Initializes the RAGStore instance with the specified language model and debug mode.
        get_summariser():
            Returns a summariser instance based on the specified language model.
        add_file(fn: str | Path, store: bool = True, overwrite: bool = False) -> Tuple[str, dict]:
            Adds a file to the document store and vector store, processing its content and generating summaries.
        add_folder(folder: str, recursive: bool = False):
            Adds all files in a folder to the document store and vector store, optionally processing files recursively.
        calculate_crc32(file_path: str | Path) -> int:
            Calculates the CRC32 hash of a file.
    """
    
    RAG_PREFIX = "RAGGAE"
    
    def __init__(self, llm:str='OpenAI', debug:bool=False):
        """
        Initializes the RagStore class.
        Args:
            llm (str): The type of language model to use. Default is 'OpenAI'.
            debug (bool): Flag to enable or disable debug mode. Default is False.
        Attributes:
            _debug (bool): Stores the debug mode status.
            _llm_type (str): Stores the type of language model.
            _docstore (DocumentStore): An instance of the DocumentStore class.
            _vectorstore (VectorStore): The vector store used to index the summaries and their embeddings.
        """
        self._debug = debug
        # LLM
        self._llm_type = llm
        match llm:
            case 'OpenAI':
                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
                if not embedding_model:
                    embedding_model = 'text-embedding-3-small'
                embedding_function = OpenAIEmbeddings(model=embedding_model)

        # An instance of the DocumentStore class
        self._docstore = DocumentStore(idx_prefix=self.RAG_PREFIX)
        # The vectorstore to use to index the summaries and their embeddings
        self._vectorstore = VectorStore(embedding_function=embedding_function)
    
    def get_summariser(self):
        """
        Retrieves the summariser instance based on the LLM (Large Language Model) type.
        Returns:
            OpenAISummariser: An instance of the OpenAISummariser class configured with the appropriate model.
        Raises:
            ValueError: If the LLM type is not recognized.
        """
        match self._llm_type:
            case 'OpenAI':
                summariser_model = os.getenv("OPENAI_SUMMARISER_MODEL")
                if not summariser_model:
                    summariser_model = 'gpt-4o-mini'
                return OpenAISummariser(model=summariser_model, debug=self._debug)
    
    def add_file(self, fn:str|Path, store:bool=True, overwrite:bool=False)->Tuple[str, dict]:
        """
        Adds a file to the document store and vector store after processing its contents.
        Args:
            fn (str | Path): The file path to be added.
            store (bool, optional): Whether to store the file. Defaults to True.
            overwrite (bool, optional): Whether to overwrite existing entries. Defaults to False.
        Returns:
            Tuple[str, dict]: A tuple containing the document ID and metadata.
        Raises:
            FileNotFoundError: If the file does not exist.
        Notes:
            - The file is partitioned into chunks and elements.
            - Each element is processed based on its type (Image, Table, Title, FigureCaption, NarrativeText, Text).
            - Metadata is set for each element.
            - Documents are stored in the document store, and summaries are stored in the vector store.
        """
        if isinstance(fn, str):
            fn = Path(fn)
        if not fn.exists():
            raise FileNotFoundError(f"File {fn} not found")
        # Calculate a CRC32 hash for the file
        crc = f'{self.RAG_PREFIX}:{self.calculate_crc32(fn)}'
        # Query the database for the CRC
        if not overwrite and self._docstore.is_in(crc):
            logger.warning(f"File {fn} already in store")
            return
        # Obtain a new summariser instance
        sf = self.get_summariser()
        with open(fn, "rb") as f:
            # Create a document summary as context for the elements
            chunks = partition(file=f, chunking_strategy="by_title")
            # Convert them to Document objects
            chunks = [Document(page_content=chunk.text) for chunk in chunks]
            sf.set_context(chunks)
            # Now process the different elements separately        
            elements = partition(file=f,
                                strategy="hi_res",                                     # mandatory to use ``hi_res`` strategy
                                extract_images_in_pdf=True,                            # mandatory to set as ``True``
                                extract_image_block_types=["Image", "Table"],          # optional
                                extract_image_block_to_payload=False,                  # optional
                                extract_image_block_output_dir="./tmp",
                                unique_element_ids=True)                               # gives globally unique element ids
            # Process each element based on its type
            # for each :
            # - Image
            #   Create a summary and uuencode the image
            # - Table, FigureCaption, NarrativeText
            #   Create a summary
            # - Title, Text
            #   Use as is
            # - Other
            #   Ignore
            # Then set metadata for the element:
            #   - doc-id: the document hash crc
            #   - chunk-id: the element id
            #   - type: the element type
            #   - source: the originating file name (no path)
            #   - page_number: the page number
            #   - languages: the languages detected in the element
            # Documents are stored in the document store, summaries are stored in the vector store
            documents = []
            summaries = []
            
            for element in elements:
                metadata = {
                    "doc-id": crc,
                    "chunk-id": element.id,
                    "source": fn.name,
                    "page_number": element.metadata.page_number,
                    "languages": ",".join(element.metadata.languages),
                    "type": element.category
                }
                summary = None
                text = None
                if isinstance(element, Table):
                    imagepath = element.metadata.image_path
                    summary = sf.summarise_table(element.text, imagepath)
                    with open(imagepath, "rb") as image_file:
                        text = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(element, Image):
                    imagepath = element.metadata.image_path
                    summary = sf.summarise_image(element.text, imagepath)
                    with open(imagepath, "rb") as image_file:
                        text = base64.b64encode(image_file.read()).decode("utf-8")
                elif isinstance(element, Title):
                    text = element.text
                elif isinstance(element, FigureCaption):
                    text = element.text
                elif isinstance(element, NarrativeText):
                    summary = sf.summarise_text(element.text)
                    text = element.text
                elif isinstance(element, Text):
                    text = element.text
                else:
                    # We don't care about the other types
                    continue
            
                if summary:
                    summaries.append(Document(page_content=summary, metadata=metadata))
                if text:
                    document = metadata.copy()
                    document['content'] = text
                    document_id = f'{self.RAG_PREFIX}:{self.get_hash(text)}'
                    documents.append({'document_id': document_id, 'document': document})   
        # We assume that the process of a round trip to the document store is cheap
        # compared with the partitioning
        self._docstore.add(documents)
        self._vectorstore.add(summaries)
    
    def add_folder(self, folder:str, recursive:bool=False):
        """
        Adds all files from the specified folder to the store.
        Args:
            folder (str): The path to the folder to add.
            recursive (bool, optional): If True, add files from subdirectories recursively. Defaults to False.
        Raises:
            FileNotFoundError: If the specified folder does not exist.
        Logs:
            Info: When a file is successfully processed.
            Error: When there is an error processing a file.
        """
        folderpath = Path(folder)
        if not folderpath.exists():
            raise FileNotFoundError(f"Folder {folder} not found")
        if folderpath.is_file():
            self.add_file(folder)
        if recursive:
            files = folderpath.rglob("*")
            files = [fn for fn in files if fn.is_file()]
        else:
            files = [fn for fn in folderpath.iterdir() if fn.is_file()]
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.add_file, fn, False): fn for fn in files}
            for future in as_completed(futures):
                fn = futures[future]
                try:
                    future.result()
                    logger.info(f"File {fn} processed")
                except Exception as e:
                    logger.error(f"Error processing file {fn}: {e}")

    def get_hash(self, text:str)->str:
        """
        Generates a hash for the given text.
        Args:
            text (str): The text to hash.
        Returns:
            str: The hash of the text.
        """
        return hashlib.md5(text.encode()).hexdigest
    
    def calculate_crc32(self,file_path: str|Path)->int:
        """
        Calculate the CRC32 checksum of a file.
        Args:
            file_path (str | Path): The path to the file for which the CRC32 checksum is to be calculated.
        Returns:
            int: The calculated CRC32 checksum as an unsigned 32-bit integer.
        """
        crc = 0  # Initial CRC value
        buffer_size = 65536  # Read the file in chunks (64 KB)
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        with open(file_path, 'rb') as file:
            while chunk := file.read(buffer_size):
                crc = zlib.crc32(chunk, crc)
        return crc & 0xFFFFFFFF  # Ensure a consistent unsigned 32-bit result


class OpenAISummariser():
    '''
    A class to summarize text, tables, and images using OpenAI's language models for semantic retrieval.
    Attributes:
    -----------
    SUMMARY_PROMPT_TEXT : str
        Template for summarizing text.
    SUMMARY_PROMPT_TABLE : str
        Template for summarizing tables.
    SUMMARY_PROMPT_IMAGE : str
        Template for summarizing images.
    SYS_PROMPT : str
        System prompt for brief answers.
    CONTEXT_WINDOW : dict
        Dictionary containing context window sizes for different models.
    Methods:
    --------
    __init__(model: str='gpt-4o', debug: bool=False):
        Initializes the summarizer with the specified model and debug mode.
    estimate_tokens(data) -> int:
        Estimates the number of tokens in the given data.
    log_prompt(data):
        Logs the prompt if debug mode is enabled.
    check_token_length(data):
        Checks if the token length of the data exceeds the maximum context length.
    set_context(chunks: list):
        Sets the context by summarizing the provided chunks.
    summarise_text(text: str) -> str:
        Summarizes the given text using the text summary prompt.
    summarise_table(text: str, image_fn: str = None) -> str:
        Summarizes the given table using the table summary prompt.
    summarise_image(text: str, image_fn: str = None) -> str:
        Summarizes the given image using the image summary prompt.
    summarise(text: str, image_fn: str, prompt: ChatPromptTemplate) -> str:
        Summarizes the given text and/or image using the specified prompt.
    '''
    SUMMARY_PROMPT_TEXT = """
        You are an assistant tasked with summarizing text particularly for semantic retrieval.
        These summaries will be embedded and used to retrieve the raw text. 
        Give a detailed summary of the text below that is well optimized for retrieval. 
        The context of the text is also available for you to use.
        Do not add redundant words like Summary.
        Just output the actual summary content.

        Text chunk:
        {content}
        
        Context:
        {context}

        """
        
    SUMMARY_PROMPT_TABLE = """
        You are an assistant tasked with summarizing tables particularly for semantic retrieval.
        These summaries will be embedded and used to retrieve the table elements.
        Give a detailed summary of the table below that is well optimized for retrieval. For this,
        use the table transcription and the image provided.
        Add in a one line description of what the table is about besides the summary.
        The context of the table is also available for you to use.
        Do not add redundant words like Summary.
        Just output the actual summary content.

        Table transcription:
        {content}
        
        Table image:
        {image_base64}
        
        Context:
        {context}
        """
    
    SUMMARY_PROMPT_IMAGE = """
        You are an assistant tasked with summarizing images for retrieval.
        Remember these images could potentially contain graphs, charts or tables also.
        These summaries will be embedded and used to retrieve the raw image for question answering.
        Give a detailed summary of the image that is well optimized for retrieval.
        The context of the image is also available for you to use.
        Do not add additional words like Summary, This image represents, etc.
        
        Image:
        {image_base64}
        
        Context:
        {context}
        """
        
    SYS_PROMPT = """Act as a helpful assistant and give brief answers"""
    
    CONTEXT_WINDOW = {
        'gpt-4o': 128000,
        'gpt-4o-mini': 128000,
    }
    
    def __init__(self, model:str='gpt-4o', debug:bool=False):
        """
        Initializes the Ragstore class.
        Args:
            model (str): The name of the model to use. Default is 'gpt-4o'.
            debug (bool): Flag to enable or disable debug mode. Default is False.
        Attributes:
            _llm (ChatOpenAI): The language model instance.
            _max_context_length (int): The maximum context length for the model.
            _debug (bool): Debug mode flag.
            _context (None): Placeholder for context, initially set to None.
            _tokenizer (tiktoken): Tokenizer instance for the specified model.
        """
        self._llm = ChatOpenAI(model_name=model, temperature=0)
        self._max_context_length = self.CONTEXT_WINDOW.get(model, 128000)
        self._debug = debug
        self._context = None
        self._tokenizer = tiktoken.encoding_for_model(model)
    
    def estimate_tokens(self, data)->int:
        """
        Estimate the number of tokens in the given data.
        This method takes a data object, which can be an instance of either 
        StringPromptValue or ChatPromptValue, and calculates the number of tokens 
        in the text content of the data using the tokenizer.
        Args:
            data: An instance of either StringPromptValue or ChatPromptValue. 
                    - If data is a StringPromptValue, the text attribute is used.
                    - If data is a ChatPromptValue, the content of each message in 
                    the messages attribute is concatenated.
        Returns:
            int: The number of tokens in the provided data.
        """        
        if isinstance(data, StringPromptValue):
            prompt = data.text
        elif isinstance(data, ChatPromptValue):
            if data.messages:
                msgs = data.messages
                for msg in msgs:
                    prompt += msg.content
        return len(self._tokenizer.encode(prompt))           
    
    def log_prompt(self, data):
        """
        Logs the final prompt data if debugging is enabled.
        Args:
            data (str): The prompt data to be logged.
        Returns:
            str: The same prompt data that was passed in.
        """
        length = self.estimate_tokens(data)
        if self._debug:
            if length > 0:
                logger.info(f"Final prompt has length {length}: {data}")
            else:
                logger.info(f"Final prompt: {data}")
        return data

    def check_token_length(self, data):
        """
        Checks if the token length of the given data exceeds 90% of the maximum context length.
        Args:
            data (str): The input data to be checked.
        Returns:
            str: The original data if the token length is within the acceptable range.
        Raises:
            ValueError: If the token length exceeds 90% of the maximum context length.
        """
        length = self.estimate_tokens(data)
        if length > int(0.9*self._max_context_length):
            logger.warning(f"Prompt length {length} exceeds maximum {self._max_context_length}")
            raise ValueError
        return data

    def set_context(self, chunks:list):
        """
        Sets the context by summarizing the provided chunks using a summarization chain.
        Args:
            chunks (list): A list of text chunks to be summarized.
        Returns:
            None
        Logs:
            Information about the cost in tokens for summarizing the document.
        """
        with get_openai_callback() as cb:
            # Load the summarization chain with map_reduce
            chain = load_summarize_chain(self._llm, chain_type="map_reduce")
            self._context = chain.invoke(chunks)
            logger.info(f"Summarising the document costed {cb.total_tokens} tokens")

    def summarise_text(self, text:str)->str:
        """
        Summarizes the given text using a predefined prompt template.
        Args:
            text (str): The text to be summarized.
        Returns:
            str: The summarized text.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TEXT)
        return self.summarise(text=text, prompt=prompt)

    def summarise_table(self, text:str, image_fn:str = None)->str:
        """
        Summarizes the content of a table provided in text format.
        Args:
            text (str): The text representation of the table to be summarized.
            image_fn (str, optional): The filename of an image associated with the table. Defaults to None.
        Returns:
            str: A summary of the table content.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TABLE)
        return self.summarise(text=text, image_fn=image_fn, prompt=prompt)

    def summarise_image(self, text:str, image_fn:str = None)->str:
        """
        Summarizes the given text and optionally an image.
        Args:
            text (str): The text to be summarized.
            image_fn (str, optional): The filename of the image to be summarized along with the text. Defaults to None.
        Returns:
            str: The summary of the text and optionally the image.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_IMAGE)
        return self.summarise(text=text, image_fn=image_fn, prompt=prompt)

    def summarise(self, text:str, prompt: ChatPromptTemplate, image_fn:str|None=None)->str:
        """
        Summarizes the given text and optionally includes an image in the prompt.
        Args:
            text (str): The text to be summarized.
            image_fn (str): The file name of the image to be included in the prompt. If empty, no image is included.
            prompt (ChatPromptTemplate): The prompt template to be used for summarization.
        Returns:
            str: The summary of the given text.
        """        
        image_base64 = ""
        if image_fn:
            with open(image_fn, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        # Steps for debugging and for token estimation
        debug_step = RunnableLambda(self.log_prompt)
        token_est_step = RunnableLambda(self.check_token_length)
        
        ingestion = {}
        invocation = {}
        if "content" in prompt.input_placeholders:
            ingestion["content"] = RunnablePassthrough()
            invocation["content"] = text
        if "image_base64" in prompt.input_placeholders:
            ingestion["image_base64"] = RunnablePassthrough()
            invocation["image_base64"] = image_base64
        if "context" in prompt.input_placeholders:
            ingestion["context"] = RunnablePassthrough()
            invocation["context"] = self._context
        
        chain = (
            ingestion 
                |
            prompt
                |
            debug_step
                |
            token_est_step
                |
            self._llm
                | 
            StrOutputParser()
        )

        # Run the chain
        try:
            summary = chain.invoke(invocation)
        except ValueError:
            summary = ""
        return summary
        
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
        return self._dbc.exists(key)
    
    def add(self, documents:list[dict]):
        def add(self, documents: list[dict]):
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
            TextField("doc_id"),          # Document hash
            TextField("chunk_id"),        # Element ID
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
                    query = query.add_filter(f"@{field}:{{{value}}}")
                else:
                    query = query.add_filter(f"@{field}:{value}")

        # Execute the search
        results = self._dbc.ft(index_name).search(query)

        # Format results
        documents = []
        for doc in results.docs:
            documents.append({
                "doc_id": doc.doc_id,
                "chunk_id": doc.chunk_id,
                "type": doc.type,
                "source": doc.source,
                "page_number": doc.page_number,
                "languages": doc.languages.split(",") if doc.languages else [],
                "content": doc.content
            })

        return documents
    
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
        