import base64

# Hash
import hashlib
import os
import time
import zlib
from pathlib import Path
import shutil 
from typing import Tuple, List
import numpy as np
# Concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.document_loaders import UnstructuredPDFLoader

# Langchain
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from openai import OpenAIError, RateLimitError
# Image management
from PIL import Image

from document_store import DocumentStore

# retry
from retry import retry

# local classes
from vector_store import VectorStore


class RAGStore:
    """
    A class to handle the storage and processing of documents using a Retrieval-Augmented Generation (RAG) approach.
    Attributes:
        RAG_PREFIX (str): Prefix used for document identifiers.
        _debug (bool): Flag to enable or disable debug mode.
        _llm_type (str): Type of the language model to use.
        _document_store (DocumentStore): Instance of the DocumentStore class.
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
    TMPDIR = './tmp'

    def __init__(self, llm: str = "OpenAI", debug: bool = False, imgsize:Tuple[int,int]=(200,150)):
        """
        Initializes the RagStore class.
        Args:
            llm (str): The type of language model to use. Default is 'OpenAI'.
            debug (bool): Flag to enable or disable debug mode. Default is False.
        Attributes:
            _debug (bool): Stores the debug mode status.
            _llm_type (str): Stores the type of language model.
            _document_store (DocumentStore): An instance of the DocumentStore class.
            _vectorstore (VectorStore): The vector store used to index the summaries and their embeddings.
        """
        self._debug = debug
        # LLM
        self._llm_type = llm
        match llm:
            case "OpenAI":
                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
                if not embedding_model:
                    embedding_model = "text-embedding-3-small"
                embedding_function = OpenAIEmbeddings(model=embedding_model)

        # An instance of the DocumentStore class
        try:
            self._document_store = DocumentStore(
                host="redis", idx_prefix=self.RAG_PREFIX
            )
        except Exception:
            try:
                self._document_store = DocumentStore(
                    host="localhost", idx_prefix=self.RAG_PREFIX
                )
            except ConnectionError:
                logger.error("Could not connect to the document store")
                return None
        # The vectorstore to use to index the summaries and their embeddings
        self._vector_store = VectorStore(embedding_function=embedding_function)
        # Max number of pixels in images
        self._imgsize = imgsize[0]*imgsize[1]

    def get_summariser(self):
        """
        Retrieves the summariser instance based on the LLM (Large Language Model) type.
        Returns:
            OpenAISummariser: An instance of the OpenAISummariser class configured with the appropriate model.
        Raises:
            ValueError: If the LLM type is not recognized.
        """
        match self._llm_type:
            case "OpenAI":
                summariser_model = os.getenv("OPENAI_SUMMARISER_MODEL")
                if not summariser_model:
                    summariser_model = "gpt-4o-mini"
                return OpenAISummariser(model=summariser_model, debug=self._debug)

    def get_vector_store(self):
        """
        Retrieves the vector store instance.
        Returns:
            VectorStore: An instance of the VectorStore class.
        """
        return self._vector_store

    def get_document_store(self):
        """
        Retrieves the document store instance.
        Returns:
            VectorStore: An instance of the VectorStore class.
        """
        return self._document_store

    def add_file(self, fn: str | Path, overwrite: bool = False, purge_folder:bool=True) -> Tuple[str, dict]:
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
        start = time.time()
        # Empty the TMPDIR folder before processing
        if purge_folder:
            if Path(self.TMPDIR).exists():
                shutil.rmtree(self.TMPDIR)
        logger.info(f"Processing of file {fn} starts")
        if isinstance(fn, str):
            fn = Path(fn)
        if not fn.exists():
            raise FileNotFoundError(f"File {fn} not found")
        # Calculate a CRC32 hash for the file
        crc = f"{self.RAG_PREFIX}:{self.calculate_crc32(fn)}"
        # Query the database for the CRC
        if not overwrite and self._document_store.is_in(crc.split(":")[1]):
            logger.warning(f"File {fn} already in store")
            return
        # Obtain a new summariser instance
        sf = self.get_summariser()
        # Now process the different elements separately
        # Tables
        logger.info("Extracting tables")
        loader = UnstructuredPDFLoader(file_path=fn,
                            strategy='hi_res',
                            extract_images_in_pdf=True,
                            infer_table_structure=True,
                            mode='elements',
                            extract_image_block_to_payload=False,                  # optional
                            extract_image_block_output_dir=self.TMPDIR+"/"+crc
                            )
        data = loader.load()
        logger.info(f"Tables extracted after {time.time() - start:.1f} seconds")
        tables = [doc for doc in data if doc.metadata['category'] == 'Table']
        # Texts
        logger.info("Extracting text elements")
        loader = UnstructuredPDFLoader(file_path=fn,
                            strategy='hi_res',
                            extract_images_in_pdf=True,
                            infer_table_structure=True,
                            chunking_strategy="by_title", # section-based chunking
                            max_characters=4000, # max size of chunks
                            new_after_n_chars=4000, # preferred size of chunks
                            combine_text_under_n_chars=2000, # smaller chunks < 2000 chars will be combined into a larger chunk
                            mode='elements',
                            extract_image_block_to_payload=False,                  # optional
                            extract_image_block_output_dir=self.TMPDIR+"/"+crc
                            )
        texts = loader.load()
        logger.info(f"Texts extracted after {time.time() - start:.1f} seconds")
        # Figures
        # First find all figures that are not related to tables
        logger.info("Extracting figures")
        referenced_figures = [table.metadata['image_path'] for table in tables if 'image_path' in table.metadata.keys()]
        figure_fns = [fn for fn in Path(self.TMPDIR+"/"+crc).glob('*') if fn not in referenced_figures]
        # Reduce images in size and provide the new filenames pointing to the reduced images
        figure_fns = self.reduce_image_size(figure_fns)
        figures = [Document(page_content='', metadata={'category': 'Image', 'chunk-id': f'{crc}.fig{i}', 'image_path': figure_fn,
                                        'page_number': 0, 'languages': ['eng']}) for i,figure_fn in enumerate(figure_fns)]
        logger.info(f"Figures extracted after {time.time() - start:.1f} seconds")
        logger.info("Summarising document")
        sf.set_context(texts)
        logger.info(f"Document summarised after {time.time() - start:.1f} seconds")
        elements = texts + tables + figures
        logger.info(f"File {fn} partitioned into {len(elements)} elements in {time.time() - start:.1f} seconds")
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

        for i,element in enumerate(elements):
            metadata = {
                "doc-id": crc,
                "source": fn.name,
                "page_number": element.metadata['page_number'],
                "languages": ",".join(element.metadata['languages']),
                "type": element.metadata['category'],
            }
            if 'chunk-id' in element.metadata.keys():
                metadata['chunk-id'] = element.metadata['chunk-id']
            else:
                metadata['chunk-id'] = f"{crc}.{i}"
            summary = None
            text = None
            if self._debug:
                logger.info(f"Processing element {metadata['chunk-id']} of type {element.metadata['category']}")
            match element.metadata['category']:
                case 'Table':
                    imagepath = None
                    summary = sf.summarise_table(element.page_content, imagepath)
                    if not summary:
                        text = element.page_content
                case 'Image':
                    imagepath = element.metadata['image_path']
                    summary = sf.summarise_image(None, imagepath)
                    if not summary:
                        text = base64.b64encode(open(imagepath, "rb").read()).decode("utf-8")
                case _:
                    text = element.page_content
                    if not text or len(text) == 0:
                        continue # No point summarising or storing an empty element
                    summary = sf.summarise_text(text)
                    if not summary:
                        summary = element.page_content

            if summary:
                if self._debug:
                    logger.info(f"Summary for element {metadata['chunk-id']} generated")
                summaries.append(Document(page_content=summary, metadata=metadata))

            if text:
                document = metadata.copy()
                document["content"] = text
                document_id = f"{self.RAG_PREFIX}:{self.get_hash(text)}"
                if self._debug:
                    logger.info(f"Document with {metadata['chunk-id']} generated")
                documents.append({"document_id": document_id, "document": document})
        if self._debug:
            logger.info(f"Processing of elements completed in {time.time() - start:.1f} seconds")
        # We assume that the process of a round trip to the document store is cheap
        # compared with the partitioning
        if documents:
            self._document_store.add(documents)
        else:
            logger.warning(f"No documents generated for file {fn}")
        if summaries:
            self._vector_store.add(summaries)
        else:
            logger.warning(f"No summaries generated for file {fn}")
        # Empty the TMPDIR folder before processing
        if purge_folder:
            if Path(self.TMPDIR).exists():
                shutil.rmtree(self.TMPDIR)
        
        logger.info(f"Processing of file {fn} completed in {time.time() - start:.1f} seconds")

    def add_folder(self, folder: str, recursive: bool = False):
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
        # Remove any images that might have been left over
        if Path(self.TMPDIR).exists():
            shutil.rmtree(self.TMPDIR)
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.add_file, fn=fn, overwrite=False, purge_folder=False): fn for fn in files}
            for future in as_completed(futures):
                fn = futures[future]
                try:
                    future.result()
                    logger.info(f"File {fn} processed")
                except Exception as e:
                    logger.error(f"Error processing file {fn}: {e}")
        # And clean up the TMPDIR folder
        if Path(self.TMPDIR).exists():
            shutil.rmtree(self.TMPDIR)
        
    def get_hash(self, text: str) -> str:
        """
        Generates a hash for the given text.
        Args:
            text (str): The text to hash.
        Returns:
            str: The hash of the text.
        """
        return hashlib.md5(text.encode()).hexdigest()

    def calculate_crc32(self, file_path: str | Path) -> int:
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
        with open(file_path, "rb") as file:
            while chunk := file.read(buffer_size):
                crc = zlib.crc32(chunk, crc)
        return crc & 0xFFFFFFFF  # Ensure a consistent unsigned 32-bit result

    def reduce_image_size(self, figure_fns:list, optimize:bool=True, quality:int=95)->List[str]:
        """
        Reduces the size of images to a maximum of 1024x1024 pixels.
        Args:
            figure_fns (list): List of image filenames.
        Returns:
            list: List of image filenames with reduced size.
        """
        reduced_fns = []
        for fn in figure_fns:
            try:
                with Image.open(fn) as img:
                    img_size = img.size
                    pixels = img_size[0]*img_size[1]
                    if pixels > self._imgsize:
                        new_x = int(round(img_size[0]/np.sqrt(pixels/self._imgsize),0))
                        new_y = int(round(img_size[1]/np.sqrt(pixels/self._imgsize),0))
                        img = img.resize((new_x, new_y), Image.LANCZOS)
                        reduced_fn = str(fn).replace(fn.suffix, f"_reduced{fn.suffix}")
                    else:
                        reduced_fn = str(fn)
                        reduced_fns.append(fn)
                    img.save(reduced_fn, optimize=optimize, quality=quality)
                    reduced_fns.append(reduced_fn)
            except Exception as e:
                logger.error(f"Error reducing image {fn}: {e}")
        # and return the new filenames
        return reduced_fns  


class OpenAISummariser:
    """
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
    """

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
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }

    def __init__(self, model: str = "gpt-4o", debug: bool = False):
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

    def estimate_tokens(self, data) -> int:
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
            prompt = ""
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
        if length > int(0.9 * self._max_context_length):
            logger.warning(
                f"Prompt length {length} exceeds maximum {self._max_context_length}"
            )
            raise OpenAIError(
                f"Prompt length {length} exceeds maximum {self._max_context_length}"
            )
        return data

    def set_context(self, chunks: list):
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
            chain = load_summarize_chain(self._llm, chain_type="stuff")
            response = chain.invoke(chunks)
            if response and 'output_text' in response.keys():
                self._context = response['output_text']
            logger.info(f"Summarising the document costed {cb.total_tokens} tokens")

    def summarise_text(self, text: str) -> str:
        """
        Summarizes the given text using a predefined prompt template.
        Args:
            text (str): The text to be summarized.
        Returns:
            str: The summarized text.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TEXT)
        return self.summarise(text=text, prompt=prompt)

    def summarise_table(self, text: str, image_fn: str = None) -> str:
        """
        Summarizes the content of a table provided in text format.
        Args:
            text (str): The text representation of the table to be summarized.
            image_fn (str, optional): The filename of an image associated with the table. Defaults to None.
        Returns:
            str: A summary of the table content.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TABLE)
        if not (summary := self.summarise(text=text, image_fn=image_fn, prompt=prompt)):
            prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TEXT)
            summary = self.summarise(text=text, prompt=prompt)
        return summary

    def summarise_image(self, text: str, image_fn: str = None) -> str:
        """
        Summarizes the given text and optionally an image.
        Args:
            text (str): The text to be summarized.
            image_fn (str, optional): The filename of the image to be summarized along with the text. Defaults to None.
        Returns:
            str: The summary of the text and optionally the image.
        """
        prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_IMAGE)
        if not (summary := self.summarise(text=text, image_fn=image_fn, prompt=prompt)):
            prompt = PromptTemplate.from_template(self.SUMMARY_PROMPT_TEXT)
            summary = self.summarise(text=text, prompt=prompt)
        return summary

    def summarise(
        self, text: str, prompt: ChatPromptTemplate, image_fn: str | None = None
    ) -> str:
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
        if "content" in prompt.input_variables:
            ingestion["content"] = RunnablePassthrough()
            invocation["content"] = text
        if "image_base64" in prompt.input_variables:
            ingestion["image_base64"] = RunnablePassthrough()
            invocation["image_base64"] = image_base64
        if "context" in prompt.input_variables:
            ingestion["context"] = RunnablePassthrough()
            invocation["context"] = self._context

        chain = (
            ingestion
            | prompt
            | debug_step
            | token_est_step
            | self._llm
            | StrOutputParser()
        )

        # Run the chain
        try:
            summary = self.invoke(invocation, chain)
        except ValueError:
            summary = ""
        return summary

    @retry(exceptions=(ValueError, RateLimitError), delay=2, retries=3)
    def invoke(self, invocation: dict, chain: RunnablePassthrough) -> str:
        """
        Invokes the chain with the specified invocation and returns the result.
        Args:
            invocation (dict): The input data for the chain.
            chain (RunnablePassthrough): The chain to be invoked.
        Returns:
            str: The result of invoking the chain.
        """
        try:
            return chain.invoke(invocation)
        except RateLimitError as e:
            if "rate limit" in str(e):
                logger.warning(f"Rate limit hit: {e}. Retrying...")
                raise RateLimitError("Rate limit") from e  # to trigger a retry
        except OpenAIError as e:
            # Handle token-related or rate-limiting issues
            if "context length" in str(e).lower():
                logger.error(f"Context length exceeded: {e}")
                return ""  # Can't do anything about this, so return an empty string
            elif (
                "prompt length" in str(e).lower()
            ):  # This is triggered by the check_token_length method
                return ""  # Can't do anything about this, so return an empty string
            else:
                logger.error(f"LLM-related error: {e}")
                raise ValueError("LLM-related error") from e  # to trigger a retry
