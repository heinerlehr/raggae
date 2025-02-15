import os
# Logging
from loguru import logger
# Langchain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain.schema import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import RateLimitError, OpenAIError
# retry
from retry import retry
# Local classes
from document_store import DocumentStore
from vector_store import VectorStore
import tiktoken 

class RAGQuery():
    '''
    RAGQuery class for handling retrieval-augmented generation queries.
    Attributes:
        FULL_PROMPT (str): Template for generating detailed answers using context and history.
        KEYWORK_PROMPT (str): Template for extracting keywords from a query and context.
        QUERY_RE_PROMPT (str): Template for re-formulating a query based on context and history.
        HISTORY_SUMMARY_PROMPT (str): Template for summarizing chat history relevant to a query.
        CONTEXT_WINDOW (dict): Dictionary mapping model names to their maximum context lengths.
    Methods:
        __init__(vector_store, document_store, llm='OpenAI', debug=False, model='gpt-4o', topk=5, n_kw=5):
            Initializes the RAGQuery instance with the specified parameters.
        query(query: str, n_kw: int = 5) -> str:
            Processes a query to generate an answer using context, history, and keyword extraction.
        get_history_summary(query: str) -> str:
            Summarizes the chat history relevant to the given query.
        estimate_tokens(data) -> int:
            Estimates the number of tokens in the given data.
        log_prompt(data):
        check_token_length(data):
        execute(query: str, prompt: ChatPromptTemplate, n_kw: str | None = None, context: str | None = None, history: str | None = None) -> str:
            Executes the prompt chain to generate a summary or answer based on the given query, context, and history.
        invoke(invocation: dict, chain: RunnablePassthrough) -> str:
    '''
    
    FULL_PROMPT = '''
    You are an assistant who is an expert in question-answering tasks.
    Answer the following question using only the following pieces of retrieved context and the history.
    If the answer is not in the context or the hisotry, do not make up answers, just say that you don't know.
    Keep the answer detailed and well formatted based on the information from the context and history.
    
    Query:
    {query}
    
    Context:
    {context}
    
    '''
    
    KEYWORK_PROMPT = '''
    You are an assistant who is an expert in finding keywords from a query and chunks of text. 
    Return a list of no more than {n_kw} keywords that are relevant to the query from the context. The keywords will be used for
    a full-text search in a document database. Return the list of keywords in a comma-separated format.
    
    Query:
    {query}
    
    Context:
    {context}
    '''
    
    QUERY_RE_PROMPT = '''
    Using the history and the context, re-formulate the query to be more specific and relevant to the context.
    
    Query:
    {query}
    
    Context:
    {context}
    
    History:
    {history}
    '''
    
    HISTORY_SUMMARY_PROMPT = '''
        Summarize the following chat history in a concise way relevant to the query.
        
        Chat History:
        {history}
        
        Query:
        {query}
        
        Concise Summary:
    '''
    
    CONTEXT_WINDOW = {
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
    }

    def __init__(self, vector_store: VectorStore, document_store:DocumentStore, llm:str='OpenAI', debug:bool=False, model:str='gpt-4o', topk:int=5, n_kw:int=5):
        """
        Initializes the RagQuery class.
        Args:
            vector_store (VectorStore): The vector store instance for similarity search.
            document_store (DocumentStore): The document store instance for storing documents.
            llm (str, optional): The type of language model to use. Defaults to 'OpenAI'.
            debug (bool, optional): Flag to enable or disable debug mode. Defaults to False.
            model (str, optional): The model name to use for the language model. Defaults to 'gpt-4o'.
            topk (int, optional): The number of top similar documents to retrieve. Defaults to 5.
            n_kw (int, optional): The number of keywords to use. Defaults to 5.
        Raises:
            ValueError: If the specified language model type is not supported.
        """
        
        self._debug = debug
        # LLM
        self._llm_type = llm
        self._n_kw = n_kw
        match llm:
            case 'OpenAI':
                self._llm = ChatOpenAI(model_name=model, temperature=0)
                self._max_context_length = self.CONTEXT_WINDOW.get(model, 128000)
                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
                if not embedding_model:
                    embedding_model = 'text-embedding-3-small'
                # embedding_function = OpenAIEmbeddings(model=embedding_model)
            case _:
                raise ValueError(f"LLM type {llm} not supported")
            
        # Vector and document stores
        self._vector_store = vector_store
        self._document_store = document_store
        # Similarity retrieval
        self._retriever = self._vector_store.get_db().as_retriever(search_type="similarity", search_kwargs={"k": topk})
        # History
        self._history = ChatMessageHistory()
        self._tokenizer = tiktoken.encoding_for_model(model)
        
        
    def query(self, query:str, n_kw:int=5)->str:
        """
        Processes a query to retrieve relevant documents, generate keywords, and reformulate the query based on history and context.
        Args:
            query (str): The input query string.
            n_kw (int, optional): The number of keywords to generate. Defaults to 5.
        Returns:
            str: The final answer generated after processing the query.
        """
        
        # Obtain relevant chunks from vector store
        documents = self._retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in documents])
        # We'll need that later
        chunk_ids = [doc.metadata['chunk-id'] for doc in documents]
        # Generate keywords from query and context
        kw_prompt = ChatPromptTemplate.from_template(self.KEYWORK_PROMPT)
        kw_str = self.execute(query=query, prompt=kw_prompt, context=context, n_kw=self._n_kw)
        keywords = kw_str.split(",")
        
        # Obtain relevant documents from document store via keyword search
        if keywords:
            docs = self._document_store.full_text_search(query, filters={"keywords": keywords})
            context += "\n".join([doc['content'] for doc in docs if doc['chunk-id'] not in chunk_ids])
        
        # Obtain history and re-formulate query based on it and the context
        history = self.get_history_summary(query)
        re_query_prompt = ChatPromptTemplate.from_template(self.QUERY_RE_PROMPT)
        re_query = self.execute(query=query, prompt=re_query_prompt, context=context, history=history)
        re_query_prompt = ChatPromptTemplate.from_template(self.FULL_PROMPT)
        answer = self.execute(query=re_query, prompt=re_query_prompt, context=context)
        
        # Add query and answer to the history
        self._history.add_message(HumanMessage(content=re_query))
        self._history.add_message(AIMessage(content=answer))
        
        # Return the answer
        return answer
    
    def get_history_summary(self, query:str)->str:
        """
        Generates a summary of the history based on the provided query.
        Args:
            query (str): The query string to generate the history summary.
        Returns:
            str: The generated summary of the history.
        """
        history_summary_prompt = ChatPromptTemplate.from_template(self.HISTORY_SUMMARY_PROMPT)
        
        formatted_history = "\n".join(
            f"{msg.type.capitalize()}: {msg.content}" for msg in self._history.messages
        )
        summary = self.execute(query=query, n_kw=self._n_kw, prompt=history_summary_prompt, history=formatted_history)
        return summary
        
        
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
        if length > int(0.9*self._max_context_length):
            logger.warning(f"Prompt length {length} exceeds maximum {self._max_context_length}")
            raise OpenAIError(f"Prompt length {length} exceeds maximum {self._max_context_length}")
        return data
            
    def execute(self, query:str, prompt: ChatPromptTemplate, n_kw:str|None=None, context:str|None=None, history:str|None=None)->str:
        """
        Summarizes the given text and optionally includes an image in the prompt.
        Args:
            text (str): The text to be summarized.
            image_fn (str): The file name of the image to be included in the prompt. If empty, no image is included.
            prompt (ChatPromptTemplate): The prompt template to be used for summarization.
        Returns:
            str: The summary of the given text.
        """        

        # Steps for debugging and for token estimation
        debug_step = RunnableLambda(self.log_prompt)
        token_est_step = RunnableLambda(self.check_token_length)
        
        ingestion = {}
        invocation = {}
        if "query" in prompt.input_variables:
            ingestion["query"] = RunnablePassthrough()
            invocation["query"] = query
        if "context" in prompt.input_variables:
            ingestion["context"] = RunnablePassthrough()
            invocation["context"] = context
        if "history" in prompt.input_variables:
            ingestion["history"] = RunnablePassthrough()
            invocation["history"] = history
        if "n_kw" in prompt.input_variables:
            ingestion["n_kw"] = RunnablePassthrough()
            invocation["n_kw"] = n_kw        

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
            summary = self.invoke(invocation, chain)
        except ValueError:
            summary = ""
        return summary
        
    @retry(exceptions=(ValueError, RateLimitError), delay=2, retries=3)
    def invoke(self, invocation:dict, chain:RunnablePassthrough)->str:
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
                raise RateLimitError("Rate limit") from e # to trigger a retry
        except OpenAIError as e:
            # Handle token-related or rate-limiting issues
            if "context length" in str(e).lower():
                logger.error(f"Context length exceeded: {e}")
                return "" # Can't do anything about this, so return an empty string
            elif "prompt length" in str(e).lower(): # This is triggered by the check_token_length method
                return "" # Can't do anything about this, so return an empty string
            else:
                logger.error(f"LLM-related error: {e}")
                raise ValueError("LLM-related error") from e # to trigger a retry