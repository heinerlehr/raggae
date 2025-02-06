import os
# Logging
from loguru import logger
# Langchain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompt_values import ChatPromptValue, StringPromptValue
from langchain.memory import ChatMessageHistory, ChatMessage
from openai import RateLimitError, OpenAIError
# retry
from retry import retry
# Local classes
from document_store import DocumentStore
from vector_store import VectorStore
import tiktoken 

class RAGQuery():
    
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
    
    
    def __init__(self, vector_store: VectorStore, document_store:DocumentStore, llm:str='OpenAI', debug:bool=False, model:str='gpt-4o', topk:int=5):
        self._debug = debug
        # LLM
        self._llm_type = llm
        match llm:
            case 'OpenAI':
                self._llm = ChatOpenAI(model_name=model, temperature=0)
                embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
                if not embedding_model:
                    embedding_model = 'text-embedding-3-small'
                # embedding_function = OpenAIEmbeddings(model=embedding_model)
        # Vector and document stores
        self._vector_store = vector_store
        self._document_store = document_store
        # Similarity retrieval
        self._retriever = self._vector_store.get_db().as_retriever(search_type="similarity", search_kwargs={"k": topk})
        # History
        self._history = ChatMessageHistory()
        self._tokenizer = tiktoken.encoding_for_model(model)
        
    def query(self, query:str, n_kw:int=5)->str:
        
        # Obtain relevant chunks from vector store
        documents = self._retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in documents])
        # We'll need that later
        chunk_ids = [doc.chunk_id for doc in documents]
        # Generate keywords from query and context
        kw_prompt = ChatPromptTemplate.from_template(self.KEYWORK_PROMPT)
        kw_str = self.execute(query=query, prompt=kw_prompt, context=context)
        keywords = kw_str.split(",")
        
    
        # Obtain relevant documents from document store via keyword search
        if keywords:
            docs = self._document_store.full_text_search(query, filters={"keywords": keywords})
            context += "\n".join([doc.content for doc in docs if doc.chunk-id not in chunk_ids])
            
        
        # Obtain history and re-formulate query based on it and the context
        history = self.get_history_summary(query)
        re_query_prompt = ChatPromptTemplate.from_template(self.QUERY_RE_PROMPT)
        re_query = self.execute(query=query, prompt=re_query_prompt, context=context, history=history)
        answer = self.execute(query=query, prompt=self.FULL_PROMPT.format(query=re_query, context=context), context=context)
        
        # Add query and answer to the history
        self._history.add_message(ChatMessage(type="human", content=re_query))
        self._history.add_message(ChatMessage(type="ai", content=answer))
        
        # Return the answer
        return answer
    
    def get_history_summary(self, query:str)->str:
        history = self._history.get_history()
        history_summary_prompt = ChatPromptTemplate.from_template(self.HISTORY_SUMMARY_PROMPT)
        
        formatted_history = "\n".join(
            f"{msg.type.capitalize()}: {msg.content}" for msg in history.messages
        )
        summary = self.execute(query=query, prompt=history_summary_prompt, history=formatted_history)
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
            
    def execute(self, query:str, prompt: ChatPromptTemplate, context:str|None=None, history:str|None=None)->str:
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