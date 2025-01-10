Collecting workspace information


# RAGGAE

RAGGAE is a Python-based project designed to facilitate the storage, retrieval, and summarization of documents using advanced language models and vector stores. The project leverages various libraries and tools to process and manage documents efficiently.

## Features

- **Document Storage**: Store documents in a Redis-based document store.
- **Vector Store**: Index and retrieve document summaries using a vector store.
- **Summarization**: Summarize text, tables, and images using OpenAI models.
- **Concurrent Processing**: Efficiently process multiple files concurrently using thread pools.
- **CRC32 Hashing**: Ensure document uniqueness using CRC32 hashing.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/raggae.git
    cd raggae
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Set up environment variables:
    - Create a 

.env

 file in the root directory.
    - Add the following environment variables:
        ```
        OPENAI_API_KEY=your_openai_api_key
        OPENAI_EMBEDDING_MODEL=text-embedding-3-small
        CHROMA_COLLECTION_NAME=RAGGAE
        CHROMA_PERSIST_FN=./db
        ```

## Usage

### Adding a File

To add a single file to the store, use the 

add_file

 method:
```python
from ragstore import RAGStore

ragstore = RAGStore()
ragstore.add_file('path/to/your/file.txt')

### Adding a Folder

To add all files from a folder to the store, use the 

add_folder

 method:
```python
ragstore.add_folder('path/to/your/folder', recursive=True)
```

### Summarization

The project supports summarizing text, tables, and images using OpenAI models. The summaries are stored in the vector store for efficient retrieval.

## Project Structure

```
raggae/
├── __init__.py
├── ragstore.py
├── README.md
├── .env
├── .gitignore
├── pyproject.toml
├── test/
│   └── RAG Project Dataset/
└── tmp/
```

## Dependencies

- `chromadb>=0.5.23`
- `filetype>=1.2.0`
- `htmltabletomd>=1.0.0`
- `ipykernel>=6.29.5`
- `ipywidgets>=8.1.5`
- 

langchain>=0.3.7


- 

langchain-chroma==0.1.4


- 

langchain-community>=0.3.7


- 

langchain-openai>=0.2.14


- 

loguru>=0.7.3


- `numpy==1.26`
- `pandas>=2.2.3`
- `pip>=24.3.1`
- `pydantic==2.9.2`
- `pydotenv>=0.0.7`
- 

redis>=5.2.1


- 

tiktoken>=0.8.0



## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [yourname@domain.com](mailto:yourname@domain.com).

---

*Note: Replace placeholders like `yourusername`, `your_openai_api_key`, and `yourname@domain.com` with actual values.*
```