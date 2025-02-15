Collecting workspace information


# RAGGAE

RAGGAE is a Python-based project designed to facilitate the storage, retrieval, and summarization of documents using advanced language models and vector stores. The project leverages various libraries and tools to process and manage documents efficiently.
**NOTE:** This is an academic exercise and provides some skeleton code for a RAG system.

## Features

- **Document Storage**: Store documents in a Redis-based document store.
- **Vector Store**: Index and retrieve document summaries using a vector store.
- **Summarization**: Summarize text, tables, and images using OpenAI models.
- **Concurrent Processing**: Efficiently process multiple files concurrently using thread pools.
- **CRC32 Hashing**: Ensure document uniqueness using CRC32 hashing.

## Installation

A Dockerfile and a docker-compose.yaml are provided for its use in a Docker container. 

If a more manual install is preferred:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/raggae.git
    cd raggae
    ```

2. Install the required dependencies:
    ```sh
    uv sync
    ```

3. Set up environment variables:
    - Create a .env file in the root directory.
    - Add the following environment variables:
        ```
        OPENAI_API_KEY=your_openai_api_key
        OPENAI_EMBEDDING_MODEL=text-embedding-3-small
        CHROMA_COLLECTION_NAME=RAGGAE
        CHROMA_PERSIST_FN=./db
        ```

## Usage

The provided Jupyter notebook "TestRaggae" shows the basic usage.

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

See uv.lock

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Contact

For any questions or inquiries, please contact [heiner@beakanalytics.com](mailto:heiner@beakanalytics.com).
