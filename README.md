# Semantic Java Code Search

A tool for building and querying a semantic search index for Java code repositories.

## Features

- Parses Java source code to extract classes and methods
- Generates semantic embeddings using OpenAI's embedding model
- Builds a vector index for fast similarity search
- Provides both CLI and interactive search interfaces

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Build the index

```bash
python semantic_search.py --index --repo /path/to/java/repo --api-key YOUR_OPENAI_API_KEY
```

Or use the environment variable for the API key:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python semantic_search.py --index --repo /path/to/java/repo
```

### Search with a specific query

```bash
python semantic_search.py --search "implement authentication logic" --top-k 5
```

### Interactive search

```bash
python semantic_search.py
```

## How it works

1. **Parsing**: The tool walks through the Java repository and extracts code units (classes and methods) using the `javalang` parser.

2. **Embedding**: Each code unit is converted into a vector embedding using OpenAI's text-embedding model.

3. **Indexing**: The embeddings are stored in a FAISS vector index for efficient similarity search.

4. **Searching**: When you search, your query is converted to an embedding and compared against the indexed code units to find the most semantically similar matches.

## Notes

- The index is stored in the `.semsearch` directory
- Embeddings are cached to avoid redundant API calls
- The tool requires an OpenAI API key for generating embeddings