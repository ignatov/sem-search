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

### Incremental indexing

For repositories that change frequently, you can use incremental indexing to only process files that have changed since the last indexing:

```bash
python semantic_search.py --index --incremental --repo /path/to/java/repo
```

This significantly reduces indexing time for large repositories with minimal changes.

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
   - With incremental indexing, only changed files are processed, and the existing index is updated rather than rebuilt from scratch.

4. **Searching**: When you search, your query is converted to an embedding and compared against the indexed code units to find the most semantically similar matches.

## Notes

- The index is stored in the `.semsearch` directory with the following structure:
  - `.semsearch/reponame.<repo_path_hash>/<git_commit_hash>` for Git repositories
  - For modified working copies, the structure is `.semsearch/reponame.<repo_path_hash>/<git_commit_hash>-latest`
  - For non-Git repositories, the old structure `.semsearch/reponame.<repo_path_hash>` is used
- When building an index for a new commit, the tool will attempt to reuse data from previous commits to speed up indexing
- Embeddings are cached to avoid redundant API calls
- File metadata (modification times and sizes) is tracked to support incremental indexing
- The tool requires an OpenAI API key for generating embeddings
