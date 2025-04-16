# Semantic Code Search

A tool for building and querying a semantic search index for code repositories (supports Java and non-Java files).

## Features

- Parses Java source code to extract classes and methods
- Includes non-Java files with their text content
- Includes package/directory information for all files
- Generates semantic embeddings using OpenAI's embedding model
- Builds a vector index for fast similarity search
- Provides both CLI and interactive search interfaces
- Supports a whitelist of file extensions for the top 50 programming languages
- Respects .gitignore patterns to skip ignored files
- Automatically ignores .git directories

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Build the index

```bash
python semantic_search.py --index --repo /path/to/code/repo --api-key YOUR_OPENAI_API_KEY
```

Or use the environment variable for the API key:

```bash
export OPENAI_API_KEY=YOUR_OPENAI_API_KEY
python semantic_search.py --index --repo /path/to/code/repo
```

### Incremental indexing

For repositories that change frequently, you can use incremental indexing to only process files that have changed since the last indexing:

```bash
python semantic_search.py --index --incremental --repo /path/to/code/repo
```

This significantly reduces indexing time for large repositories with minimal changes.

### Dry run mode

If you want to parse all files without creating embeddings (useful for testing or validation):

```bash
python semantic_search.py --index --dry-run --repo /path/to/code/repo
```

This mode traverses the repository, parses all files, and reports detailed statistics about the parsing process, including:
- Total parsing time
- Number of files found, parsed, and skipped
- Files skipped due to extension whitelist or .gitignore patterns
- Folders skipped due to .gitignore patterns or being .git directories
- Parsing errors encountered and their types
- Breakdown of code unit types found

It's useful for validating your configuration and understanding what will be included in the index without actually creating embeddings or building the index.

### Search with a specific query

```bash
python semantic_search.py --search "implement authentication logic" --top-k 5
```

### Interactive search

```bash
python semantic_search.py
```

## How it works

1. **Parsing**: 
   - For Java files: The tool extracts code units (classes and methods) using the `javalang` parser.
   - For non-Java files: The tool includes the entire file content and uses the directory structure as package information.
   - The tool only processes files with extensions in the whitelist (covering the top 50 programming languages).
   - Files and directories matching patterns in .gitignore are skipped.
   - The .git directory is always ignored.
   - Multiple encodings (utf-8, latin-1, cp1252, iso-8859-1) are tried when reading files to handle different character encodings.
   - For Java files that can't be parsed, a fallback mechanism creates a code unit for the entire file to ensure the content is still indexed.
   - Detailed statistics are collected and reported, including parsing time, number of files processed, files/folders skipped, and parsing errors encountered.

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
- Embeddings are cached in a shared repository-wide cache to avoid redundant API calls and reduce storage space
- Each revision's index references the shared cache instead of duplicating embeddings
- File metadata (modification times and sizes) is tracked to support incremental indexing
- Deleted files are automatically detected and removed from the index during both incremental and full indexing
- The tool requires an OpenAI API key for generating embeddings
