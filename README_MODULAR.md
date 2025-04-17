# Modular Semantic Search

This is a modular version of the semantic search system. The code has been reorganized into a package structure to improve maintainability and testability.

## Package Structure

The code is now organized into the following modules:

- `semsearch/models.py`: Contains the `CodeUnit` class, which represents a unit of code for semantic search.
- `semsearch/parsers.py`: Contains the parsing classes (`GenericFileParser`, `UnifiedParser`, `TreeSitterParser`) for extracting code units from files.
- `semsearch/embedding.py`: Contains the `CodeEmbedder` class for embedding code units into vectors.
- `semsearch/indexing.py`: Contains the `VectorIndex` class for indexing and searching vectors.
- `semsearch/search.py`: Contains the `SearchEngine` class and search-related functions.
- `semsearch/utils.py`: Contains utility functions for building indexes and other operations.
- `semsearch/main.py`: Contains the command-line interface for the semantic search system.
- `semsearch/__init__.py`: Re-exports the public API for backward compatibility.

## Backward Compatibility

The modular version maintains backward compatibility with the original code. The `semantic_search.py` file now imports from the `semsearch` package and calls the `main()` function. This means that existing scripts and tools that use the original code will continue to work.

The `web_server.py` file has been updated to import from the `semsearch` package instead of `semantic_search`.

## Tests

The modular version includes a comprehensive test suite. The tests are organized into the following files:

- `tests/test_models.py`: Tests for the `CodeUnit` class.
- `tests/test_parsers.py`: Tests for the parsing classes.
- `tests/test_embedding.py`: Tests for the `CodeEmbedder` class.
- `tests/test_indexing.py`: Tests for the `VectorIndex` class.
- `tests/test_search.py`: Tests for the `SearchEngine` class.
- `tests/run_tests.py`: A script to run all the tests.

To run the tests, use the following command:

```bash
python tests/run_tests.py
```

## Usage

The modular version can be used in the same way as the original code. For example, to build an index:

```bash
python semantic_search.py --index --repo /path/to/repo
```

To search for code:

```bash
python semantic_search.py --search "query"
```

To run the web server:

```bash
python web_server.py
```

## Development

When developing new features or fixing bugs, you should now work with the modular code in the `semsearch` package. The `semantic_search.py` file should not be modified directly, as it is now just a thin wrapper around the modular code.

To add new functionality, you should:

1. Identify the appropriate module for the new code.
2. Add the new code to the module.
3. Update the `__init__.py` file if the new code should be part of the public API.
4. Write tests for the new code.
5. Run the tests to ensure everything works correctly.
