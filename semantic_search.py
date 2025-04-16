#!/usr/bin/env python3
"""
Entry point for semantic search command-line interface.

This script is a thin wrapper around the modular semantic search system.
It maintains backward compatibility with the original semantic_search.py.
"""

from semsearch.main import main

if __name__ == "__main__":
    main()