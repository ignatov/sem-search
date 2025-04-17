"""
Main module for semantic search command-line interface.

This module provides the command-line interface for the semantic search system.
"""

import os
import argparse
from dotenv import load_dotenv

from semsearch.utils import build_index
from semsearch.search import search, interactive_search, list_available_indexes


def main():
    """
    Main entry point for the semantic search command-line interface.
    """
    # Load API key from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Semantic search for code repositories (supports Java and non-Java files)")
    parser.add_argument("--index", action="store_true", help="Build the search index")
    parser.add_argument("--incremental", action="store_true", help="Perform incremental indexing (only index changed files)")
    parser.add_argument("--dry-run", action="store_true", help="Parse files without creating embeddings (dry run)")
    parser.add_argument("--no-report", action="store_true", help="Skip generating HTML report")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--repo", type=str, help="Path to the code repository")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--list-indexes", action="store_true", help="List available indexes")
    parser.add_argument("--index-name", type=str, help="Specify which index to use for search")

    args = parser.parse_args()

    if not args.api_key:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        if not args.api_key and not args.list_indexes:
            print("Error: OpenAI API key required. Provide it with --api-key, in .env file, or set OPENAI_API_KEY environment variable.")
            return

    if args.list_indexes:
        list_available_indexes()
    elif args.index:
        if not args.repo:
            print("Error: Repository path required for indexing.")
            return
        build_index(args.repo, args.api_key, incremental=args.incremental, dry_run=args.dry_run, generate_report=not args.no_report)
    elif args.search:
        search(args.search, args.top_k, args.api_key, args.index_name)
    else:
        interactive_search(args.api_key)


if __name__ == "__main__":
    main()
