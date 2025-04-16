"""
Search module for semantic search.

This module contains the search engine and related functions for semantic search.
"""

import os
import pickle
import datetime
import numpy as np
from typing import List, Dict, Tuple, Optional

from semsearch.models import CodeUnit
from semsearch.embedding import CodeEmbedder
from semsearch.indexing import VectorIndex


class SearchEngine:
    """
    Search engine for semantic code search.
    
    Combines an embedder and a vector index to perform semantic searches.
    """
    
    def __init__(self, embedder, index):
        """
        Initialize the search engine.
        
        Args:
            embedder: CodeEmbedder instance for embedding queries
            index: VectorIndex instance for searching
        """
        self.embedder = embedder
        self.index = index

    def search(self, query: str, top_k: int) -> List[Tuple[CodeUnit, float]]:
        """
        Search for code units matching the query.
        
        Args:
            query: The search query
            top_k: The number of results to return
            
        Returns:
            List of tuples containing (CodeUnit, similarity_score)
        """
        query_vector = self.embedder.embed_query(query)
        return self.index.search(query_vector, top_k)


def get_repo_base_dir(repo_path):
    """
    Get the base directory for a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Base directory path for storing index files
    """
    # Create a safe directory name from the repo path
    repo_name = os.path.basename(os.path.normpath(repo_path))
    # Replace any non-alphanumeric characters with underscores
    safe_name = ''.join(c if c.isalnum() else '_' for c in repo_name)

    # Create a hash of the full path
    import hashlib
    path_hash = hashlib.md5(repo_path.encode('utf-8')).hexdigest()[:8]

    # Return the base directory
    return os.path.join(".semsearch", f"{safe_name}.{path_hash}")


def get_shared_cache_path(repo_path):
    """
    Get the path to the shared cache for a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Path to the shared cache file
    """
    base_dir = get_repo_base_dir(repo_path)
    return os.path.join(base_dir, "shared_cache.pkl")


def get_git_info(repo_path):
    """
    Get git commit hash and status for a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Tuple of (commit_hash, has_changes) or (None, False) if not a git repository
    """
    import subprocess
    import os

    # Check if the repository is a git repository
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        return None, False

    try:
        # Get the current commit hash
        result = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        commit_hash = result.stdout.strip()

        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "-C", repo_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        has_changes = bool(result.stdout.strip())

        return commit_hash, has_changes
    except subprocess.CalledProcessError:
        return None, False
    except Exception as e:
        print(f"Error getting git info: {e}")
        return None, False


def get_index_path(repo_path):
    """
    Generate a path for the index based on the repository path and git commit.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Path to the index directory
    """
    # Get the base directory
    base_dir = get_repo_base_dir(repo_path)

    # Get git commit info
    commit_hash, has_changes = get_git_info(repo_path)

    # If we have a git commit hash, use it for the subdirectory
    if commit_hash:
        if has_changes:
            # For modified versions, append "-latest"
            commit_dir = f"{commit_hash}-latest"
        else:
            commit_dir = commit_hash
        return os.path.join(base_dir, commit_dir)
    else:
        # If not a git repository, use the old structure
        return base_dir


def find_previous_index(repo_path):
    """
    Find a previous index for the same repository that we can reuse.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Path to a previous index directory, or None if none found
    """
    # Get the base directory for this repository
    base_dir = get_repo_base_dir(repo_path)

    # If the base directory doesn't exist, there's no previous index
    if not os.path.exists(base_dir):
        return None

    # Get the current commit hash
    current_commit, _ = get_git_info(repo_path)
    if not current_commit:
        return None

    # Look for indexes from previous commits
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "index.index")):
            # Skip the current commit
            if item == current_commit or item == f"{current_commit}-latest":
                continue

            # Found a previous index
            return item_path

    return None


def load_cache(index_path):
    """
    Load the cache for an index, using shared cache if available.
    
    Args:
        index_path: Path to the index directory
        
    Returns:
        Dictionary mapping content hashes to embedding vectors
    """
    # Check if there's a reference to a shared cache
    cache_ref_path = os.path.join(index_path, "cache_ref.txt")
    if os.path.exists(cache_ref_path):
        try:
            # Parse the reference file to get the shared cache path
            with open(cache_ref_path, "r") as f:
                for line in f:
                    if line.startswith("Using shared cache at:"):
                        shared_cache_path = line.split(":", 1)[1].strip()
                        break
                else:
                    shared_cache_path = None

            # If we found a shared cache path, try to load it
            if shared_cache_path and os.path.exists(shared_cache_path):
                with open(shared_cache_path, "rb") as f:
                    cache = pickle.load(f)
                print(f"Loaded shared embedding cache with {len(cache)} entries")
                return cache
        except Exception as e:
            print(f"Error loading shared cache: {e}")
            # Fall back to local cache

    # Try to load the local cache
    cache_path = os.path.join(index_path, "cache.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded local embedding cache with {len(cache)} entries")
            return cache
        except Exception as e:
            print(f"Error loading local cache: {e}")

    # If all else fails, return an empty cache
    return {}


def update_index_list():
    """
    Update the list of available indexes.
    """
    if not os.path.exists(".semsearch"):
        return

    indexes = []
    # First level: repo directories
    for repo_dir in os.listdir(".semsearch"):
        repo_path = os.path.join(".semsearch", repo_dir)
        if not os.path.isdir(repo_path):
            continue

        # Check if this is an old-style index
        if os.path.exists(os.path.join(repo_path, "info.txt")):
            indexes.append(repo_dir)
            continue

        # Second level: commit directories
        for commit_dir in os.listdir(repo_path):
            commit_path = os.path.join(repo_path, commit_dir)
            if os.path.isdir(commit_path) and os.path.exists(os.path.join(commit_path, "info.txt")):
                # Store as repo_dir/commit_dir
                indexes.append(f"{repo_dir}/{commit_dir}")

    with open(os.path.join(".semsearch", "indexes.txt"), "w") as f:
        for index in indexes:
            f.write(f"{index}\n")


def list_available_indexes():
    """
    List all available indexes.
    
    Returns:
        List of index names
    """
    if not os.path.exists(".semsearch"):
        print("No indexes found.")
        return []

    indexes = []
    # First level: repo directories
    for repo_dir in os.listdir(".semsearch"):
        repo_path = os.path.join(".semsearch", repo_dir)
        if not os.path.isdir(repo_path):
            continue

        # Check if this is an old-style index
        if os.path.exists(os.path.join(repo_path, "info.txt")):
            # Read info file
            with open(os.path.join(repo_path, "info.txt"), "r") as f:
                info = f.read()

            # Parse repo name and hash from the directory name
            parts = repo_dir.split('.')
            if len(parts) > 1:
                name = parts[0]
                hash_part = parts[1]
                display_name = f"{name} ({hash_part})"
            else:
                display_name = repo_dir

            indexes.append((repo_dir, display_name, info))
            continue

        # Second level: commit directories
        for commit_dir in os.listdir(repo_path):
            commit_path = os.path.join(repo_path, commit_dir)
            if os.path.isdir(commit_path) and os.path.exists(os.path.join(commit_path, "info.txt")):
                # Read info file
                with open(os.path.join(commit_path, "info.txt"), "r") as f:
                    info = f.read()

                # Parse repo name and hash from the directory name
                parts = repo_dir.split('.')
                if len(parts) > 1:
                    name = parts[0]
                    hash_part = parts[1]

                    # Check if this is a modified version
                    if commit_dir.endswith("-latest"):
                        commit_display = f"{commit_dir[:-7]} (modified)"
                    else:
                        commit_display = commit_dir[:8]  # Show first 8 chars of commit hash

                    display_name = f"{name} ({hash_part}) @ {commit_display}"
                else:
                    display_name = f"{repo_dir}/{commit_dir}"

                # Store as repo_dir/commit_dir
                indexes.append((f"{repo_dir}/{commit_dir}", display_name, info))

    if not indexes:
        print("No indexes found.")
        return []

    print("\nAvailable indexes:")
    for i, (id_name, display_name, info) in enumerate(indexes):
        print(f"{i+1}. {display_name}")
        for line in info.split("\n"):
            if line.strip():
                print(f"   {line}")
        print()

    return [id_name for id_name, _, _ in indexes]


def search(query, top_k, api_key, index_name=None):
    """
    Search for code units matching a query.
    
    Args:
        query: The search query
        top_k: The number of results to return
        api_key: OpenAI API key
        index_name: Optional name of the index to use
        
    Returns:
        List of tuples containing (CodeUnit, similarity_score)
    """
    import openai
    openai.api_key = api_key

    # List available indexes if none specified
    if index_name is None:
        indexes = list_available_indexes()
        if not indexes:
            return

        if len(indexes) == 1:
            index_name = indexes[0]
            print(f"Using the only available index: {index_name}")
        else:
            while True:
                try:
                    choice = input("\nEnter index number to use (or 'exit' to quit): ")
                    if choice.lower() == 'exit':
                        return
                    index_num = int(choice) - 1
                    if 0 <= index_num < len(indexes):
                        index_name = indexes[index_num]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(indexes)}")
                except ValueError:
                    print("Please enter a valid number")

    # Construct index path
    if "/" in index_name:
        # New-style path: repo_dir/commit_dir
        repo_dir, commit_dir = index_name.split("/", 1)
        index_path = os.path.join(".semsearch", repo_dir, commit_dir)
    else:
        # Old-style path
        index_path = os.path.join(".semsearch", index_name)

    # Load the index and cache
    try:
        index = VectorIndex.load(os.path.join(index_path, "index"))
        cache = load_cache(index_path)
    except FileNotFoundError:
        print(f"Index '{index_name}' not found. Please build the index first.")
        return

    # Use the dimensions from the loaded index
    dimensions = index.dimensions
    print(f"Using embedding dimensions: {dimensions}")

    embedder = CodeEmbedder(cache=cache, dimensions=dimensions)
    engine = SearchEngine(embedder, index)

    results = engine.search(query, top_k)
    display_results(results)
    return results


def display_results(results):
    """
    Display search results in a human-readable format.
    
    Args:
        results: List of tuples containing (CodeUnit, similarity_score)
    """
    for i, (unit, score) in enumerate(results):
        print(f"{i+1}. {unit.path} ({score:.2f})")
        print(f"Type: {unit.unit_type}, Name: {unit.name}")
        if unit.package:
            print(f"Package: {unit.package}")
        if unit.class_name:
            print(f"Class: {unit.class_name}")
        print("-" * 40)
        print(unit.content[:200] + "..." if len(unit.content) > 200 else unit.content)
        print()


def interactive_search(api_key):
    """
    Run an interactive search session.
    
    Args:
        api_key: OpenAI API key
    """
    import openai
    openai.api_key = api_key

    # List available indexes
    indexes = list_available_indexes()
    if not indexes:
        print("No indexes found. Please build an index first.")
        return

    # Select an index
    if len(indexes) == 1:
        index_name = indexes[0]
        print(f"Using the only available index: {index_name}")
    else:
        while True:
            try:
                choice = input("\nEnter index number to use (or 'exit' to quit): ")
                if choice.lower() == 'exit':
                    return
                index_num = int(choice) - 1
                if 0 <= index_num < len(indexes):
                    index_name = indexes[index_num]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(indexes)}")
            except ValueError:
                print("Please enter a valid number")

    # Construct index path
    if "/" in index_name:
        # New-style path: repo_dir/commit_dir
        repo_dir, commit_dir = index_name.split("/", 1)
        index_path = os.path.join(".semsearch", repo_dir, commit_dir)
    else:
        # Old-style path
        index_path = os.path.join(".semsearch", index_name)

    # Load the index and cache
    try:
        index = VectorIndex.load(os.path.join(index_path, "index"))
        cache = load_cache(index_path)
    except FileNotFoundError:
        print(f"Index '{index_name}' not found. Please build the index first.")
        return

    # Use the dimensions from the loaded index
    dimensions = index.dimensions
    print(f"Using embedding dimensions: {dimensions}")

    embedder = CodeEmbedder(cache=cache, dimensions=dimensions)
    engine = SearchEngine(embedder, index)

    while True:
        query = input("Enter search query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        results = engine.search(query, 10)
        display_results(results)