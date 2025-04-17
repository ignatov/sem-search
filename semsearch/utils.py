"""
Utilities module for semantic search.

This module contains utility functions for building indexes and other operations.
"""

import os
import time
import pickle
import datetime
import openai
from typing import Dict, List, Optional

from semsearch.models import CodeUnit
from semsearch.parsers import UnifiedParser
from semsearch.embedding import CodeEmbedder
from semsearch.indexing import VectorIndex
from semsearch.search import (
    get_repo_base_dir, get_shared_cache_path, get_index_path,
    find_previous_index, load_cache, update_index_list
)


def build_index(repo_path, api_key, incremental=False, dry_run=False):
    """
    Build a search index for a repository.

    Args:
        repo_path: Path to the repository
        api_key: OpenAI API key
        incremental: Whether to perform incremental indexing
        dry_run: Whether to only parse files without creating embeddings
    """
    openai.api_key = api_key

    # Get the index path for this specific repository
    index_path = get_index_path(repo_path)

    # Check if we're doing an incremental update and if the index exists
    existing_index_exists = os.path.exists(os.path.join(index_path, "index.index"))
    existing_cache = {}
    existing_file_metadata = {}
    existing_dimensions = 1536  # Default

    # If the index doesn't exist but we want incremental, try to find a previous index to reuse
    previous_index_path = None
    if incremental and not existing_index_exists:
        previous_index_path = find_previous_index(repo_path)
        if previous_index_path:
            print(f"Found previous index at {previous_index_path} that we can reuse")
            existing_index_exists = True

    # Check for shared cache
    shared_cache_path = get_shared_cache_path(repo_path)
    shared_cache = {}
    if os.path.exists(shared_cache_path):
        try:
            with open(shared_cache_path, "rb") as f:
                shared_cache = pickle.load(f)
            print(f"Loaded shared embedding cache with {len(shared_cache)} entries")
        except Exception as e:
            print(f"Error loading shared cache: {e}")
            shared_cache = {}

    # Determine which path to use for loading existing data
    load_path = previous_index_path if previous_index_path and incremental else index_path

    if incremental and existing_index_exists:
        print(f"Performing incremental update of index for {repo_path}...")

        # Load existing cache
        cache_path = os.path.join(load_path, "cache.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    existing_cache = pickle.load(f)
                print(f"Loaded existing embedding cache with {len(existing_cache)} entries")
            except Exception as e:
                print(f"Error loading cache: {e}")
                existing_cache = {}

        # Load existing index to get dimensions
        try:
            index = VectorIndex.load(os.path.join(load_path, "index"))
            existing_dimensions = index.dimensions
            print(f"Loaded existing index with dimensions: {existing_dimensions}")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            index = None
    else:
        print(f"Building new index for {repo_path}...")
        index = None

    # Load file metadata (for change detection) - do this for both incremental and full indexing
    metadata_path = os.path.join(load_path, "file_metadata.pkl")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "rb") as f:
                existing_file_metadata = pickle.load(f)
            print(f"Loaded metadata for {len(existing_file_metadata)} files")
        except Exception as e:
            print(f"Error loading file metadata: {e}")
            existing_file_metadata = {}

    # Merge existing cache with shared cache
    # Shared cache takes precedence as it might be more up-to-date
    combined_cache = {**existing_cache, **shared_cache}

    # Collect current file metadata for change detection
    current_file_metadata = {}

    print(f"Parsing files in {repo_path}...")
    unified_parser = UnifiedParser()

    # If incremental, we'll collect code units differently
    if incremental and existing_index_exists and index is not None:
        # Get all files in the repository
        all_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        # Determine which files are new or modified
        changed_files = []
        for file_path in all_files:
            relative_path = os.path.relpath(file_path, repo_path)

            # Get file stats
            stat = os.stat(file_path)
            mtime = stat.st_mtime
            size = stat.st_size

            # Store current metadata
            current_file_metadata[relative_path] = {
                'mtime': mtime,
                'size': size
            }

            # Check if file is new or modified
            if relative_path not in existing_file_metadata or \
               existing_file_metadata[relative_path]['mtime'] != mtime or \
               existing_file_metadata[relative_path]['size'] != size:
                changed_files.append(file_path)

        # Check for deleted files
        deleted_files = []
        for relative_path in existing_file_metadata:
            full_path = os.path.join(repo_path, relative_path)
            if not os.path.exists(full_path):
                deleted_files.append(relative_path)

        print(f"Found {len(all_files)} files: {len(changed_files)} new/modified, {len(deleted_files)} deleted")

        # Parse only the changed files
        new_code_units = []

        # Process all changed files in a single loop
        for file_path in changed_files:
            relative_path = os.path.relpath(file_path, repo_path)

            # Get directory structure as package name
            dir_path = os.path.dirname(relative_path)
            package_name = dir_path.replace(os.path.sep, '.') if dir_path else None

            # Choose the appropriate parser based on file extension
            if file_path.endswith('.java'):
                new_code_units.extend(unified_parser.parse_java_file(file_path, repo_path))
            else:
                new_code_units.extend(unified_parser.generic_parser.parse_file(file_path, repo_path, unified_parser.stats))

        print(f"Extracted {len(new_code_units)} code units from changed files")

        # Get existing code units (excluding those from deleted or changed files)
        existing_code_units = []
        for i, unit in enumerate(index.id_to_code_unit.values()):
            # Skip units from deleted files
            if any(unit.path.startswith(deleted_path) for deleted_path in deleted_files):
                continue

            # Skip units from changed files
            if any(os.path.join(repo_path, unit.path) == changed_file for changed_file in changed_files):
                continue

            existing_code_units.append(unit)

        print(f"Keeping {len(existing_code_units)} unchanged code units")

        # Combine existing and new code units
        code_units = existing_code_units + new_code_units
    else:
        # For full indexing, parse all files with the unified parser
        code_units = unified_parser.parse_repository(repo_path)

        # Collect metadata for all files
        for root, _, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)

                # Get file stats
                stat = os.stat(file_path)
                mtime = stat.st_mtime
                size = stat.st_size

                # Store current metadata
                current_file_metadata[relative_path] = {
                    'mtime': mtime,
                    'size': size
                }

        # Check for deleted files (do this for full indexing too)
        if existing_file_metadata:
            deleted_files = []
            for relative_path in existing_file_metadata:
                full_path = os.path.join(repo_path, relative_path)
                if not os.path.exists(full_path):
                    deleted_files.append(relative_path)

            if deleted_files:
                print(f"Found {len(deleted_files)} deleted files since last indexing")
                # No need to do anything else for full indexing as we're rebuilding the index from scratch
                # and only including files that currently exist

    print(f"Total code units to index: {len(code_units)}")

    # Print parsing statistics
    stats = unified_parser.stats
    print("\nParsing statistics:")
    print(f"  Total files found: {stats['total_files']}")
    print(f"  Files parsed: {stats['parsed_files']}")
    print(f"  Files skipped due to extension: {stats['skipped_files_extension']}")
    print(f"  Files skipped due to .gitignore: {stats['skipped_files_gitignore']}")
    print(f"  Folders skipped due to .gitignore: {stats['skipped_folders_gitignore']}")
    print(f"  .git folders skipped: {stats['skipped_folders_git']}")
    print(f"  Parsing errors: {stats['parsing_errors']}")
    print(f"  Total parsing time: {stats['parsing_time']:.2f} seconds")

    # Print error details if any
    if stats['parsing_errors'] > 0 and stats['parsing_errors_details']:
        print("\nParsing error types:")
        for error_type, count in stats['parsing_errors_details'].items():
            print(f"  {error_type}: {count}")

    # If dry_run is True, skip embedding creation and index building
    if dry_run:
        print("Dry run mode: Successfully parsed all files without creating embeddings.")
        print(f"Found {len(code_units)} code units.")

        # Print some statistics about the code units
        unit_types = {}
        for unit in code_units:
            unit_type = unit.unit_type
            if unit_type in unit_types:
                unit_types[unit_type] += 1
            else:
                unit_types[unit_type] = 1

        print("\nCode unit types:")
        for unit_type, count in unit_types.items():
            print(f"  {unit_type}: {count}")

        # Print code unit size statistics
        code_unit_sizes = unified_parser.stats['code_unit_sizes']
        total_size = code_unit_sizes['total']
        print(f"\nCode unit size statistics:")
        print(f"  Total size of all code units: {total_size:,} characters")

        if code_unit_sizes['by_type']:
            print("\nSize by unit type:")
            for unit_type, data in code_unit_sizes['by_type'].items():
                count = data['count']
                size = data['size']
                avg_size = size / count if count > 0 else 0
                print(f"  {unit_type}: {size:,} characters total, {avg_size:.2f} average per unit")

        # Print parsing statistics
        stats = unified_parser.stats
        print("\nParsing statistics:")
        print(f"  Total files found: {stats['total_files']}")
        print(f"  Files parsed: {stats['parsed_files']}")
        print(f"  Files skipped due to extension: {stats['skipped_files_extension']}")
        print(f"  Files skipped due to .gitignore: {stats['skipped_files_gitignore']}")
        print(f"  Folders skipped due to .gitignore: {stats['skipped_folders_gitignore']}")
        print(f"  .git folders skipped: {stats['skipped_folders_git']}")
        print(f"  Parsing errors: {stats['parsing_errors']}")
        print(f"  Total parsing time: {stats['parsing_time']:.2f} seconds")

        # Print error details if any
        if stats['parsing_errors'] > 0 and stats['parsing_errors_details']:
            print("\nParsing error types:")
            for error_type, count in stats['parsing_errors_details'].items():
                print(f"  {error_type}: {count}")

        # Print skipped folders (limit to 10 for readability)
        if stats['skipped_folders']:
            print("\nSkipped folders (up to 10):")
            for folder in list(stats['skipped_folders'])[:10]:
                print(f"  {folder}")
            if len(stats['skipped_folders']) > 10:
                print(f"  ... and {len(stats['skipped_folders']) - 10} more")

        return

    # Determine embedding dimensions
    dimensions = existing_dimensions
    if not incremental or not existing_index_exists:
        # Get a sample embedding to determine dimensions
        try:
            sample_response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=["Sample text to determine embedding dimensions"]
            )
            dimensions = len(sample_response.data[0].embedding)
            print(f"Using embedding dimensions: {dimensions}")
        except Exception as e:
            print(f"Error determining embedding dimensions: {e}")
            dimensions = 1536  # Default to 1536 for text-embedding-3-large
            print(f"Defaulting to {dimensions} dimensions")

    # Initialize embedder with combined cache
    embedder = CodeEmbedder(cache=combined_cache, dimensions=dimensions)
    embeddings = embedder.embed_code_units(code_units)

    print("Building index...")
    index = VectorIndex(dimensions=dimensions)
    index.build_index(embeddings)

    # Save the index
    os.makedirs(index_path, exist_ok=True)
    index.save(os.path.join(index_path, "index"))

    # Update and save the shared cache
    shared_cache.update(embedder.cache)
    os.makedirs(os.path.dirname(shared_cache_path), exist_ok=True)
    with open(shared_cache_path, "wb") as f:
        pickle.dump(shared_cache, f)

    # Save a reference to the shared cache in the index directory
    cache_ref_path = os.path.join(index_path, "cache_ref.txt")
    with open(cache_ref_path, "w") as f:
        f.write(f"Using shared cache at: {shared_cache_path}\n")
        f.write(f"Cache entries: {len(shared_cache)}\n")

    # Save the file metadata for future incremental updates
    metadata_path = os.path.join(index_path, "file_metadata.pkl")
    with open(metadata_path, "wb") as f:
        pickle.dump(current_file_metadata, f)

    # Save the repo path info for later reference
    info_path = os.path.join(index_path, "info.txt")
    with open(info_path, "w") as f:
        f.write(f"Repository: {repo_path}\n")
        # Use proper datetime import and formatting
        indexed_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Indexed on: {indexed_time}\n")
        f.write(f"Code units: {len(code_units)}\n")
        f.write(f"Dimensions: {dimensions}\n")
        f.write(f"Incremental: {incremental}\n")

    # Also save a list of available indexes to .semsearch/indexes.txt
    update_index_list()

    print(f"Index built successfully at {index_path}")
