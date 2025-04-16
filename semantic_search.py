#!/usr/bin/env python3
#

import os
import argparse
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import javalang
import faiss
import openai
from dotenv import load_dotenv

@dataclass(frozen=True)  # Make it immutable and hashable
class CodeUnit:
    path: str
    content: str
    unit_type: str  # class, method, field, etc.
    name: str
    package: Optional[str] = None
    class_name: Optional[str] = None

    def get_content_hash(self) -> str:
        # Use a more stable hashing method that won't change between sessions
        import hashlib
        return hashlib.md5(self.content.encode('utf-8')).hexdigest()

    def __hash__(self):
        # Create a hash based on all fields to make the class hashable
        return hash((self.path, self.name, self.unit_type, self.package, self.class_name))

    def __eq__(self, other):
        if not isinstance(other, CodeUnit):
            return False
        return (self.path == other.path and 
                self.name == other.name and 
                self.unit_type == other.unit_type and 
                self.package == other.package and 
                self.class_name == other.class_name)

class JavaParser:
    def parse_repository(self, repo_path: str) -> List[CodeUnit]:
        code_units = []

        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()

                        tree = javalang.parse.parse(content)
                        relative_path = os.path.relpath(file_path, repo_path)
                        code_units.extend(self._extract_code_units(tree, relative_path, content))
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")

        return code_units

    def _extract_code_units(self, tree, file_path, content):
        code_units = []
        package_name = None

        # Extract package
        if tree.package:
            package_name = tree.package.name

        # Extract classes
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_start, class_end = self._get_node_positions(content, node)
            class_content = content[class_start:class_end]

            # Limit content size to avoid token limit issues
            if len(class_content) > 8000:  # Roughly 2000 tokens
                class_content = class_content[:8000] + "... [content truncated]"

            code_units.append(CodeUnit(
                path=file_path,
                content=class_content,
                unit_type="class",
                name=node.name,
                package=package_name
            ))

            # Extract methods within the class
            for method_node in node.methods:
                method_start, method_end = self._get_node_positions(content, method_node)
                method_content = content[method_start:method_end]

                # Limit content size to avoid token limit issues
                if len(method_content) > 4000:  # Roughly 1000 tokens
                    method_content = method_content[:4000] + "... [content truncated]"

                code_units.append(CodeUnit(
                    path=file_path,
                    content=method_content,
                    unit_type="method",
                    name=method_node.name,
                    package=package_name,
                    class_name=node.name
                ))

        return code_units

    def _get_node_positions(self, content, node):
        # This is a simplification. In a real implementation, 
        # you would need to determine exact positions in the source code
        # Using javalang's token positions and parsing
        start_line = node.position.line if hasattr(node, 'position') and node.position else 0
        end_line = len(content.splitlines())

        lines = content.splitlines()
        start_pos = sum(len(lines[i]) + 1 for i in range(start_line - 1)) if start_line > 0 else 0
        end_pos = len(content)

        return start_pos, end_pos

class CodeEmbedder:
    def __init__(self, model_name="text-embedding-3-large", cache=None, dimensions=1536):
        self.model_name = model_name
        self.cache = cache or {}
        self.dimensions = dimensions

    def embed_code_units(self, code_units: List[CodeUnit]) -> Dict[CodeUnit, np.ndarray]:
        embeddings = {}
        units_to_embed = []
        texts_to_embed = []

        # Check cache first
        for unit in code_units:
            content_hash = unit.get_content_hash()
            if content_hash in self.cache:
                embeddings[unit] = self.cache[content_hash]
            else:
                units_to_embed.append(unit)
                texts_to_embed.append(unit.content)

        # Batch process remaining units
        if texts_to_embed:
            batch_embeddings = self._get_embeddings(texts_to_embed)

            for unit, embedding in zip(units_to_embed, batch_embeddings):
                embeddings[unit] = embedding
                self.cache[unit.get_content_hash()] = embedding

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self._get_embeddings([query])[0]

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        # Process in batches to avoid token limits
        batch_size = 100  # Adjust based on your average text length
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            try:
                response = openai.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")
                # If a batch is still too large, process one by one
                if "max_tokens" in str(e).lower():
                    print("Batch too large, processing items individually...")
                    for j, text in enumerate(batch_texts):
                        try:
                            response = openai.embeddings.create(
                                model=self.model_name,
                                input=[text]
                            )
                            all_embeddings.append(np.array(response.data[0].embedding))
                            print(f"  Processed item {i+j+1}/{len(texts)}")
                        except Exception as e2:
                            print(f"  Skipping item {i+j+1} due to error: {e2}")
                            # Add a zero vector as placeholder to maintain alignment
                            all_embeddings.append(np.zeros(self.dimensions if hasattr(self, 'dimensions') else 1536))
                else:
                    raise

        return all_embeddings

class VectorIndex:
    def __init__(self, dimensions=1536):
        self.dimensions = dimensions
        self.index = None
        self.id_to_code_unit = {}

    def build_index(self, embeddings: Dict[CodeUnit, np.ndarray]):
        vectors = []
        code_units = []

        for unit, vector in embeddings.items():
            # Check if vector has the expected dimension
            if len(vector) != self.dimensions:
                print(f"Warning: Vector dimension mismatch. Expected {self.dimensions}, got {len(vector)}.")
                # Resize vector to match expected dimensions
                if len(vector) > self.dimensions:
                    vector = vector[:self.dimensions]  # Truncate
                else:
                    # Pad with zeros
                    padded = np.zeros(self.dimensions)
                    padded[:len(vector)] = vector
                    vector = padded

            vectors.append(vector)
            code_units.append(unit)

        if vectors:
            # Create index
            self.index = faiss.IndexFlatL2(self.dimensions)

            vectors_array = np.array(vectors).astype('float32')
            self.index.add(vectors_array)

            self.id_to_code_unit = {i: unit for i, unit in enumerate(code_units)}

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[CodeUnit, float]]:
        query_vector = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # FAISS may return -1 for no results
                continue
            distance = distances[0][i]
            similarity = 1.0 / (1.0 + distance)
            results.append((self.id_to_code_unit[int(idx)], similarity))

        return results

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump(self.id_to_code_unit, f)

    @classmethod
    def load(cls, path: str):
        index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            id_to_code_unit = pickle.load(f)

        instance = cls(dimensions=index.d)
        instance.index = index
        instance.id_to_code_unit = id_to_code_unit
        return instance

class SearchEngine:
    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index

    def search(self, query: str, top_k: int) -> List[Tuple[CodeUnit, float]]:
        query_vector = self.embedder.embed_query(query)
        return self.index.search(query_vector, top_k)

def get_git_info(repo_path):
    """Get git commit hash and status for a repository"""
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

def get_repo_base_dir(repo_path):
    """Get the base directory for a repository"""
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
    """Get the path to the shared cache for a repository"""
    base_dir = get_repo_base_dir(repo_path)
    return os.path.join(base_dir, "shared_cache.pkl")

def get_index_path(repo_path):
    """Generate a path for the index based on the repository path and git commit"""
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
    """Find a previous index for the same repository that we can reuse"""
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
    """Load the cache for an index, using shared cache if available"""
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

def build_index(repo_path, api_key, incremental=False):
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

    print(f"Parsing Java files in {repo_path}...")
    parser = JavaParser()

    # If incremental, we'll collect code units differently
    if incremental and existing_index_exists and index is not None:
        # Get all Java files in the repository
        all_java_files = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.java'):
                    file_path = os.path.join(root, file)
                    all_java_files.append(file_path)

        # Determine which files are new or modified
        changed_files = []
        for file_path in all_java_files:
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

        print(f"Found {len(all_java_files)} Java files: {len(changed_files)} new/modified, {len(deleted_files)} deleted")

        # Parse only the changed files
        new_code_units = []
        for file_path in changed_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = javalang.parse.parse(content)
                relative_path = os.path.relpath(file_path, repo_path)
                new_code_units.extend(parser._extract_code_units(tree, relative_path, content))
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

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
        # For full indexing, parse all files
        code_units = parser.parse_repository(repo_path)

        # Collect metadata for all files
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.java'):
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
        import datetime
        indexed_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"Indexed on: {indexed_time}\n")
        f.write(f"Code units: {len(code_units)}\n")
        f.write(f"Dimensions: {dimensions}\n")
        f.write(f"Incremental: {incremental}\n")

    # Also save a list of available indexes to .semsearch/indexes.txt
    update_index_list()

    print(f"Index built successfully at {index_path}")

def update_index_list():
    """Update the list of available indexes"""
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
    """List all available indexes"""
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

def interactive_search(api_key):
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

def display_results(results):
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

def main():
    # Load API key from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Semantic search for Java code repositories")
    parser.add_argument("--index", action="store_true", help="Build the search index")
    parser.add_argument("--incremental", action="store_true", help="Perform incremental indexing (only index changed files)")
    parser.add_argument("--search", help="Search query")
    parser.add_argument("--repo", type=str, help="Path to the Java repository")
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
        build_index(args.repo, args.api_key, incremental=args.incremental)
    elif args.search:
        search(args.search, args.top_k, args.api_key, args.index_name)
    else:
        interactive_search(args.api_key)

if __name__ == "__main__":
    main()
