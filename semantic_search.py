#!/usr/bin/env python3
#

import os
import argparse
import pickle
import numpy as np
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import javalang
import faiss
import openai
import pathspec
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
    def parse_file(self, file_path: str, repo_path: str) -> List[CodeUnit]:
        """Parse a single Java file and extract code units."""
        # This is now just a wrapper around UnifiedParser for backward compatibility
        unified_parser = UnifiedParser()
        return unified_parser.parse_java_file(file_path, repo_path)

class GenericFileParser:
    def parse_file(self, file_path: str, repo_path: str, stats=None) -> List[CodeUnit]:
        """Parse a single non-Java file and extract code units."""
        code_units = []

        # Try different encodings in order of likelihood
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                relative_path = os.path.relpath(file_path, repo_path)

                # Use directory structure as package name
                dir_path = os.path.dirname(relative_path)
                package = dir_path.replace(os.path.sep, '.') if dir_path else None

                # Create a code unit for the entire file
                code_units.append(CodeUnit(
                    path=relative_path,
                    content=content,
                    unit_type="file",
                    name=os.path.basename(file_path),
                    package=package
                ))

                # If we got here, we successfully read the file
                if encoding != 'utf-8':
                    print(f"Successfully read {file_path} using {encoding} encoding")

                break  # Exit the loop if successful

            except UnicodeDecodeError as e:
                # If this is the last encoding we tried, report the error
                if encoding == encodings[-1]:
                    error_msg = f"Error reading file {file_path}: Unable to decode with any of the attempted encodings"
                    print(error_msg)

                    # Track the error in statistics if stats is provided
                    if stats is not None:
                        stats['parsing_errors'] += 1
                        error_type = "UnicodeDecodeError"
                        if error_type in stats['parsing_errors_details']:
                            stats['parsing_errors_details'][error_type] += 1
                        else:
                            stats['parsing_errors_details'][error_type] = 1
                # Otherwise, try the next encoding
                continue

            except Exception as e:
                error_msg = f"Error reading file {file_path}: {str(e)}"
                print(error_msg)

                # Track the error in statistics if stats is provided
                if stats is not None:
                    stats['parsing_errors'] += 1
                    error_type = type(e).__name__
                    if error_type in stats['parsing_errors_details']:
                        stats['parsing_errors_details'][error_type] += 1
                    else:
                        stats['parsing_errors_details'][error_type] = 1

                break  # Exit the loop for non-encoding errors

        return code_units

class UnifiedParser:
    def __init__(self):
        self.generic_parser = GenericFileParser()
        # Whitelist of extensions for the top 50 programming languages
        self.supported_extensions = {
            # Programming Languages
            '.py', '.java', '.js', '.jsx', '.ts', '.tsx', '.c', '.cpp', '.cc', '.cxx', '.h', '.hpp',
            '.cs', '.go', '.rb', '.php', '.swift', '.kt', '.rs', '.scala', '.m', '.mm', '.pl', '.pm',
            '.sh', '.bash', '.ps1', '.psm1', '.lua', '.groovy', '.dart', '.r', '.d', '.f', '.f90',
            '.jl', '.clj', '.cljs', '.erl', '.ex', '.exs', '.elm', '.hs', '.ml', '.mli', '.fs', '.fsx',
            '.lisp', '.cl', '.scm', '.rkt', '.v', '.vhd', '.vhdl', '.asm', '.s',
            # Web Development
            '.html', '.htm', '.css', '.scss', '.sass', '.less', '.json', '.xml', '.yaml', '.yml',
            # Configuration and Data
            '.toml', '.ini', '.cfg', '.conf', '.properties', '.md', '.markdown', '.rst', '.tex',
            '.sql', '.graphql', '.proto', '.plist',
            # Other common code files
            '.gradle', '.cmake', '.make', '.dockerfile', '.tf', '.ipynb'
        }

        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'parsed_files': 0,
            'skipped_files_extension': 0,
            'skipped_files_gitignore': 0,
            'skipped_folders_gitignore': 0,
            'skipped_folders_git': 0,
            'skipped_folders': set(),  # Store actual folder names
            'parsing_time': 0,
            'parsing_errors': 0,
            'parsing_errors_details': {}  # Store error types and counts
        }

    def _load_gitignore(self, repo_path: str) -> pathspec.PathSpec:
        """Load .gitignore patterns from the repository."""
        gitignore_path = os.path.join(repo_path, '.gitignore')
        patterns = []

        # Always ignore .git directory
        patterns.append('.git/')

        # Add patterns from .gitignore if it exists
        if os.path.exists(gitignore_path):
            try:
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        # Skip empty lines and comments
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception as e:
                print(f"Error reading .gitignore file: {e}")

        return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

    def parse_repository(self, repo_path: str) -> List[CodeUnit]:
        """Walk through the repository once and parse all files."""
        code_units = []

        # Reset statistics
        self.stats = {
            'total_files': 0,
            'parsed_files': 0,
            'skipped_files_extension': 0,
            'skipped_files_gitignore': 0,
            'skipped_folders_gitignore': 0,
            'skipped_folders_git': 0,
            'skipped_folders': set(),  # Store actual folder names
            'parsing_time': 0,
            'parsing_errors': 0,
            'parsing_errors_details': {}  # Store error types and counts
        }

        # Start timing
        start_time = time.time()

        # Load .gitignore patterns
        gitignore_spec = self._load_gitignore(repo_path)

        for root, dirs, files in os.walk(repo_path):
            # Skip .git directory (modify dirs in-place to prevent os.walk from descending into it)
            if '.git' in dirs:
                dirs.remove('.git')
                self.stats['skipped_folders_git'] += 1
                self.stats['skipped_folders'].add(os.path.join(root, '.git'))

            # Get relative path for gitignore matching
            rel_root = os.path.relpath(root, repo_path)
            if rel_root == '.':
                rel_root = ''

            # Filter out ignored directories
            for d in list(dirs):  # Create a copy to modify during iteration
                rel_path = os.path.join(rel_root, d)
                if gitignore_spec.match_file(rel_path) or gitignore_spec.match_file(rel_path + '/'):
                    dirs.remove(d)
                    self.stats['skipped_folders_gitignore'] += 1
                    self.stats['skipped_folders'].add(os.path.join(root, d))

            # Count total files before filtering
            self.stats['total_files'] += len(files)

            for file in files:
                # Get file's relative path for gitignore matching
                rel_path = os.path.join(rel_root, file)

                # Skip files that match gitignore patterns
                if gitignore_spec.match_file(rel_path):
                    self.stats['skipped_files_gitignore'] += 1
                    continue

                # Check if file extension is in the whitelist
                _, ext = os.path.splitext(file.lower())
                if ext not in self.supported_extensions:
                    self.stats['skipped_files_extension'] += 1
                    continue

                file_path = os.path.join(root, file)

                # Choose the appropriate parser based on file extension
                if file.endswith('.java'):
                    code_units.extend(self.parse_java_file(file_path, repo_path))
                else:
                    # Include non-Java files in the index
                    code_units.extend(self.generic_parser.parse_file(file_path, repo_path, self.stats))

                self.stats['parsed_files'] += 1

        # End timing
        end_time = time.time()
        self.stats['parsing_time'] = end_time - start_time

        return code_units

    def parse_java_file(self, file_path: str, repo_path: str) -> List[CodeUnit]:
        """Parse a single Java file and extract code units."""
        code_units = []

        # Try different encodings in order of likelihood
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        content = None

        # First, try to read the file with different encodings
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # If we got here, we successfully read the file
                if encoding != 'utf-8':
                    print(f"Successfully read Java file {file_path} using {encoding} encoding")

                break  # Exit the loop if successful

            except UnicodeDecodeError:
                # If this is the last encoding we tried, report the error
                if encoding == encodings[-1]:
                    error_msg = f"Error reading Java file {file_path}: Unable to decode with any of the attempted encodings"
                    print(error_msg)

                    # Track the error in statistics
                    self.stats['parsing_errors'] += 1
                    error_type = "UnicodeDecodeError"
                    if error_type in self.stats['parsing_errors_details']:
                        self.stats['parsing_errors_details'][error_type] += 1
                    else:
                        self.stats['parsing_errors_details'][error_type] = 1

                    return code_units  # Return empty list if we can't read the file
                # Otherwise, try the next encoding
                continue

            except Exception as e:
                error_msg = f"Error reading Java file {file_path}: {str(e)}"
                print(error_msg)

                # Track the error in statistics
                self.stats['parsing_errors'] += 1
                error_type = type(e).__name__
                if error_type in self.stats['parsing_errors_details']:
                    self.stats['parsing_errors_details'][error_type] += 1
                else:
                    self.stats['parsing_errors_details'][error_type] = 1

                return code_units  # Return empty list for non-encoding errors

        # If we couldn't read the file, return empty list
        if content is None:
            return code_units

        # Now try to parse the Java content
        try:
            tree = javalang.parse.parse(content)
            relative_path = os.path.relpath(file_path, repo_path)
            code_units.extend(self._extract_java_code_units(tree, relative_path, content))
        except Exception as e:
            error_msg = f"Error parsing Java file {file_path}: {str(e)}"
            print(error_msg)

            # Track the error in statistics
            self.stats['parsing_errors'] += 1
            error_type = type(e).__name__
            if error_type in self.stats['parsing_errors_details']:
                self.stats['parsing_errors_details'][error_type] += 1
            else:
                self.stats['parsing_errors_details'][error_type] = 1

            # For Java parsing errors, create a code unit for the entire file
            # so we at least have the content indexed
            try:
                relative_path = os.path.relpath(file_path, repo_path)

                # Use directory structure as package name
                dir_path = os.path.dirname(relative_path)
                package = dir_path.replace(os.path.sep, '.') if dir_path else None

                # Create a code unit for the entire file
                code_units.append(CodeUnit(
                    path=relative_path,
                    content=content,
                    unit_type="file",
                    name=os.path.basename(file_path),
                    package=package
                ))

                print(f"Created a fallback code unit for {file_path}")
            except Exception as fallback_error:
                print(f"Error creating fallback code unit for {file_path}: {str(fallback_error)}")

        return code_units

    def _extract_java_code_units(self, tree, file_path, content):
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

        # Ensure all texts are valid strings
        valid_texts = []
        for text in texts:
            if text is None:
                valid_texts.append("")  # Replace None with empty string
            elif isinstance(text, str):
                valid_texts.append(text)
            else:
                # Convert non-string values to strings
                try:
                    valid_texts.append(str(text))
                except:
                    valid_texts.append("")

        texts = valid_texts

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
            try:
                # Ensure each text in the batch is a valid string
                cleaned_batch = []
                for idx, text in enumerate(batch_texts):
                    # Remove any characters that might cause issues with the API
                    if text is not None:
                        # Ensure text is not too long (OpenAI has token limits)
                        if len(text) > 25000:  # Approximate limit
                            text = text[:25000]

                        # Check for potentially problematic items (e.g., item 476)
                        item_number = i + idx + 1
                        if item_number == 476:
                            print(f"  Pre-emptively applying aggressive sanitization to known problematic item {item_number}")
                            text = self._sanitize_text(text, aggressive=True)

                            # Log details for debugging
                            print(f"  After aggressive sanitization:")
                            print(f"  Text length: {len(text)}")
                            if len(text) > 0:
                                print(f"  First 50 chars: {repr(text[:50])}")
                                print(f"  Last 50 chars: {repr(text[-50:] if len(text) >= 50 else text)}")
                        else:
                            # Standard sanitization for other items
                            text = self._sanitize_text(text)

                        # Skip empty texts
                        if not text.strip():
                            print(f"  Replacing empty item {item_number} with placeholder text")
                            text = "placeholder_text_for_embedding"

                        cleaned_batch.append(text)
                    else:
                        cleaned_batch.append("")  # Empty string for None values

                response = openai.embeddings.create(
                    model=self.model_name,
                    input=cleaned_batch
                )
                batch_embeddings = [np.array(data.embedding) for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"Error in batch {i//batch_size + 1}: {e}")

                # If it's an input validation error, try with more aggressive sanitization first
                if "$.input" in str(e):
                    print("Invalid input in batch, attempting with aggressive sanitization...")
                    try:
                        # Apply aggressive sanitization to all items in the batch
                        aggressive_batch = []
                        for idx, text in enumerate(batch_texts):
                            if text is not None:
                                # Truncate and sanitize aggressively
                                if len(text) > 25000:
                                    text = text[:25000]
                                text = self._sanitize_text(text, aggressive=True)

                                # Ensure text is not empty
                                if not text.strip():
                                    text = "placeholder_text_for_embedding"

                                aggressive_batch.append(text)
                            else:
                                aggressive_batch.append("")

                        # Try the API call with aggressively sanitized batch
                        response = openai.embeddings.create(
                            model=self.model_name,
                            input=aggressive_batch
                        )
                        batch_embeddings = [np.array(data.embedding) for data in response.data]
                        all_embeddings.extend(batch_embeddings)
                        print(f"Successfully processed batch with aggressive sanitization")
                        continue  # Skip the individual processing
                    except Exception as batch_retry_error:
                        print(f"Batch still failed after aggressive sanitization: {batch_retry_error}")
                        # Fall back to individual processing

                # If a batch is still too large or has invalid input, process one by one
                if "max_tokens" in str(e).lower() or "$.input" in str(e):
                    print("Batch too large or invalid input, processing items individually...")
                    for j, text in enumerate(batch_texts):
                        try:
                            if text is None or not isinstance(text, str):
                                text = ""
                            # Truncate very long texts
                            if len(text) > 25000:
                                text = text[:25000]
                            # Log problematic text details for debugging
                            if i+j+1 == 476 or "$.input" in str(e):
                                print(f"  Debugging item {i+j+1} before sanitization:")
                                print(f"  Text type: {type(text)}")
                                print(f"  Text length: {len(text) if isinstance(text, str) else 'N/A'}")
                                if isinstance(text, str) and len(text) > 0:
                                    print(f"  First 50 chars: {repr(text[:50])}")
                                    print(f"  Last 50 chars: {repr(text[-50:] if len(text) >= 50 else text)}")

                            # Apply more aggressive sanitization for individual processing
                            original_text = text
                            text = self._sanitize_text(text, aggressive=True)

                            # Log after sanitization
                            if i+j+1 == 476 or "$.input" in str(e):
                                print(f"  After sanitization:")
                                print(f"  Text length: {len(text)}")
                                if len(text) > 0:
                                    print(f"  First 50 chars: {repr(text[:50])}")
                                    print(f"  Last 50 chars: {repr(text[-50:] if len(text) >= 50 else text)}")

                                # Check for specific problematic patterns
                                import re
                                surrogate_pairs = re.findall(r'[\uD800-\uDFFF]', original_text)
                                if surrogate_pairs:
                                    print(f"  Found {len(surrogate_pairs)} surrogate pairs")

                                control_chars = re.findall(r'[\x00-\x1F\x7F-\x9F]', original_text)
                                if control_chars:
                                    print(f"  Found {len(control_chars)} control characters")

                            # Skip empty texts
                            if not text.strip():
                                print(f"  Skipping empty item {i+j+1}")
                                all_embeddings.append(np.zeros(self.dimensions if hasattr(self, 'dimensions') else 1536))
                                continue

                            try:
                                # First attempt with current sanitization
                                response = openai.embeddings.create(
                                    model=self.model_name,
                                    input=[text]
                                )
                                all_embeddings.append(np.array(response.data[0].embedding))
                                print(f"  Processed item {i+j+1}/{len(texts)}")
                            except Exception as api_error:
                                if "$.input" in str(api_error):
                                    print(f"  API error with item {i+j+1}, attempting extreme sanitization: {api_error}")

                                    # Extreme sanitization - only keep basic ASCII letters, numbers, and spaces
                                    extreme_text = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in text)
                                    extreme_text = re.sub(r'\s+', ' ', extreme_text).strip()

                                    if not extreme_text:
                                        extreme_text = "placeholder_text_for_embedding"

                                    print(f"  After extreme sanitization: {repr(extreme_text[:50])}...")

                                    try:
                                        response = openai.embeddings.create(
                                            model=self.model_name,
                                            input=[extreme_text]
                                        )
                                        all_embeddings.append(np.array(response.data[0].embedding))
                                        print(f"  Successfully processed item {i+j+1} after extreme sanitization")
                                    except Exception as extreme_error:
                                        print(f"  Still failed after extreme sanitization: {extreme_error}")
                                        # Add a zero vector as placeholder
                                        all_embeddings.append(np.zeros(self.dimensions if hasattr(self, 'dimensions') else 1536))
                                else:
                                    # Re-raise if it's not an input validation error
                                    raise
                        except Exception as e2:
                            print(f"  Skipping item {i+j+1} due to error: {e2}")
                            # Add a zero vector as placeholder to maintain alignment
                            all_embeddings.append(np.zeros(self.dimensions if hasattr(self, 'dimensions') else 1536))
                else:
                    raise

        return all_embeddings

    def _sanitize_text(self, text, aggressive=False):
        """Sanitize text to ensure it's valid for the OpenAI API."""
        if not text:
            return ""

        # Basic sanitization
        # Replace null bytes and other control characters
        text = ''.join(ch if ord(ch) >= 32 or ch in '\n\r\t' else ' ' for ch in text)

        # Check for and remove invalid surrogate pairs and other problematic Unicode
        import re
        # Remove invalid UTF-8 sequences and unpaired surrogates
        text = re.sub(r'[\uD800-\uDFFF]', ' ', text)

        # Remove zero-width characters and other invisible formatting
        text = re.sub(r'[\u200B-\u200F\u202A-\u202E\uFEFF]', '', text)

        if aggressive:
            # More aggressive sanitization for problematic texts
            # Replace any non-ASCII characters
            text = ''.join(ch if ord(ch) < 128 else ' ' for ch in text)

            # Limit consecutive whitespace
            text = re.sub(r'\s+', ' ', text)

            # Remove any characters that might cause JSON parsing issues
            text = re.sub(r'[\\"\'\x00-\x1F\x7F-\x9F]', ' ', text)

            # Ensure the text is not empty after sanitization
            if not text.strip():
                return "empty_content"

        return text

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

def build_index(repo_path, api_key, incremental=False, dry_run=False):
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

    parser = argparse.ArgumentParser(description="Semantic search for code repositories (supports Java and non-Java files)")
    parser.add_argument("--index", action="store_true", help="Build the search index")
    parser.add_argument("--incremental", action="store_true", help="Perform incremental indexing (only index changed files)")
    parser.add_argument("--dry-run", action="store_true", help="Parse files without creating embeddings (dry run)")
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
        build_index(args.repo, args.api_key, incremental=args.incremental, dry_run=args.dry_run)
    elif args.search:
        search(args.search, args.top_k, args.api_key, args.index_name)
    else:
        interactive_search(args.api_key)

if __name__ == "__main__":
    main()
