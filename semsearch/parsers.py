"""
Parsers module for semantic search.

This module contains parsers for different file types to extract code units for semantic search.
"""

import os
import time
import re
import javalang
import pathspec
from typing import List, Dict, Set, Optional

from semsearch.models import CodeUnit


class GenericFileParser:
    """Parser for generic (non-Java) files."""
    
    def parse_file(self, file_path: str, repo_path: str, stats=None) -> List[CodeUnit]:
        """
        Parse a single non-Java file and extract code units.
        
        Args:
            file_path: Path to the file to parse
            repo_path: Path to the repository root
            stats: Optional dictionary to track parsing statistics
            
        Returns:
            List of CodeUnit objects extracted from the file
        """
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
    """Parser that can handle multiple file types."""
    
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
        """
        Load .gitignore patterns from the repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            PathSpec object with gitignore patterns
        """
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
        """
        Walk through the repository once and parse all files.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            List of CodeUnit objects extracted from all files in the repository
        """
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
        """
        Parse a single Java file and extract code units.
        
        Args:
            file_path: Path to the Java file to parse
            repo_path: Path to the repository root
            
        Returns:
            List of CodeUnit objects extracted from the Java file
        """
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
        """
        Extract code units from a parsed Java syntax tree.
        
        Args:
            tree: Parsed Java syntax tree
            file_path: Path to the Java file
            content: Content of the Java file
            
        Returns:
            List of CodeUnit objects extracted from the Java syntax tree
        """
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
        """
        Get the start and end positions of a node in the source code.
        
        Args:
            content: Content of the Java file
            node: Java syntax tree node
            
        Returns:
            Tuple of (start_position, end_position)
        """
        # This is a simplification. In a real implementation, 
        # you would need to determine exact positions in the source code
        # Using javalang's token positions and parsing
        start_line = node.position.line if hasattr(node, 'position') and node.position else 0
        end_line = len(content.splitlines())

        lines = content.splitlines()
        start_pos = sum(len(lines[i]) + 1 for i in range(start_line - 1)) if start_line > 0 else 0
        end_pos = len(content)

        return start_pos, end_pos


class JavaParser:
    """
    Legacy parser for Java files.
    
    This is now just a wrapper around UnifiedParser for backward compatibility.
    """
    
    def parse_file(self, file_path: str, repo_path: str) -> List[CodeUnit]:
        """
        Parse a single Java file and extract code units.
        
        Args:
            file_path: Path to the Java file to parse
            repo_path: Path to the repository root
            
        Returns:
            List of CodeUnit objects extracted from the Java file
        """
        # This is now just a wrapper around UnifiedParser for backward compatibility
        unified_parser = UnifiedParser()
        return unified_parser.parse_java_file(file_path, repo_path)