"""
Parsers module for semantic search.

This module contains parsers for different file types to extract code units for semantic search.
"""

import os
import time
from typing import List

import pathspec
import tree_sitter_language_pack
from tree_sitter import Parser

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
        self.tree_sitter_parser = TreeSitterParser()
        # Process Java, Python, and XML files
        # We'll use tree-sitter for java, python, and xml
        self.supported_extensions = {
            '.java',
            '.py',
            '.xml'
        }

        # Blacklist of file extensions that should not be parsed
        self.blacklisted_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',  # Images
            '.beam',  # Erlang compiled files
            '.jar', '.war', '.ear', '.class',  # Java compiled files
            '.pyc', '.pyo', '.pyd',  # Python compiled files
            '.so', '.dll', '.dylib', '.a', '.lib',  # Native libraries
            '.exe', '.bin', '.o', '.obj',  # Executables and object files
            '.zip', '.tar', '.gz', '.bz2', '.7z', '.rar',  # Archives
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',  # Documents
            '.mp3', '.mp4', '.avi', '.mov', '.flv', '.wav',  # Media files
        }

        # Statistics tracking
        self.stats = {
            'total_files': 0,
            'parsed_files': 0,
            'skipped_files_extension': 0,
            'skipped_files_blacklisted': 0,  # Count of blacklisted files
            'skipped_files_gitignore': 0,
            'skipped_folders_gitignore': 0,
            'skipped_folders_git': 0,
            'skipped_folders': set(),  # Store actual folder names
            'parsing_time': 0,
            'parsing_errors': 0,
            'parsing_errors_details': {},  # Store error types and counts
            'code_unit_sizes': {
                'total': 0,
                'by_type': {}
            }  # Track sizes of code units
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
            'skipped_files_blacklisted': 0,  # Count of blacklisted files
            'skipped_files_gitignore': 0,
            'skipped_folders_gitignore': 0,
            'skipped_folders_git': 0,
            'skipped_folders': set(),  # Store actual folder names
            'parsing_time': 0,
            'parsing_errors': 0,
            'parsing_errors_details': {},  # Store error types and counts
            'code_unit_sizes': {
                'total': 0,
                'by_type': {}
            }  # Track sizes of code units
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

                # Get the file extension
                _, ext = os.path.splitext(file.lower())

                # Check if the file extension is in the blacklist
                if ext in self.blacklisted_extensions:
                    self.stats['skipped_files_blacklisted'] += 1
                    continue

                # Skip files that don't have supported extensions
                if ext not in self.supported_extensions:
                    self.stats['skipped_files_extension'] += 1
                    continue

                # We're now only processing files with supported extensions

                file_path = os.path.join(root, file)

                try:
                    # Choose the appropriate parser based on file extension
                    if ext == '.java':
                        # Use the parse_java_file method which now tries tree-sitter first
                        file_code_units = self.parse_java_file(file_path, repo_path)
                    elif ext in self.tree_sitter_parser.language_by_extension:
                        # Try to use tree-sitter parser for other supported languages
                        lang_name = self.tree_sitter_parser.language_by_extension[ext]
                        if lang_name in self.tree_sitter_parser.languages:
                            # Use tree-sitter parser for this language
                            file_code_units = self.tree_sitter_parser.parse_file(file_path, repo_path, self.stats)
                        else:
                            # Fall back to generic parser if tree-sitter language is not available
                            file_code_units = self.generic_parser.parse_file(file_path, repo_path, self.stats)
                    else:
                        # Fall back to generic parser for unsupported languages
                        file_code_units = self.generic_parser.parse_file(file_path, repo_path, self.stats)

                    code_units.extend(file_code_units)
                    self.stats['parsed_files'] += 1

                    # Store file processing status for reporting
                    if 'processed_files' not in self.stats:
                        self.stats['processed_files'] = []
                    self.stats['processed_files'].append({
                        'path': file_path,
                        'relative_path': rel_path,
                        'extension': ext,
                        'size': os.path.getsize(file_path),
                        'code_units': len(file_code_units)
                    })

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

                    # Track the error in statistics
                    self.stats['parsing_errors'] += 1
                    error_type = type(e).__name__
                    if error_type in self.stats['parsing_errors_details']:
                        self.stats['parsing_errors_details'][error_type] += 1
                    else:
                        self.stats['parsing_errors_details'][error_type] = 1

                    # Store error details for reporting
                    if 'error_files' not in self.stats:
                        self.stats['error_files'] = []
                    self.stats['error_files'].append({
                        'path': file_path,
                        'relative_path': rel_path,
                        'extension': ext,
                        'error': str(e),
                        'error_type': error_type
                    })

        # End timing
        end_time = time.time()
        self.stats['parsing_time'] = end_time - start_time

        # Track code unit sizes
        for unit in code_units:
            unit_size = len(unit.content)
            self.stats['code_unit_sizes']['total'] += unit_size

            # Track by type
            unit_type = unit.unit_type
            if unit_type in self.stats['code_unit_sizes']['by_type']:
                self.stats['code_unit_sizes']['by_type'][unit_type]['count'] += 1
                self.stats['code_unit_sizes']['by_type'][unit_type]['size'] += unit_size
            else:
                self.stats['code_unit_sizes']['by_type'][unit_type] = {
                    'count': 1,
                    'size': unit_size
                }

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
        # Use tree-sitter for Java parsing
        return self.tree_sitter_parser.parse_file(file_path, repo_path, self.stats)




class TreeSitterParser:
    """Parser for files using tree-sitter."""
    language_by_extension: dict[str, str]

    def __init__(self):
        """Initialize the TreeSitterParser."""
        self.parser = Parser()
        self.languages = {}
        self.parsers = {}  # Store parsers from tree-sitter-language-pack
        self.language_by_extension = {
            # Map file extensions to language names
            '.py': 'python',
            '.java': 'java',
            '.xml': 'xml',
            # '.erl': 'erlang',
        }

        # Try to load available languages
        self._load_languages()

    def _load_languages(self):
        """Load available tree-sitter languages."""
        # List of languages to try loading, based on SupportedLanguage in tree_sitter_language_pack
        # languages_to_try = [
        #     "actionscript", "ada", "agda", "arduino", "asm", "astro", "bash", "beancount", "bibtex",
        #     "bicep", "bitbake", "c", "cairo", "capnp", "chatito", "clarity", "clojure", "cmake",
        #     "comment", "commonlisp", "cpon", "cpp", "csharp", "css", "csv", "cuda", "d", "dart",
        #     "dockerfile", "doxygen", "dtd", "elisp", "elixir", "elm", "embeddedtemplate", "erlang",
        #     "fennel", "firrtl", "fish", "fortran", "func", "gdscript", "gitattributes", "gitcommit",
        #     "gitignore", "gleam", "glsl", "gn", "go", "gomod", "gosum", "groovy", "gstlaunch", "hack",
        #     "hare", "haskell", "haxe", "hcl", "heex", "hlsl", "html", "hyprlang", "ispc", "janet",
        #     "java", "javascript", "jsdoc", "json", "jsonnet", "julia", "kconfig", "kdl", "kotlin",
        #     "latex", "linkerscript", "llvm", "lua", "luadoc", "luap", "luau", "make", "markdown",
        #     "markdown_inline", "matlab", "mermaid", "meson", "ninja", "nix", "nqc", "objc", "ocaml",
        #     "ocaml_interface", "odin", "org", "pascal", "pem", "perl", "pgn", "php", "po", "pony",
        #     "powershell", "printf", "prisma", "properties", "proto", "psv", "puppet", "purescript",
        #     "pymanifest", "python", "qmldir", "qmljs", "query", "r", "racket", "re2c", "readline",
        #     "requirements", "ron", "rst", "ruby", "rust", "scala", "scheme", "scss", "smali", "smithy",
        #     "solidity", "sparql", "swift", "sql", "squirrel", "starlark", "svelte", "tablegen", "tcl",
        #     "terraform", "test", "thrift", "toml", "tsv", "tsx", "twig", "typescript", "typst", "udev",
        #     "ungrammar", "uxntal", "v", "verilog", "vhdl", "vim", "vue", "wgsl", "xcompose", "xml",
        #     "yaml", "yuck", "zig", "magik"
        # ]

        for lang_name in self.language_by_extension.values():
            try:
                # Get the binding, language, and parser for this language
                binding = tree_sitter_language_pack.get_binding(lang_name)
                language = tree_sitter_language_pack.get_language(lang_name)
                parser = tree_sitter_language_pack.get_parser(lang_name)

                # Store the language and parser
                self.languages[lang_name] = language
                self.parsers[lang_name] = parser

                print(f"Loaded tree-sitter language and parser from language pack: {lang_name}")
            except Exception as e:
                print(f"Failed to load tree-sitter language {lang_name} from language pack: {e}")

    def parse_file(self, file_path: str, repo_path: str, stats=None) -> List[CodeUnit]:
        """
        Parse a file using tree-sitter.

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
        content = None

        # First, try to read the file with different encodings
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # If we got here, we successfully read the file
                if encoding != 'utf-8':
                    print(f"Successfully read file {file_path} using {encoding} encoding")

                break  # Exit the loop if successful

            except UnicodeDecodeError:
                # If this is the last encoding we tried, report the error
                if encoding == encodings[-1]:
                    error_msg = f"Error reading file {file_path}: Unable to decode with any of the attempted encodings"
                    print(error_msg)

                    # Track the error in statistics
                    if stats is not None:
                        stats['parsing_errors'] += 1
                        error_type = "UnicodeDecodeError"
                        if error_type in stats['parsing_errors_details']:
                            stats['parsing_errors_details'][error_type] += 1
                        else:
                            stats['parsing_errors_details'][error_type] = 1

                    return code_units  # Return empty list if we can't read the file
                # Otherwise, try the next encoding
                continue

            except Exception as e:
                error_msg = f"Error reading file {file_path}: {str(e)}"
                print(error_msg)

                # Track the error in statistics
                if stats is not None:
                    stats['parsing_errors'] += 1
                    error_type = type(e).__name__
                    if error_type in stats['parsing_errors_details']:
                        stats['parsing_errors_details'][error_type] += 1
                    else:
                        stats['parsing_errors_details'][error_type] = 1

                return code_units  # Return empty list for non-encoding errors

        # If we couldn't read the file, return empty list
        if content is None:
            return code_units

        # Get the file extension
        _, ext = os.path.splitext(file_path.lower())

        # Get the relative path
        relative_path = os.path.relpath(file_path, repo_path)

        # Use directory structure as package name
        dir_path = os.path.dirname(relative_path)
        package = dir_path.replace(os.path.sep, '.') if dir_path else None

        # Try to parse the file with tree-sitter
        try:
            # Get the language for this file extension
            lang_name = self.language_by_extension.get(ext)

            if lang_name and lang_name in self.languages:
                # Check if we have a pre-configured parser from tree-sitter-language-pack
                if lang_name in self.parsers:
                    # Use the parser from tree-sitter-language-pack
                    parser = self.parsers[lang_name]
                    tree = parser.parse(bytes(content, 'utf-8'))
                else:
                    # Set the language for the parser
                    self.parser.set_language(self.languages[lang_name])

                    # Parse the file
                    tree = self.parser.parse(bytes(content, 'utf-8'))

                # Extract code units from the tree
                code_units.extend(self._extract_code_units(tree, relative_path, content, package, lang_name))

                # If we didn't extract any code units, create a fallback code unit for the entire file
                # For Java files, we only want to create code units for classes and methods
                if not code_units and lang_name != 'java':
                    code_units.append(CodeUnit(
                        path=relative_path,
                        content=content,
                        unit_type="file",
                        name=os.path.basename(file_path),
                        package=package
                    ))
            else:
                # If we don't have a tree-sitter parser for this language, create a code unit for the entire file
                # For Java files, we only want to create code units for classes and methods
                if ext != '.java':
                    code_units.append(CodeUnit(
                        path=relative_path,
                        content=content,
                        unit_type="file",
                        name=os.path.basename(file_path),
                        package=package
                    ))

        except Exception as e:
            error_msg = f"Error parsing file {file_path} with tree-sitter: {str(e)}"
            print(error_msg)

            # Track the error in statistics
            if stats is not None:
                stats['parsing_errors'] += 1
                error_type = type(e).__name__
                if error_type in stats['parsing_errors_details']:
                    stats['parsing_errors_details'][error_type] += 1
                else:
                    stats['parsing_errors_details'][error_type] = 1

            # Create a fallback code unit for the entire file
            # For Java files, we only want to create code units for classes and methods
            if ext != '.java':
                code_units.append(CodeUnit(
                    path=relative_path,
                    content=content,
                    unit_type="file",
                    name=os.path.basename(file_path),
                    package=package
                ))

        return code_units

    def _extract_code_units(self, tree, file_path, content, package, lang_name):
        """
        Extract code units from a parsed tree-sitter syntax tree.

        Args:
            tree: Parsed tree-sitter syntax tree
            file_path: Path to the file
            content: Content of the file
            package: Package name
            lang_name: Language name

        Returns:
            List of CodeUnit objects extracted from the syntax tree
        """
        code_units = []
        root_node = tree.root_node

        # Create a code unit for the entire file only for non-Java files
        # For Java files, we'll only create code units for classes and methods
        if lang_name != 'java':
            code_units.append(CodeUnit(
                path=file_path,
                content=content,
                unit_type="file",
                name=os.path.basename(file_path),
                package=package
            ))

        # Extract classes and functions based on the language
        if lang_name == 'python':
            # Extract Python classes and functions
            for node in root_node.children:
                if node.type == 'class_definition':
                    # Extract class name
                    class_name_node = node.child_by_field_name('name')
                    if class_name_node:
                        class_name = content[class_name_node.start_byte:class_name_node.end_byte]

                        # Extract parent classes (bases)
                        parent_classes = []
                        bases_node = node.child_by_field_name('bases')
                        if bases_node:
                            for base in bases_node.children:
                                if base.type not in ['(', ')', ',']:
                                    base_name = content[base.start_byte:base.end_byte]
                                    parent_classes.append(base_name)

                        # Create a stub representation of the class
                        if parent_classes:
                            class_stub = f"class {class_name}({', '.join(parent_classes)}): ... {{\n"
                        else:
                            class_stub = f"class {class_name}: ... {{\n"

                        # Extract fields and methods
                        fields = []
                        methods = []

                        # Find the block node (class body)
                        for child in node.children:
                            if child.type == 'block':
                                # Extract fields and methods from the class body
                                for item in child.children:
                                    if item.type == 'function_definition':
                                        # Extract method name and parameters
                                        method_name_node = item.child_by_field_name('name')
                                        if method_name_node:
                                            method_name = content[method_name_node.start_byte:method_name_node.end_byte]

                                            # Extract parameters
                                            params = []
                                            parameters_node = item.child_by_field_name('parameters')
                                            if parameters_node:
                                                for param in parameters_node.children:
                                                    if param.type not in ['(', ')', ',']:
                                                        param_text = content[param.start_byte:param.end_byte]
                                                        params.append(param_text)

                                            methods.append(f"  {method_name}({', '.join(params)})")

                                    elif item.type == 'expression_statement':
                                        # This might be a field assignment
                                        for expr in item.children:
                                            if expr.type == 'assignment':
                                                left_node = expr.child_by_field_name('left')
                                                if left_node and left_node.type == 'identifier':
                                                    field_name = content[left_node.start_byte:left_node.end_byte]
                                                    fields.append(f"  {field_name}")

                        # Add fields and methods to the stub
                        for field in fields:
                            class_stub += field + "\n"

                        for method in methods:
                            class_stub += method + "\n"

                        class_stub += "}"

                        code_units.append(CodeUnit(
                            path=file_path,
                            content=class_stub,
                            unit_type="class",
                            name=class_name,
                            package=package
                        ))

                        # Extract methods within the class
                        for class_child in node.children:
                            if class_child.type == 'block':
                                for method_node in class_child.children:
                                    if method_node.type == 'function_definition':
                                        method_name_node = method_node.child_by_field_name('name')
                                        if method_name_node:
                                            method_name = content[method_name_node.start_byte:method_name_node.end_byte]
                                            method_content = content[method_node.start_byte:method_node.end_byte]

                                            # Limit content size to avoid token limit issues
                                            if len(method_content) > 4000:  # Roughly 1000 tokens
                                                method_content = method_content[:4000] + "... [content truncated]"

                                            code_units.append(CodeUnit(
                                                path=file_path,
                                                content=method_content,
                                                unit_type="method",
                                                name=method_name,
                                                package=package,
                                                class_name=class_name
                                            ))

                elif node.type == 'function_definition':
                    # Extract function name
                    func_name_node = node.child_by_field_name('name')
                    if func_name_node:
                        func_name = content[func_name_node.start_byte:func_name_node.end_byte]
                        func_content = content[node.start_byte:node.end_byte]

                        # Limit content size to avoid token limit issues
                        if len(func_content) > 4000:  # Roughly 1000 tokens
                            func_content = func_content[:4000] + "... [content truncated]"

                        code_units.append(CodeUnit(
                            path=file_path,
                            content=func_content,
                            unit_type="function",
                            name=func_name,
                            package=package
                        ))

        elif lang_name == 'java':
            # Extract Java classes and methods
            for node in root_node.children:
                if node.type == 'class_declaration':
                    # Extract class name
                    class_name_node = node.child_by_field_name('name')
                    if class_name_node:
                        class_name = content[class_name_node.start_byte:class_name_node.end_byte]

                        # Extract superclass
                        superclass = None
                        superclass_node = node.child_by_field_name('superclass')
                        if superclass_node:
                            superclass = content[superclass_node.start_byte:superclass_node.end_byte]

                        # Extract interfaces
                        interfaces = []
                        interfaces_node = node.child_by_field_name('interfaces')
                        if interfaces_node:
                            for interface in interfaces_node.children:
                                if interface.type not in ['implements', ',']:
                                    interface_name = content[interface.start_byte:interface.end_byte]
                                    interfaces.append(interface_name)

                        # Create a stub representation of the class
                        class_stub = f"class {class_name}"
                        if superclass:
                            class_stub += f" extends {superclass}"
                        if interfaces:
                            class_stub += f" implements {', '.join(interfaces)}"
                        class_stub += ": ... {\n"

                        # Extract fields and methods
                        fields = []
                        methods = []

                        # Find the class_body node
                        for child in node.children:
                            if child.type == 'class_body':
                                # Extract fields and methods from the class body
                                for item in child.children:
                                    if item.type == 'method_declaration':
                                        # Extract method name and parameters
                                        method_name_node = item.child_by_field_name('name')
                                        if method_name_node:
                                            method_name = content[method_name_node.start_byte:method_name_node.end_byte]

                                            # Extract parameters
                                            params = []
                                            formal_parameters = item.child_by_field_name('parameters')
                                            if formal_parameters:
                                                for param in formal_parameters.children:
                                                    if param.type == 'formal_parameter':
                                                        # Get parameter type
                                                        param_type_node = param.child_by_field_name('type')
                                                        param_type = ""
                                                        if param_type_node:
                                                            param_type = content[param_type_node.start_byte:param_type_node.end_byte]

                                                        # Get parameter name
                                                        param_name_node = param.child_by_field_name('name')
                                                        param_name = ""
                                                        if param_name_node:
                                                            param_name = content[param_name_node.start_byte:param_name_node.end_byte]

                                                        if param_type and param_name:
                                                            params.append(f"{param_type} {param_name}")
                                                        elif param_name:
                                                            params.append(param_name)

                                            # Get return type
                                            return_type = ""
                                            return_type_node = item.child_by_field_name('type')
                                            if return_type_node:
                                                return_type = content[return_type_node.start_byte:return_type_node.end_byte]

                                            method_signature = f"  {method_name}({', '.join(params)})"
                                            if return_type:
                                                method_signature = f"  {return_type} {method_signature}"

                                            methods.append(method_signature)

                                    elif item.type == 'field_declaration':
                                        # Extract field type and name
                                        field_type_node = item.child_by_field_name('type')
                                        field_type = ""
                                        if field_type_node:
                                            field_type = content[field_type_node.start_byte:field_type_node.end_byte]

                                        # Find variable declarator
                                        for decl in item.children:
                                            if decl.type == 'variable_declarator':
                                                field_name_node = decl.child_by_field_name('name')
                                                if field_name_node:
                                                    field_name = content[field_name_node.start_byte:field_name_node.end_byte]
                                                    if field_type:
                                                        fields.append(f"  {field_type} {field_name}")
                                                    else:
                                                        fields.append(f"  {field_name}")

                        # Add fields and methods to the stub
                        for field in fields:
                            class_stub += field + "\n"

                        for method in methods:
                            class_stub += method + "\n"

                        class_stub += "}"

                        code_units.append(CodeUnit(
                            path=file_path,
                            content=class_stub,
                            unit_type="class",
                            name=class_name,
                            package=package
                        ))

                        # Extract methods within the class
                        for class_child in node.children:
                            if class_child.type == 'class_body':
                                for method_node in class_child.children:
                                    if method_node.type == 'method_declaration':
                                        method_name_node = method_node.child_by_field_name('name')
                                        if method_name_node:
                                            method_name = content[method_name_node.start_byte:method_name_node.end_byte]
                                            method_content = content[method_node.start_byte:method_node.end_byte]

                                            # Limit content size to avoid token limit issues
                                            if len(method_content) > 4000:  # Roughly 1000 tokens
                                                method_content = method_content[:4000] + "... [content truncated]"

                                            code_units.append(CodeUnit(
                                                path=file_path,
                                                content=method_content,
                                                unit_type="method",
                                                name=method_name,
                                                package=package,
                                                class_name=class_name
                                            ))

        elif lang_name == 'erlang':
            # Extract Erlang functions
            # Try to find function declarations
            for node in root_node.children:
                if node.type == 'fun_decl':
                    # Extract function name
                    # Look for the function_clause child, which contains the function name
                    for child in node.children:
                        if child.type == 'function_clause':
                            # Look for the atom child of the function_clause, which is the function name
                            for clause_child in child.children:
                                if clause_child.type == 'atom':
                                    func_name = content[clause_child.start_byte:clause_child.end_byte]
                                    func_content = content[node.start_byte:node.end_byte]

                                    # Limit content size to avoid token limit issues
                                    if len(func_content) > 4000:  # Roughly 1000 tokens
                                        func_content = func_content[:4000] + "... [content truncated]"

                                    code_units.append(CodeUnit(
                                        path=file_path,
                                        content=func_content,
                                        unit_type="function",
                                        name=func_name,
                                        package=package
                                    ))
                                    break  # Only use the first atom as the function name
                            break  # Only use the first function_clause
                elif node.type == 'function':
                    # Extract function name
                    func_name_node = node.child_by_field_name('name')
                    if func_name_node:
                        func_name = content[func_name_node.start_byte:func_name_node.end_byte]
                        func_content = content[node.start_byte:node.end_byte]

                        # Limit content size to avoid token limit issues
                        if len(func_content) > 4000:  # Roughly 1000 tokens
                            func_content = func_content[:4000] + "... [content truncated]"

                        code_units.append(CodeUnit(
                            path=file_path,
                            content=func_content,
                            unit_type="function",
                            name=func_name,
                            package=package
                        ))

        # Add more language-specific parsing as needed

        return code_units
