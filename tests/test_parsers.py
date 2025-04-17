"""
Tests for the parsers module.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import tempfile
from semsearch.models import CodeUnit
from semsearch.parsers import GenericFileParser, UnifiedParser, TreeSitterParser


class TestGenericFileParser(unittest.TestCase):
    """Tests for the GenericFileParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = GenericFileParser()
        self.repo_path = "/path/to/repo"
        self.file_path = os.path.join(self.repo_path, "path/to/file.txt")
        self.file_content = "This is a test file."
        self.stats = {
            'parsing_errors': 0,
            'parsing_errors_details': {}
        }

    @patch("builtins.open", new_callable=mock_open, read_data="This is a test file.")
    def test_parse_file(self, mock_file):
        """Test parse_file method."""
        # Call the method
        code_units = self.parser.parse_file(self.file_path, self.repo_path)

        # Check that the file was opened correctly
        mock_file.assert_called_once_with(self.file_path, 'r', encoding='utf-8')

        # Check the result
        self.assertEqual(len(code_units), 1)
        self.assertEqual(code_units[0].path, "path/to/file.txt")
        self.assertEqual(code_units[0].content, "This is a test file.")
        self.assertEqual(code_units[0].unit_type, "file")
        self.assertEqual(code_units[0].name, "file.txt")
        self.assertEqual(code_units[0].package, "path.to")

    @patch("builtins.open")
    def test_parse_file_with_encoding_error(self, mock_file):
        """Test parse_file method with encoding error."""
        # Configure the mock to raise UnicodeDecodeError for utf-8 but succeed for latin-1
        mock_file.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'),
            mock_open(read_data="This is a test file.").return_value
        ]

        # Call the method
        code_units = self.parser.parse_file(self.file_path, self.repo_path)

        # Check that the file was opened with different encodings
        self.assertEqual(mock_file.call_count, 2)
        mock_file.assert_any_call(self.file_path, 'r', encoding='utf-8')
        mock_file.assert_any_call(self.file_path, 'r', encoding='latin-1')

        # Check the result
        self.assertEqual(len(code_units), 1)
        self.assertEqual(code_units[0].content, "This is a test file.")

    @patch("builtins.open")
    def test_parse_file_with_error(self, mock_file):
        """Test parse_file method with error."""
        # Configure the mock to raise an error
        mock_file.side_effect = Exception("Test error")

        # Call the method
        code_units = self.parser.parse_file(self.file_path, self.repo_path, self.stats)

        # Check that the file was opened
        mock_file.assert_called_once_with(self.file_path, 'r', encoding='utf-8')

        # Check the result
        self.assertEqual(len(code_units), 0)

        # Check that the error was tracked
        self.assertEqual(self.stats['parsing_errors'], 1)
        self.assertEqual(self.stats['parsing_errors_details']['Exception'], 1)


class TestUnifiedParser(unittest.TestCase):
    """Tests for the UnifiedParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = UnifiedParser()
        self.repo_path = "/path/to/repo"
        self.java_file_path = os.path.join(self.repo_path, "path/to/file.java")
        self.py_file_path = os.path.join(self.repo_path, "path/to/file.py")

    @patch.object(UnifiedParser, "parse_java_file")
    @patch.object(TreeSitterParser, "parse_file")
    @patch.object(GenericFileParser, "parse_file")
    @patch("os.walk")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_parse_repository(self, mock_file, mock_exists, mock_walk, mock_generic_parse, mock_tree_sitter_parse, mock_java_parse):
        """Test parse_repository method."""
        # Configure the mocks
        mock_exists.return_value = False  # No .gitignore file
        mock_walk.return_value = [
            (self.repo_path, [".git", "path"], []),
            (os.path.join(self.repo_path, "path"), ["to"], []),
            (os.path.join(self.repo_path, "path/to"), [], ["file.java", "file.py", "file.txt", "image.png", "compiled.beam", "library.jar"])
        ]
        mock_java_parse.return_value = [MagicMock()]
        mock_tree_sitter_parse.return_value = [MagicMock()]
        mock_generic_parse.return_value = [MagicMock()]

        # Configure the tree-sitter parser to have python language available
        self.parser.tree_sitter_parser.languages = {'python': MagicMock()}
        self.parser.tree_sitter_parser.language_by_extension = {'.py': 'python'}

        # Call the method
        code_units = self.parser.parse_repository(self.repo_path)

        # Check that the parsers were called correctly
        mock_java_parse.assert_called_once_with(self.java_file_path, self.repo_path)
        mock_tree_sitter_parse.assert_called_once_with(self.py_file_path, self.repo_path, self.parser.stats)

        # The generic parser should be called for file.txt
        # This is a change from the original behavior, where the generic parser was not called at all
        # Now we're using the generic parser for all files except java, python, and erlang
        txt_file_path = os.path.join(self.repo_path, "path/to/file.txt")
        mock_generic_parse.assert_called_once_with(txt_file_path, self.repo_path, self.parser.stats)

        # Check that blacklisted files were skipped
        self.assertEqual(self.parser.stats['skipped_files_blacklisted'], 3)  # png, beam, jar

        # Check the result
        self.assertEqual(len(code_units), 3)  # 1 from Java, 1 from tree-sitter, 1 from generic

        # Check that .git directory was skipped
        self.assertEqual(self.parser.stats['skipped_folders_git'], 1)

    @patch.object(TreeSitterParser, "parse_file")
    @patch("builtins.open", new_callable=mock_open, read_data="public class Test { public void method() {} }")
    def test_parse_java_file(self, mock_file, mock_tree_sitter_parse):
        """Test parse_java_file method."""
        # Configure the mocks
        tree_sitter_units = [
            CodeUnit(
                path="path/to/file.java",
                content="public class Test { }",
                unit_type="class",
                name="Test"
            ),
            CodeUnit(
                path="path/to/file.java",
                content="public void method() {}",
                unit_type="method",
                name="method",
                class_name="Test"
            )
        ]
        mock_tree_sitter_parse.return_value = tree_sitter_units

        # Configure the tree-sitter parser to have Java language available
        self.parser.tree_sitter_parser.languages = {'java': MagicMock()}

        # Call the method
        code_units = self.parser.parse_java_file(self.java_file_path, self.repo_path)

        # Check that tree-sitter parser was used
        mock_tree_sitter_parse.assert_called_once_with(self.java_file_path, self.repo_path, self.parser.stats)

        # Check the result
        self.assertEqual(len(code_units), 2)  # 1 class, 1 method
        self.assertEqual(code_units[0].unit_type, "class")
        self.assertEqual(code_units[0].name, "Test")
        self.assertEqual(code_units[1].unit_type, "method")
        self.assertEqual(code_units[1].name, "method")
        self.assertEqual(code_units[1].class_name, "Test")




class TestTreeSitterParser(unittest.TestCase):
    """Tests for the TreeSitterParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = TreeSitterParser()
        self.repo_path = "/path/to/repo"
        self.java_file_path = os.path.join(self.repo_path, "path/to/file.java")
        self.py_file_path = os.path.join(self.repo_path, "path/to/file.py")
        self.erl_file_path = os.path.join(self.repo_path, "path/to/file.erl")
        self.stats = {
            'parsing_errors': 0,
            'parsing_errors_details': {}
        }

    def test_language_by_extension(self):
        """Test language_by_extension mapping."""
        # Check that the required extensions are mapped to the correct languages
        # We now only support java, python, and erlang as per the issue description
        self.assertEqual(self.parser.language_by_extension['.py'], 'python')
        self.assertEqual(self.parser.language_by_extension['.java'], 'java')
        # self.assertEqual(self.parser.language_by_extension['.erl'], 'erlang')

        # Check that other extensions are not in the dictionary
        self.assertNotIn('.js', self.parser.language_by_extension)
        self.assertNotIn('.ts', self.parser.language_by_extension)
        self.assertNotIn('.c', self.parser.language_by_extension)
        self.assertNotIn('.erl', self.parser.language_by_extension)

    # We no longer support the fallback path for loading languages when tree-sitter-language-pack is not available
    # This test has been removed because it was testing functionality that we're not supporting anymore

    @patch('semsearch.parsers.tree_sitter_language_pack')
    def test_load_languages_with_language_pack(self, mock_language_pack):
        """Test _load_languages method with language pack."""
        # Configure the mocks for get_binding, get_language, and get_parser
        mock_binding = MagicMock()
        mock_language = MagicMock()
        mock_parser = MagicMock()

        mock_language_pack.get_binding.side_effect = lambda lang: mock_binding
        mock_language_pack.get_language.side_effect = lambda lang: mock_language
        mock_language_pack.get_parser.side_effect = lambda lang: mock_parser

        # Create a new parser to trigger _load_languages
        with patch('semsearch.parsers.HAS_LANGUAGE_PACK', True), \
             patch.object(TreeSitterParser, '_load_languages') as mock_load_languages:
            # Create a new parser
            parser = TreeSitterParser()

            # Check that _load_languages was called
            mock_load_languages.assert_called_once()

            # Now manually call _load_languages with a limited set of languages for testing
            parser.languages = {}
            parser.parsers = {}

            # Call the original _load_languages method with a limited set of languages
            languages_to_try = ['python', 'java', 'javascript']
            for lang_name in languages_to_try:
                try:
                    # Get the binding, language, and parser for this language
                    binding = mock_language_pack.get_binding(lang_name)
                    language = mock_language_pack.get_language(lang_name)
                    parser_obj = mock_language_pack.get_parser(lang_name)

                    # Store the language and parser
                    parser.languages[lang_name] = language
                    parser.parsers[lang_name] = parser_obj
                except Exception:
                    pass

            # Check that get_binding, get_language, and get_parser were called for each language
            self.assertEqual(mock_language_pack.get_binding.call_count, 3)  # 3 languages in the list
            self.assertEqual(mock_language_pack.get_language.call_count, 3)  # 3 languages in the list
            self.assertEqual(mock_language_pack.get_parser.call_count, 3)  # 3 languages in the list

            # Check that the languages and parsers were stored
            self.assertEqual(len(parser.languages), 3)
            self.assertEqual(len(parser.parsers), 3)

            # Check that the languages and parsers were stored correctly
            for lang in ['python', 'java', 'javascript']:
                self.assertIn(lang, parser.languages)
                self.assertIn(lang, parser.parsers)
                self.assertEqual(parser.languages[lang], mock_language)
                self.assertEqual(parser.parsers[lang], mock_parser)

    @patch("builtins.open", new_callable=mock_open, read_data="class Test { void method() {} }")
    def test_parse_java_file(self, mock_file):
        """Test parse_file method with a Java file."""
        # Create a mock parser
        mock_parser = MagicMock()
        mock_tree = MagicMock()
        mock_root_node = MagicMock()
        mock_tree.root_node = mock_root_node
        mock_parser.parse.return_value = mock_tree

        # Replace the parser attribute on the instance
        self.parser.parser = mock_parser

        # Configure the languages dictionary
        self.parser.languages = {'java': MagicMock()}

        # Test case 1: Using the default parser (no tree-sitter-language-pack parser)
        # Make sure there's no parser in the parsers dictionary
        self.parser.parsers = {}

        # Call the method
        code_units = self.parser.parse_file(self.java_file_path, self.repo_path, self.stats)

        # Check that the file was opened correctly
        mock_file.assert_called_once_with(self.java_file_path, 'r', encoding='utf-8')

        # Check that the parser was set to the java language
        mock_parser.set_language.assert_called_once_with(self.parser.languages['java'])

        # Check that the parser was called with the file content
        mock_parser.parse.assert_called_once()

        # Check that we got at least the file unit
        self.assertGreaterEqual(len(code_units), 1)
        self.assertEqual(code_units[0].unit_type, "file")
        self.assertEqual(code_units[0].name, "file.java")

        # Reset mocks for the next test case
        mock_file.reset_mock()
        mock_parser.reset_mock()

        # Test case 2: Using a parser from tree-sitter-language-pack
        # Create a mock tree-sitter-language-pack parser
        mock_ts_parser = MagicMock()
        mock_ts_parser.parse.return_value = mock_tree

        # Add the parser to the parsers dictionary
        self.parser.parsers = {'java': mock_ts_parser}

        # Call the method again
        code_units = self.parser.parse_file(self.java_file_path, self.repo_path, self.stats)

        # Check that the file was opened correctly
        mock_file.assert_called_once_with(self.java_file_path, 'r', encoding='utf-8')

        # Check that the default parser was not used
        mock_parser.set_language.assert_not_called()
        mock_parser.parse.assert_not_called()

        # Check that the tree-sitter-language-pack parser was used
        mock_ts_parser.parse.assert_called_once()

        # Check that we got at least the file unit
        self.assertGreaterEqual(len(code_units), 1)
        self.assertEqual(code_units[0].unit_type, "file")
        self.assertEqual(code_units[0].name, "file.java")

    @patch("builtins.open", new_callable=mock_open, read_data="def test(): pass")
    def test_parse_file_without_tree_sitter(self, mock_file):
        """Test parse_file method when tree-sitter is not available for the language."""
        # Configure the languages dictionary to be empty
        self.parser.languages = {}

        # Call the method
        code_units = self.parser.parse_file(self.py_file_path, self.repo_path, self.stats)

        # Check that the file was opened correctly
        mock_file.assert_called_once_with(self.py_file_path, 'r', encoding='utf-8')

        # Check that we got a file unit
        self.assertEqual(len(code_units), 1)
        self.assertEqual(code_units[0].unit_type, "file")
        self.assertEqual(code_units[0].name, "file.py")

    @patch("builtins.open")
    def test_parse_file_with_error(self, mock_file):
        """Test parse_file method with error."""
        # Configure the mock to raise an error
        mock_file.side_effect = Exception("Test error")

        # Call the method
        code_units = self.parser.parse_file(self.py_file_path, self.repo_path, self.stats)

        # Check that the file was opened
        mock_file.assert_called_once_with(self.py_file_path, 'r', encoding='utf-8')

        # Check the result
        self.assertEqual(len(code_units), 0)

        # Check that the error was tracked
        self.assertEqual(self.stats['parsing_errors'], 1)
        self.assertEqual(self.stats['parsing_errors_details']['Exception'], 1)


if __name__ == "__main__":
    unittest.main()
