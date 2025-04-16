"""
Tests for the parsers module.
"""

import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import tempfile
from semsearch.models import CodeUnit
from semsearch.parsers import JavaParser, GenericFileParser, UnifiedParser


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
    @patch.object(GenericFileParser, "parse_file")
    @patch("os.walk")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="")
    def test_parse_repository(self, mock_file, mock_exists, mock_walk, mock_generic_parse, mock_java_parse):
        """Test parse_repository method."""
        # Configure the mocks
        mock_exists.return_value = False  # No .gitignore file
        mock_walk.return_value = [
            (self.repo_path, [".git", "path"], []),
            (os.path.join(self.repo_path, "path"), ["to"], []),
            (os.path.join(self.repo_path, "path/to"), [], ["file.java", "file.py", "file.txt"])
        ]
        mock_java_parse.return_value = [MagicMock()]
        mock_generic_parse.return_value = [MagicMock()]

        # Call the method
        code_units = self.parser.parse_repository(self.repo_path)

        # Check that the parsers were called correctly
        mock_java_parse.assert_called_once_with(self.java_file_path, self.repo_path)
        self.assertEqual(mock_generic_parse.call_count, 1)  # Only for .py file (.txt is not in supported_extensions)

        # Check the result
        self.assertEqual(len(code_units), 2)  # 1 from Java, 1 from generic

        # Check that .git directory was skipped
        self.assertEqual(self.parser.stats['skipped_folders_git'], 1)

    @patch.object(UnifiedParser, "_extract_java_code_units")
    @patch("javalang.parse.parse")
    @patch("builtins.open", new_callable=mock_open, read_data="public class Test { public void method() {} }")
    def test_parse_java_file(self, mock_file, mock_parse, mock_extract):
        """Test parse_java_file method."""
        # Configure the mocks
        mock_tree = MagicMock()
        mock_parse.return_value = mock_tree

        # Set up the mock to return two code units (class and method)
        class_unit = CodeUnit(
            path="path/to/file.java",
            content="public class Test { }",
            unit_type="class",
            name="Test"
        )
        method_unit = CodeUnit(
            path="path/to/file.java",
            content="public void method() {}",
            unit_type="method",
            name="method",
            class_name="Test"
        )
        mock_extract.return_value = [class_unit, method_unit]

        # Call the method
        code_units = self.parser.parse_java_file(self.java_file_path, self.repo_path)

        # Check that the file was opened correctly
        mock_file.assert_called_once_with(self.java_file_path, 'r', encoding='utf-8')

        # Check that the parser was called correctly
        mock_parse.assert_called_once_with("public class Test { public void method() {} }")

        # Check that the extract method was called
        mock_extract.assert_called_once()

        # Check the result
        self.assertEqual(len(code_units), 2)  # 1 class, 1 method
        self.assertEqual(code_units[0].unit_type, "class")
        self.assertEqual(code_units[0].name, "Test")
        self.assertEqual(code_units[1].unit_type, "method")
        self.assertEqual(code_units[1].name, "method")
        self.assertEqual(code_units[1].class_name, "Test")


class TestJavaParser(unittest.TestCase):
    """Tests for the JavaParser class."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = JavaParser()
        self.repo_path = "/path/to/repo"
        self.file_path = os.path.join(self.repo_path, "path/to/file.java")

    @patch.object(UnifiedParser, "parse_java_file")
    def test_parse_file(self, mock_parse):
        """Test parse_file method."""
        # Configure the mock
        mock_parse.return_value = [MagicMock()]

        # Call the method
        code_units = self.parser.parse_file(self.file_path, self.repo_path)

        # Check that the UnifiedParser was called correctly
        mock_parse.assert_called_once_with(self.file_path, self.repo_path)

        # Check the result
        self.assertEqual(len(code_units), 1)


if __name__ == "__main__":
    unittest.main()
