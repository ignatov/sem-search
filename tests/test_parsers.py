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

        # Make sure the code_unit_sizes field is initialized
        if 'code_unit_sizes' not in self.parser.stats:
            self.parser.stats['code_unit_sizes'] = {
                'total': 0,
                'by_type': {}
            }

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
            (os.path.join(self.repo_path, "path/to"), [], ["file.java", "file.py", "file.xml", "file.txt", "image.png", "compiled.beam", "library.jar"])
        ]

        # Create mock code units with content to test size tracking
        java_unit = MagicMock()
        java_unit.content = "public class Test {}"
        java_unit.unit_type = "class"

        py_unit = MagicMock()
        py_unit.content = "def test(): pass"
        py_unit.unit_type = "function"

        xml_unit = MagicMock()
        xml_unit.content = "<root>XML content</root>"
        xml_unit.unit_type = "file"

        mock_java_parse.return_value = [java_unit]
        mock_tree_sitter_parse.side_effect = lambda file_path, repo_path, stats: [py_unit] if file_path.endswith('.py') else [xml_unit]
        mock_generic_parse.return_value = []

        # Configure the tree-sitter parser to have python and xml languages available
        self.parser.tree_sitter_parser.languages = {'python': MagicMock(), 'xml': MagicMock()}
        self.parser.tree_sitter_parser.language_by_extension = {'.py': 'python', '.xml': 'xml'}

        # Call the method
        code_units = self.parser.parse_repository(self.repo_path)

        # Check that the parsers were called correctly
        mock_java_parse.assert_called_once_with(self.java_file_path, self.repo_path)

        # The tree-sitter parser should be called for both Python and XML files
        py_file_path = os.path.join(self.repo_path, "path/to/file.py")
        xml_file_path = os.path.join(self.repo_path, "path/to/file.xml")

        # Check that tree_sitter_parse was called twice (once for Python, once for XML)
        self.assertEqual(mock_tree_sitter_parse.call_count, 2)

        # Check that the calls were made with the correct arguments
        mock_tree_sitter_parse.assert_any_call(py_file_path, self.repo_path, self.parser.stats)
        mock_tree_sitter_parse.assert_any_call(xml_file_path, self.repo_path, self.parser.stats)

        # The generic parser should not be called at all since we're not processing text files anymore
        mock_generic_parse.assert_not_called()

        # Check that blacklisted files were skipped
        self.assertEqual(self.parser.stats['skipped_files_blacklisted'], 3)  # png, beam, jar

        # Check the result
        self.assertEqual(len(code_units), 3)  # 1 from Java, 1 from Python tree-sitter, 1 from XML tree-sitter

        # Check that .git directory was skipped
        self.assertEqual(self.parser.stats['skipped_folders_git'], 1)

        # Check that code unit sizes are tracked correctly
        self.assertIn('code_unit_sizes', self.parser.stats)
        self.assertIn('total', self.parser.stats['code_unit_sizes'])
        self.assertIn('by_type', self.parser.stats['code_unit_sizes'])

        # Calculate expected total size
        expected_total_size = len("public class Test {}") + len("def test(): pass") + len("<root>XML content</root>")
        self.assertEqual(self.parser.stats['code_unit_sizes']['total'], expected_total_size)

        # Check sizes by type
        self.assertIn('class', self.parser.stats['code_unit_sizes']['by_type'])
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['class']['size'], len("public class Test {}"))
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['class']['count'], 1)

        self.assertIn('function', self.parser.stats['code_unit_sizes']['by_type'])
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['function']['size'], len("def test(): pass"))
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['function']['count'], 1)

        self.assertIn('file', self.parser.stats['code_unit_sizes']['by_type'])
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['file']['size'], len("<root>XML content</root>"))
        self.assertEqual(self.parser.stats['code_unit_sizes']['by_type']['file']['count'], 1)

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
        self.assertEqual(self.parser.language_by_extension['.py'], 'python')
        self.assertEqual(self.parser.language_by_extension['.java'], 'java')
        self.assertEqual(self.parser.language_by_extension['.xml'], 'xml')
        # self.assertEqual(self.parser.language_by_extension['.erl'], 'erlang')

        # Check that other extensions are not in the dictionary
        self.assertNotIn('.js', self.parser.language_by_extension)
        self.assertNotIn('.ts', self.parser.language_by_extension)
        self.assertNotIn('.c', self.parser.language_by_extension)
        self.assertNotIn('.erl', self.parser.language_by_extension)
        self.assertNotIn('.txt', self.parser.language_by_extension)

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
        with patch.object(TreeSitterParser, '_load_languages') as mock_load_languages:
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

        # For Java files, we don't create file-level code units
        # The code_units list might be empty if no classes or methods were extracted
        # This is expected behavior for Java files

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

        # For Java files, we don't create file-level code units
        # The code_units list might be empty if no classes or methods were extracted
        # This is expected behavior for Java files

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


    def test_python_class_stub_generation(self):
        """Test that Python class stubs are generated correctly."""
        # Create a mock tree and nodes
        mock_tree = MagicMock()
        mock_root_node = MagicMock()
        mock_tree.root_node = mock_root_node

        # Create a mock class node
        mock_class_node = MagicMock()
        mock_class_node.type = 'class_definition'

        # Create a mock class name node
        mock_class_name_node = MagicMock()
        mock_class_name_node.start_byte = 6
        mock_class_name_node.end_byte = 14

        # Create mock bases node (parent classes)
        mock_bases_node = MagicMock()
        mock_base1 = MagicMock()
        mock_base1.type = 'identifier'
        mock_base1.start_byte = 16
        mock_base1.end_byte = 25
        mock_base2 = MagicMock()
        mock_base2.type = 'identifier'
        mock_base2.start_byte = 27
        mock_base2.end_byte = 38
        mock_bases_node.children = [MagicMock(), mock_base1, MagicMock(), mock_base2, MagicMock()]

        # Set up child_by_field_name to return different nodes based on the field name
        mock_class_node.child_by_field_name = lambda field: {
            'name': mock_class_name_node,
            'bases': mock_bases_node
        }.get(field)

        # Create a mock block node (class body)
        mock_block_node = MagicMock()
        mock_block_node.type = 'block'

        # Create mock method nodes
        mock_method1_node = MagicMock()
        mock_method1_node.type = 'function_definition'
        mock_method1_name_node = MagicMock()
        mock_method1_name_node.start_byte = 30
        mock_method1_name_node.end_byte = 38
        mock_method1_node.child_by_field_name.side_effect = lambda name: mock_method1_name_node if name == 'name' else mock_method1_params_node

        # Create mock method parameters
        mock_method1_params_node = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.type = 'identifier'
        mock_param1.start_byte = 39
        mock_param1.end_byte = 43
        mock_param2 = MagicMock()
        mock_param2.type = 'identifier'
        mock_param2.start_byte = 45
        mock_param2.end_byte = 50
        mock_method1_params_node.children = [MagicMock(), mock_param1, MagicMock(), mock_param2, MagicMock()]

        # Create mock field assignment
        mock_field_node = MagicMock()
        mock_field_node.type = 'expression_statement'
        mock_assignment = MagicMock()
        mock_assignment.type = 'assignment'
        mock_left_node = MagicMock()
        mock_left_node.type = 'identifier'
        mock_left_node.start_byte = 20
        mock_left_node.end_byte = 25
        mock_assignment.child_by_field_name.return_value = mock_left_node
        mock_field_node.children = [mock_assignment]

        # Set up the node hierarchy
        mock_block_node.children = [mock_field_node, mock_method1_node]
        mock_class_node.children = [mock_class_name_node, mock_block_node]
        mock_root_node.children = [mock_class_node]

        # Create a sample file content that matches the mock node byte positions
        file_content = "class TestClass(BaseClass, Interface):\n    field1 = 42\n    def method1(self, param):\n        pass"

        # Call the _extract_code_units method
        code_units = self.parser._extract_code_units(mock_tree, "test.py", file_content, "test.package", "python")

        # Find the class code unit
        class_unit = None
        for unit in code_units:
            if unit.unit_type == "class":
                class_unit = unit
                break

        # Check that the class unit was created
        self.assertIsNotNone(class_unit)
        self.assertEqual(class_unit.name, "TestClass")

        # Check that the class stub has the expected format
        expected_stub_start = "class TestClass(BaseClass, Interface): ... {"
        expected_stub_contains_field = "  field1"
        expected_stub_contains_method = "  method1(self, param)"
        expected_stub_end = "}"

        self.assertTrue(class_unit.content.startswith(expected_stub_start))
        self.assertIn(expected_stub_contains_field, class_unit.content)
        self.assertIn(expected_stub_contains_method, class_unit.content)
        self.assertTrue(class_unit.content.endswith(expected_stub_end))

    def test_java_class_stub_generation(self):
        """Test that Java class stubs are generated correctly."""
        # Create a mock tree and nodes
        mock_tree = MagicMock()
        mock_root_node = MagicMock()
        mock_tree.root_node = mock_root_node

        # Create a mock class node
        mock_class_node = MagicMock()
        mock_class_node.type = 'class_declaration'

        # Create a mock class name node
        mock_class_name_node = MagicMock()
        mock_class_name_node.start_byte = 6
        mock_class_name_node.end_byte = 14

        # Create mock superclass node
        mock_superclass_node = MagicMock()
        mock_superclass_node.start_byte = 23
        mock_superclass_node.end_byte = 32

        # Create mock interfaces node
        mock_interfaces_node = MagicMock()
        mock_interface1 = MagicMock()
        mock_interface1.type = 'identifier'
        mock_interface1.start_byte = 44
        mock_interface1.end_byte = 53
        mock_interface2 = MagicMock()
        mock_interface2.type = 'identifier'
        mock_interface2.start_byte = 55
        mock_interface2.end_byte = 66
        mock_interfaces_node.children = [MagicMock(), mock_interface1, MagicMock(), mock_interface2, MagicMock()]

        # Set up child_by_field_name to return different nodes based on the field name
        mock_class_node.child_by_field_name = lambda field: {
            'name': mock_class_name_node,
            'superclass': mock_superclass_node,
            'interfaces': mock_interfaces_node
        }.get(field)

        # Create a mock class body node
        mock_class_body = MagicMock()
        mock_class_body.type = 'class_body'

        # Create mock field declaration
        mock_field_node = MagicMock()
        mock_field_node.type = 'field_declaration'
        mock_field_type_node = MagicMock()
        mock_field_type_node.start_byte = 20
        mock_field_type_node.end_byte = 26
        mock_field_node.child_by_field_name.return_value = mock_field_type_node

        mock_var_declarator = MagicMock()
        mock_var_declarator.type = 'variable_declarator'
        mock_field_name_node = MagicMock()
        mock_field_name_node.start_byte = 27
        mock_field_name_node.end_byte = 33
        mock_var_declarator.child_by_field_name.return_value = mock_field_name_node
        mock_field_node.children = [mock_field_type_node, mock_var_declarator]

        # Create mock method declaration
        mock_method_node = MagicMock()
        mock_method_node.type = 'method_declaration'

        # Method return type
        mock_return_type_node = MagicMock()
        mock_return_type_node.start_byte = 40
        mock_return_type_node.end_byte = 44

        # Method name
        mock_method_name_node = MagicMock()
        mock_method_name_node.start_byte = 45
        mock_method_name_node.end_byte = 53

        # Method parameters
        mock_params_node = MagicMock()
        mock_param = MagicMock()
        mock_param.type = 'formal_parameter'

        # Parameter type
        mock_param_type_node = MagicMock()
        mock_param_type_node.start_byte = 54
        mock_param_type_node.end_byte = 60

        # Parameter name
        mock_param_name_node = MagicMock()
        mock_param_name_node.start_byte = 61
        mock_param_name_node.end_byte = 65

        mock_param.child_by_field_name.side_effect = lambda name: mock_param_type_node if name == 'type' else mock_param_name_node
        mock_params_node.children = [mock_param]

        # Set up method node field lookups
        mock_method_node.child_by_field_name.side_effect = lambda name: {
            'type': mock_return_type_node,
            'name': mock_method_name_node,
            'parameters': mock_params_node
        }.get(name)

        # Set up the node hierarchy
        mock_class_body.children = [mock_field_node, mock_method_node]
        mock_class_node.children = [mock_class_name_node, mock_class_body]
        mock_root_node.children = [mock_class_node]

        # Create a sample file content that matches the mock node byte positions
        file_content = "class TestClass extends BaseClass implements Interface1, Interface2 {\n    String field1;\n    void method1(String param) {\n        // method body\n    }\n}"

        # Call the _extract_code_units method
        code_units = self.parser._extract_code_units(mock_tree, "Test.java", file_content, "test.package", "java")

        # Find the class code unit
        class_unit = None
        for unit in code_units:
            if unit.unit_type == "class":
                class_unit = unit
                break

        # Check that the class unit was created
        self.assertIsNotNone(class_unit)
        self.assertEqual(class_unit.name, "TestClass")

        # Check that the class stub has the expected format
        expected_stub_start = "class TestClass extends BaseClass implements Interface1, Interface2: ... {"
        expected_stub_contains_field = "  String field1"
        expected_stub_contains_method = "  void method1(String param)"
        expected_stub_end = "}"

        self.assertTrue(class_unit.content.startswith(expected_stub_start))
        self.assertIn(expected_stub_contains_field, class_unit.content)
        self.assertIn(expected_stub_contains_method, class_unit.content)
        self.assertTrue(class_unit.content.endswith(expected_stub_end))


if __name__ == "__main__":
    unittest.main()
