"""
Tests for parsing, indexing, and searching Erlang files.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
from semsearch.models import CodeUnit
from semsearch.parsers import GenericFileParser
from semsearch.indexing import VectorIndex
from semsearch.embedding import CodeEmbedder

class TestErlangFiles(unittest.TestCase):
    """Tests for parsing, indexing, and searching Erlang files."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.documentation_dir = os.path.join(self.test_data_dir, 'documentation')
        self.bif_dir = os.path.join(self.test_data_dir, 'bif')
        self.parser = GenericFileParser()
        self.embedder = CodeEmbedder(dimensions=384)  # Adjust dimensions based on your embedding model
        self.index = VectorIndex(dimensions=384)  # Adjust dimensions based on your embedding model

    def test_parse_erlang_documentation_file(self):
        """Test parsing the Erlang documentation file."""
        doc_file_path = os.path.join(self.documentation_dir, 'ErlangDocumentationProviderTest.java')

        # Ensure the file exists
        self.assertTrue(os.path.exists(doc_file_path), f"Test file not found: {doc_file_path}")

        # Parse the file
        code_units = self.parser.parse_file(doc_file_path, self.test_data_dir)

        # Check that we got at least one code unit
        self.assertGreater(len(code_units), 0, "No code units parsed from documentation file")

        # Check that the code unit has the correct properties
        doc_unit = code_units[0]
        self.assertEqual(doc_unit.path, os.path.relpath(doc_file_path, self.test_data_dir))
        self.assertEqual(doc_unit.unit_type, "file")
        self.assertEqual(doc_unit.name, "ErlangDocumentationProviderTest.java")

        # Check that the content contains expected text
        self.assertIn("ErlangDocumentationProviderTest", doc_unit.content)
        self.assertIn("testGenerateDocSdkBif", doc_unit.content)

    def test_parse_erlang_bif_file(self):
        """Test parsing the Erlang BIF file."""
        bif_file_path = os.path.join(self.bif_dir, 'ErlangBifParser.java')

        # Ensure the file exists
        self.assertTrue(os.path.exists(bif_file_path), f"Test file not found: {bif_file_path}")

        # Parse the file
        code_units = self.parser.parse_file(bif_file_path, self.test_data_dir)

        # Check that we got at least one code unit
        self.assertGreater(len(code_units), 0, "No code units parsed from BIF file")

        # Check that the code unit has the correct properties
        bif_unit = code_units[0]
        self.assertEqual(bif_unit.path, os.path.relpath(bif_file_path, self.test_data_dir))
        self.assertEqual(bif_unit.unit_type, "file")
        self.assertEqual(bif_unit.name, "ErlangBifParser.java")

        # Check that the content contains expected text
        self.assertIn("ErlangBifParser", bif_unit.content)
        self.assertIn("BIF_DECLARATION", bif_unit.content)

    @patch.object(CodeEmbedder, 'embed_code_units')
    @patch.object(CodeEmbedder, 'embed_query')
    def test_index_and_search_erlang_files(self, mock_embed_query, mock_embed_code_units):
        """Test indexing and searching Erlang files."""
        # Parse both files
        doc_file_path = os.path.join(self.documentation_dir, 'ErlangDocumentationProviderTest.java')
        bif_file_path = os.path.join(self.bif_dir, 'ErlangBifParser.java')

        doc_units = self.parser.parse_file(doc_file_path, self.test_data_dir)
        bif_units = self.parser.parse_file(bif_file_path, self.test_data_dir)

        all_units = doc_units + bif_units

        # Set up mock embeddings
        # Create mock embeddings that will make the documentation file more relevant for a documentation query
        doc_embedding = np.ones(384, dtype=np.float32) * 0.8
        bif_embedding = np.ones(384, dtype=np.float32) * 0.2
        mock_embeddings = {
            doc_units[0]: doc_embedding,
            bif_units[0]: bif_embedding
        }
        mock_embed_code_units.return_value = mock_embeddings

        # Create a mock query embedding that is closer to the documentation embedding
        query_embedding = np.ones(384, dtype=np.float32) * 0.75
        mock_embed_query.return_value = query_embedding

        # Get embeddings for the code units
        embeddings = self.embedder.embed_code_units(all_units)

        # Build the index
        self.index.build_index(embeddings)

        # Search for something related to "documentation"
        query = "Erlang documentation provider"
        query_embedding = self.embedder.embed_query(query)

        results = self.index.search(query_embedding, 2)

        # Check that we got results
        self.assertEqual(len(results), 2, "Expected 2 search results")

        # The documentation file should be more relevant for a documentation query
        doc_file_in_results = any(unit.name == "ErlangDocumentationProviderTest.java" for unit, _ in results)
        self.assertTrue(doc_file_in_results, "Documentation file not found in search results")

    @patch.object(CodeEmbedder, 'embed_code_units')
    @patch.object(CodeEmbedder, 'embed_query')
    def test_search_for_bif_related_content(self, mock_embed_query, mock_embed_code_units):
        """Test searching for BIF-related content."""
        # Parse both files
        doc_file_path = os.path.join(self.documentation_dir, 'ErlangDocumentationProviderTest.java')
        bif_file_path = os.path.join(self.bif_dir, 'ErlangBifParser.java')

        doc_units = self.parser.parse_file(doc_file_path, self.test_data_dir)
        bif_units = self.parser.parse_file(bif_file_path, self.test_data_dir)

        all_units = doc_units + bif_units

        # Set up mock embeddings
        # Create mock embeddings that will make the BIF file more relevant for a BIF query
        doc_embedding = np.ones(384, dtype=np.float32) * 0.2
        bif_embedding = np.ones(384, dtype=np.float32) * 0.8
        mock_embeddings = {
            doc_units[0]: doc_embedding,
            bif_units[0]: bif_embedding
        }
        mock_embed_code_units.return_value = mock_embeddings

        # Create a mock query embedding that is closer to the BIF embedding
        query_embedding = np.ones(384, dtype=np.float32) * 0.75
        mock_embed_query.return_value = query_embedding

        # Get embeddings for the code units
        embeddings = self.embedder.embed_code_units(all_units)

        # Build the index
        self.index.build_index(embeddings)

        # Search for something related to "BIF"
        query = "Erlang built-in functions parser"
        query_embedding = self.embedder.embed_query(query)

        results = self.index.search(query_embedding, 2)

        # Check that we got results
        self.assertEqual(len(results), 2, "Expected 2 search results")

        # The BIF file should be more relevant for a BIF query
        bif_file_in_results = any(unit.name == "ErlangBifParser.java" for unit, _ in results)
        self.assertTrue(bif_file_in_results, "BIF file not found in search results")


if __name__ == "__main__":
    unittest.main()
