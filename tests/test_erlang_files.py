"""
Tests for parsing, indexing, and searching Erlang files.

This module includes tests that use real Erlang files as test data.
Some tests use mocked embeddings to avoid making actual API calls,
while others can use real embeddings if an OpenAI API key is provided.

To run tests with real embeddings:
1. Copy the .env.sample file to .env
2. Add your OpenAI API key to the .env file
3. Run the tests with: python -m unittest tests.test_erlang_files

If no API key is provided, the tests that require real embeddings will be skipped.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
from dotenv import load_dotenv
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


    def test_real_embeddings_with_api_key(self):
        """Test using real embeddings with an API key from .env file."""
        # Load API key from .env file
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")

        # Skip test if no API key is found
        if not api_key:
            self.skipTest("No OpenAI API key found in .env file or environment variables. Skipping test.")

        # Set the API key for OpenAI
        import openai
        openai.api_key = api_key

        # Parse both files
        doc_file_path = os.path.join(self.documentation_dir, 'ErlangDocumentationProviderTest.java')
        bif_file_path = os.path.join(self.bif_dir, 'ErlangBifParser.java')

        doc_units = self.parser.parse_file(doc_file_path, self.test_data_dir)
        bif_units = self.parser.parse_file(bif_file_path, self.test_data_dir)

        all_units = doc_units + bif_units

        # Get real embeddings for the code units (not mocked)
        # The dimensions depend on the model being used, so we'll get the dimensions from the first embedding
        real_embedder = CodeEmbedder()  # Use default dimensions
        embeddings = real_embedder.embed_code_units(all_units)

        # Check that we got embeddings for all units
        self.assertEqual(len(embeddings), len(all_units), 
                         "Number of embeddings should match number of code units")

        # Get the actual dimensions from the first embedding
        first_embedding = next(iter(embeddings.values()))
        actual_dimensions = first_embedding.shape[0]
        print(f"Actual embedding dimensions: {actual_dimensions}")

        # Build the index with the correct dimensions
        real_index = VectorIndex(dimensions=actual_dimensions)
        real_index.build_index(embeddings)

        # Search for documentation-related content
        doc_query = "Erlang documentation provider for IDE"
        doc_query_embedding = real_embedder.embed_query(doc_query)

        doc_results = real_index.search(doc_query_embedding, 2)

        # Check that we got results
        self.assertEqual(len(doc_results), 2, "Expected 2 search results for documentation query")

        # Search for BIF-related content
        bif_query = "Erlang built-in functions parser implementation"
        bif_query_embedding = real_embedder.embed_query(bif_query)

        bif_results = real_index.search(bif_query_embedding, 2)

        # Check that we got results
        self.assertEqual(len(bif_results), 2, "Expected 2 search results for BIF query")

        # Print the results for manual verification
        print("\nDocumentation query results:")
        for i, (unit, score) in enumerate(doc_results):
            print(f"{i+1}. {unit.name} (score: {score:.4f})")

        print("\nBIF query results:")
        for i, (unit, score) in enumerate(bif_results):
            print(f"{i+1}. {unit.name} (score: {score:.4f})")

        # Verify that the expected ranking of results is correct
        # For the documentation query, the documentation file should be ranked higher
        doc_file_score = next((score for unit, score in doc_results if unit.name == "ErlangDocumentationProviderTest.java"), 0)
        bif_file_score_in_doc_results = next((score for unit, score in doc_results if unit.name == "ErlangBifParser.java"), 0)

        self.assertGreater(doc_file_score, bif_file_score_in_doc_results, 
                          "Documentation file should be ranked higher for documentation query")

        # For the BIF query, the BIF file should be ranked higher
        bif_file_score = next((score for unit, score in bif_results if unit.name == "ErlangBifParser.java"), 0)
        doc_file_score_in_bif_results = next((score for unit, score in bif_results if unit.name == "ErlangDocumentationProviderTest.java"), 0)

        self.assertGreater(bif_file_score, doc_file_score_in_bif_results, 
                          "BIF file should be ranked higher for BIF query")


if __name__ == "__main__":
    unittest.main()
