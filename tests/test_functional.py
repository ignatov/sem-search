"""
Functional tests for the semantic search system.

These tests perform end-to-end testing of the semantic search system,
including opening a folder from test data, building a new index,
building embeddings, testing queries, and cleaning up.
"""

import unittest
import os
import tempfile
import shutil
import sys
from unittest.mock import patch
import numpy as np
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the semsearch package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semsearch.models import CodeUnit
from semsearch.parsers import UnifiedParser
from semsearch.embedding import CodeEmbedder
from semsearch.indexing import VectorIndex
from semsearch.search import SearchEngine


class TestFunctionalWorkflow(unittest.TestCase):
    """Functional tests for the semantic search workflow."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the index
        self.temp_dir = tempfile.mkdtemp()

        # Path to test data
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data', 'bif')

        # Try to load API key from .env file in parent directory
        parent_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        if os.path.exists(parent_env_path):
            load_dotenv(parent_env_path)
        else:
            load_dotenv()  # Try default locations

        # Get API key from environment or use mock
        self.api_key = os.environ.get("OPENAI_API_KEY", "mock_api_key")

        # Set the API key for OpenAI
        import openai
        openai.api_key = self.api_key

    def tearDown(self):
        """Clean up after tests."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    @patch('openai.embeddings.create')
    def test_end_to_end_workflow(self, mock_embeddings_create):
        """Test the end-to-end workflow of the semantic search system."""
        # Mock the OpenAI embeddings API response
        mock_embeddings_create.return_value.data = [
            type('obj', (object,), {'embedding': np.random.rand(1536).tolist()})
            for _ in range(100)  # Ensure we have enough mock embeddings
        ]

        # Step 1: Parse the test data directory
        parser = UnifiedParser()
        code_units = parser.parse_repository(self.test_data_dir)

        # Verify that we found some code units
        self.assertGreater(len(code_units), 0, "No code units found in test data")
        print(f"Found {len(code_units)} code units in test data")

        # Step 2: Build embeddings
        embedder = CodeEmbedder(cache={}, dimensions=1536)
        embeddings = embedder.embed_code_units(code_units)

        # Verify that we created embeddings for all code units
        self.assertEqual(len(embeddings), len(code_units), 
                         "Number of embeddings doesn't match number of code units")
        print(f"Created {len(embeddings)} embeddings")

        # Step 3: Build the index
        index_path = os.path.join(self.temp_dir, "index")
        index = VectorIndex(dimensions=1536)
        index.build_index(embeddings)

        # Save the index
        index.save(index_path)

        # Verify that the index files were created
        self.assertTrue(os.path.exists(f"{index_path}.index"), "Index file not created")
        self.assertTrue(os.path.exists(f"{index_path}.meta"), "Meta file not created")
        print(f"Index saved to {index_path}")

        # Step 4: Load the index and perform a search
        loaded_index = VectorIndex.load(index_path)
        search_engine = SearchEngine(embedder, loaded_index)

        # Test a query
        query = "parse Erlang BIF functions"
        results = search_engine.search(query, top_k=5)

        # Verify that we got some results
        self.assertGreater(len(results), 0, "No search results found")
        print(f"Found {len(results)} results for query: '{query}'")

        # Verify the structure of the results
        for unit, score in results:
            self.assertIsInstance(unit, CodeUnit, "Result is not a CodeUnit")
            # Score can be either float or np.float32
            self.assertTrue(isinstance(score, (float, np.float32)), 
                           f"Score is not a float or np.float32, got {type(score)}")
            self.assertTrue(0 <= score <= 1, "Score is not between 0 and 1")
            print(f"Result: {unit.name} ({score:.2f})")


if __name__ == "__main__":
    unittest.main()
