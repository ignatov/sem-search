"""
Tests for the embedding module.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from semsearch.models import CodeUnit
from semsearch.embedding import CodeEmbedder


class TestCodeEmbedder(unittest.TestCase):
    """Tests for the CodeEmbedder class."""

    def setUp(self):
        """Set up test fixtures."""
        self.embedder = CodeEmbedder(cache={}, dimensions=4)
        self.code_unit = CodeUnit(
            path="path/to/file.py",
            content="def hello(): pass",
            unit_type="function",
            name="hello"
        )
        self.code_unit2 = CodeUnit(
            path="path/to/file2.py",
            content="def world(): pass",
            unit_type="function",
            name="world"
        )
        # Create a hash for the code unit to use in the cache
        self.content_hash = self.code_unit.get_content_hash()
        # Create a sample embedding
        self.sample_embedding = np.array([0.1, 0.2, 0.3, 0.4])

    @patch.object(CodeEmbedder, '_get_embeddings')
    def test_embed_query(self, mock_get_embeddings):
        """Test embed_query method."""
        # Mock the _get_embeddings method to return a sample embedding
        mock_get_embeddings.return_value = [np.array([0.1, 0.2, 0.3, 0.4])]

        # Call the method
        result = self.embedder.embed_query("test query")

        # Check the result
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4,))
        np.testing.assert_array_almost_equal(result, np.array([0.1, 0.2, 0.3, 0.4]))

        # Check that the method was called correctly
        mock_get_embeddings.assert_called_once_with(["test query"])

    @patch.object(CodeEmbedder, '_get_embeddings')
    def test_embed_code_units_with_cache(self, mock_get_embeddings):
        """Test embed_code_units method with cache hit."""
        # Set up a cache with a pre-computed embedding
        embedder = CodeEmbedder(
            cache={self.content_hash: self.sample_embedding},
            dimensions=4
        )

        # Call the method
        result = embedder.embed_code_units([self.code_unit])

        # Check the result
        self.assertIn(self.code_unit, result)
        np.testing.assert_array_almost_equal(
            result[self.code_unit],
            self.sample_embedding
        )

        # Check that the method was not called
        mock_get_embeddings.assert_not_called()

    @patch.object(CodeEmbedder, '_get_embeddings')
    def test_embed_code_units_without_cache(self, mock_get_embeddings):
        """Test embed_code_units method with cache miss."""
        # Mock the _get_embeddings method to return a sample embedding
        mock_get_embeddings.return_value = [np.array([0.1, 0.2, 0.3, 0.4])]

        # Call the method
        result = self.embedder.embed_code_units([self.code_unit])

        # Check the result
        self.assertIn(self.code_unit, result)
        np.testing.assert_array_almost_equal(
            result[self.code_unit],
            np.array([0.1, 0.2, 0.3, 0.4])
        )

        # Check that the method was called correctly
        mock_get_embeddings.assert_called_once_with(["def hello(): pass"])

        # Check that the result was cached
        self.assertIn(self.content_hash, self.embedder.cache)
        np.testing.assert_array_almost_equal(
            self.embedder.cache[self.content_hash],
            np.array([0.1, 0.2, 0.3, 0.4])
        )

    @patch.object(CodeEmbedder, '_get_embeddings')
    def test_embed_multiple_code_units(self, mock_get_embeddings):
        """Test embed_code_units method with multiple code units."""
        # Mock the _get_embeddings method to return sample embeddings
        mock_get_embeddings.return_value = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8])
        ]

        # Call the method
        result = self.embedder.embed_code_units([self.code_unit, self.code_unit2])

        # Check the result
        self.assertIn(self.code_unit, result)
        self.assertIn(self.code_unit2, result)
        np.testing.assert_array_almost_equal(
            result[self.code_unit],
            np.array([0.1, 0.2, 0.3, 0.4])
        )
        np.testing.assert_array_almost_equal(
            result[self.code_unit2],
            np.array([0.5, 0.6, 0.7, 0.8])
        )

        # Check that the method was called correctly
        mock_get_embeddings.assert_called_once_with(["def hello(): pass", "def world(): pass"])

        # Check that the results were cached
        self.assertIn(self.content_hash, self.embedder.cache)
        np.testing.assert_array_almost_equal(
            self.embedder.cache[self.content_hash],
            np.array([0.1, 0.2, 0.3, 0.4])
        )
        self.assertIn(self.code_unit2.get_content_hash(), self.embedder.cache)
        np.testing.assert_array_almost_equal(
            self.embedder.cache[self.code_unit2.get_content_hash()],
            np.array([0.5, 0.6, 0.7, 0.8])
        )

    def test_sanitize_text(self):
        """Test _sanitize_text method."""
        # Test basic sanitization
        text = "Hello\x00World"
        result = self.embedder._sanitize_text(text)
        self.assertEqual(result, "Hello World")

        # Test aggressive sanitization
        text = "Hello\x00World\uD800\uDFFF"
        result = self.embedder._sanitize_text(text, aggressive=True)
        self.assertEqual(result, "Hello World ")

        # Test empty text
        self.assertEqual(self.embedder._sanitize_text(""), "")
        self.assertEqual(self.embedder._sanitize_text(None), "")

        # Test aggressive sanitization with empty result
        text = "\uD800\uDFFF"
        result = self.embedder._sanitize_text(text, aggressive=True)
        self.assertEqual(result, "empty_content")


if __name__ == "__main__":
    unittest.main()
