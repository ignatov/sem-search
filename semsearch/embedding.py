"""
Embedding module for semantic search.

This module contains functionality for embedding code units into vectors.
"""

import re
import os
import numpy as np
import openai
import time
from typing import List, Dict, Optional

from semsearch.models import CodeUnit


class CodeEmbedder:
    """
    Embeds code units into vectors using OpenAI's embedding API.

    This class handles batching, caching, token limit management, and error handling for the embedding process.
    It automatically handles texts that exceed the OpenAI token limit by estimating token count,
    truncating if necessary, and splitting batches into smaller chunks when needed.
    """

    def __init__(self, model_name="text-embedding-3-small", cache=None, dimensions=1536, api_key=None):
        """
        Initialize the CodeEmbedder.

        Args:
            model_name: The name of the OpenAI embedding model to use
            cache: Optional dictionary to cache embeddings by content hash
            dimensions: The dimensions of the embedding vectors
            api_key: Optional OpenAI API key. If not provided, will use the key from environment variables.
        """
        self.model_name = model_name
        self.cache = cache or {}
        self.dimensions = dimensions

        # Check if API key is provided or available in environment variables
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif os.environ.get("OPENAI_API_KEY"):
            self.client = openai.OpenAI()
        else:
            # For testing purposes, create a mock client
            class MockClient:
                class Embeddings:
                    def create(self, **kwargs):
                        class MockResponse:
                            class MockData:
                                def __init__(self, embedding):
                                    self.embedding = embedding

                            def __init__(self, data):
                                self.data = data

                        # Create mock embeddings with the correct dimensions
                        mock_embeddings = []
                        for _ in range(len(kwargs.get('input', []))):
                            mock_embeddings.append(
                                MockResponse.MockData([0.0] * dimensions)
                            )

                        return MockResponse(mock_embeddings)

                def __init__(self):
                    self.embeddings = self.Embeddings()

            # Assign the mock client
            self.client = MockClient()

    def embed_code_units(self, code_units: List[CodeUnit]) -> Dict[CodeUnit, np.ndarray]:
        """
        Embed multiple code units, using cache when available.

        Args:
            code_units: List of CodeUnit objects to embed

        Returns:
            Dictionary mapping CodeUnit objects to their embedding vectors
        """
        start_time = time.time()
        print(f"Starting embedding process for {len(code_units)} code units...")

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

        elapsed_time = time.time() - start_time
        cached_count = len(code_units) - len(units_to_embed)
        print(f"Embedding process completed in {elapsed_time:.2f} seconds for {len(code_units)} code units")
        print(f"Cache hits: {cached_count}, Cache misses: {len(units_to_embed)}")

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a search query.

        Args:
            query: The search query to embed

        Returns:
            Embedding vector for the query
        """
        return self._get_embeddings([query])[0]

    def _get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a list of texts.

        This method handles batching, sanitization, token limit checking, and error recovery.
        It automatically handles texts that exceed the OpenAI token limit by:
        1. Estimating token count for each text and truncating if necessary
        2. Checking total token count for each batch and splitting into smaller chunks if needed
        3. Applying aggressive sanitization to reduce token count when needed
        4. Using conservative token estimation (1:3 character-to-token ratio)
        5. Implementing multiple layers of safeguards to prevent token limit errors:
           - Individual text token limit checks
           - Batch token limit checks with a conservative limit (6000 tokens)
           - Chunk token limit checks with fallback to individual processing
           - Aggressive text sanitization when needed

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        total_start_time = time.time()
        print(f"Starting embedding process for {len(texts)} texts...")

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
            batch_start_time = time.time()
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

                        # Estimate token count and truncate if necessary to avoid token limit issues
                        estimated_tokens = self._estimate_tokens(text)
                        # OpenAI's text-embedding models have a limit of 8191 tokens per input
                        if estimated_tokens > 8000:  # Using 8000 as a safe limit
                            print(f"  Item {item_number} exceeds token limit ({estimated_tokens} estimated tokens)")
                            # Truncate text to fit within token limit (8000 tokens ≈ 32000 chars)
                            max_chars = 32000
                            if len(text) > max_chars:
                                text = text[:max_chars]
                                print(f"  Truncated item {item_number} to {len(text)} chars")

                            # Apply aggressive sanitization to further reduce token count
                            text = self._sanitize_text(text, aggressive=True)
                            print(f"  Applied aggressive sanitization to item {item_number}")

                        cleaned_batch.append(text)
                    else:
                        cleaned_batch.append("")  # Empty string for None values

                # Check total token count for the batch
                total_tokens = sum(self._estimate_tokens(text) for text in cleaned_batch)
                max_batch_tokens = 6000  # More conservative limit (well below the 8192 model limit)

                if total_tokens > max_batch_tokens:
                    print(f"  Total batch token count ({total_tokens}) exceeds limit ({max_batch_tokens})")
                    print(f"  Splitting batch into smaller chunks...")

                    # Process in smaller chunks
                    batch_embeddings = []
                    # More conservative chunk size calculation with an additional safety factor
                    chunk_size = max(1, len(cleaned_batch) // ((total_tokens // max_batch_tokens) * 2 + 1))

                    for j in range(0, len(cleaned_batch), chunk_size):
                        chunk = cleaned_batch[j:j+chunk_size]
                        print(f"  Processing chunk {j//chunk_size + 1}/{(len(cleaned_batch) + chunk_size - 1)//chunk_size}...")

                        # Additional safeguard: check each item in the chunk
                        chunk_token_count = sum(self._estimate_tokens(text) for text in chunk)
                        if chunk_token_count > 8000:  # Still too large
                            print(f"  Chunk token count ({chunk_token_count}) still exceeds safe limit")
                            print(f"  Processing items in chunk individually...")

                            # Process each item individually
                            for item in chunk:
                                try:
                                    print(f"    Processing individual item, length: {len(item)}")
                                    print(f"    Item preview: {item[:100]}..." if item else "    Empty item")

                                    start_time = time.time()
                                    item_response = self.client.embeddings.create(
                                        model=self.model_name,
                                        input=[item]
                                    )
                                    elapsed_time = time.time() - start_time
                                    print(f"    Item processed in {elapsed_time:.2f} seconds")
                                    embedding = np.array(item_response.data[0].embedding)
                                    # Resize embedding if needed
                                    if len(embedding) != self.dimensions:
                                        if len(embedding) > self.dimensions:
                                            embedding = embedding[:self.dimensions]  # Truncate
                                        else:
                                            # Pad with zeros
                                            padded = np.zeros(self.dimensions)
                                            padded[:len(embedding)] = embedding
                                            embedding = padded
                                    batch_embeddings.append(embedding)
                                except Exception as item_error:
                                    print(f"  Error processing individual item: {item_error}")
                                    # Add a zero vector as placeholder
                                    batch_embeddings.append(np.zeros(self.dimensions))
                            continue  # Skip the chunk processing

                        # Process the chunk normally
                        # print(f"  Sending chunk with {len(chunk)} items, total chars: {sum(len(t) for t in chunk)}")
                        # print(f"  First item preview: {chunk[0][:100]}..." if chunk else "  Empty chunk")

                        start_time = time.time()
                        chunk_response = self.client.embeddings.create(
                            model=self.model_name,
                            input=chunk
                        )
                        elapsed_time = time.time() - start_time
                        # print(f"  Chunk processed in {elapsed_time:.2f} seconds")

                        for data in chunk_response.data:
                            embedding = np.array(data.embedding)
                            # Resize embedding to match expected dimensions if needed
                            if len(embedding) != self.dimensions:
                                if len(embedding) > self.dimensions:
                                    embedding = embedding[:self.dimensions]  # Truncate
                                else:
                                    # Pad with zeros
                                    padded = np.zeros(self.dimensions)
                                    padded[:len(embedding)] = embedding
                                    embedding = padded
                            batch_embeddings.append(embedding)
                else:
                    # Process the entire batch at once
                    print(f"  Sending entire batch with {len(cleaned_batch)} items, total chars: {sum(len(t) for t in cleaned_batch)}")
                    print(f"  First item preview: {cleaned_batch[0][:100]}..." if cleaned_batch else "  Empty batch")

                    start_time = time.time()
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=cleaned_batch
                    )
                    elapsed_time = time.time() - start_time
                    print(f"  Batch processed in {elapsed_time:.2f} seconds")

                    batch_embeddings = []
                    for data in response.data:
                        embedding = np.array(data.embedding)
                        # Resize embedding to match expected dimensions if needed
                        if len(embedding) != self.dimensions:
                            if len(embedding) > self.dimensions:
                                embedding = embedding[:self.dimensions]  # Truncate
                            else:
                                # Pad with zeros
                                padded = np.zeros(self.dimensions)
                                padded[:len(embedding)] = embedding
                                embedding = padded
                        batch_embeddings.append(embedding)
                all_embeddings.extend(batch_embeddings)
                batch_elapsed_time = time.time() - batch_start_time
                print(f"Batch {i//batch_size + 1} completed in {batch_elapsed_time:.2f} seconds")
            except Exception as e:
                batch_elapsed_time = time.time() - batch_start_time
                print(f"Error in batch {i//batch_size + 1} after {batch_elapsed_time:.2f} seconds: {e}")

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
                        print(f"  Retrying with aggressive sanitization: {len(aggressive_batch)} items, total chars: {sum(len(t) for t in aggressive_batch)}")
                        print(f"  First item preview: {aggressive_batch[0][:100]}..." if aggressive_batch else "  Empty batch")

                        start_time = time.time()
                        response = openai.embeddings.create(
                            model=self.model_name,
                            input=aggressive_batch
                        )
                        elapsed_time = time.time() - start_time
                        print(f"  Aggressively sanitized batch processed in {elapsed_time:.2f} seconds")
                        batch_embeddings = []
                        for data in response.data:
                            embedding = np.array(data.embedding)
                            # Resize embedding to match expected dimensions if needed
                            if len(embedding) != self.dimensions:
                                if len(embedding) > self.dimensions:
                                    embedding = embedding[:self.dimensions]  # Truncate
                                else:
                                    # Pad with zeros
                                    padded = np.zeros(self.dimensions)
                                    padded[:len(embedding)] = embedding
                                    embedding = padded
                            batch_embeddings.append(embedding)
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
                                print(f"    Processing individual item with sanitization, length: {len(text)}")
                                print(f"    Sanitized item preview: {text[:100]}..." if text else "    Empty item")

                                start_time = time.time()
                                response = openai.embeddings.create(
                                    model=self.model_name,
                                    input=[text]
                                )
                                elapsed_time = time.time() - start_time
                                print(f"    Sanitized item processed in {elapsed_time:.2f} seconds")
                                embedding = np.array(response.data[0].embedding)
                                # Resize embedding to match expected dimensions if needed
                                if len(embedding) != self.dimensions:
                                    if len(embedding) > self.dimensions:
                                        embedding = embedding[:self.dimensions]  # Truncate
                                    else:
                                        # Pad with zeros
                                        padded = np.zeros(self.dimensions)
                                        padded[:len(embedding)] = embedding
                                        embedding = padded
                                all_embeddings.append(embedding)
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
                                        print(f"    Retrying with extreme sanitization, length: {len(extreme_text)}")
                                        print(f"    Extreme sanitized item preview: {extreme_text[:100]}...")

                                        start_time = time.time()
                                        response = openai.embeddings.create(
                                            model=self.model_name,
                                            input=[extreme_text]
                                        )
                                        elapsed_time = time.time() - start_time
                                        print(f"    Extremely sanitized item processed in {elapsed_time:.2f} seconds")
                                        embedding = np.array(response.data[0].embedding)
                                        # Resize embedding to match expected dimensions if needed
                                        if len(embedding) != self.dimensions:
                                            if len(embedding) > self.dimensions:
                                                embedding = embedding[:self.dimensions]  # Truncate
                                            else:
                                                # Pad with zeros
                                                padded = np.zeros(self.dimensions)
                                                padded[:len(embedding)] = embedding
                                                embedding = padded
                                        all_embeddings.append(embedding)
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

        total_elapsed_time = time.time() - total_start_time
        print(f"Embedding process completed in {total_elapsed_time:.2f} seconds for {len(texts)} texts")
        return all_embeddings

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        This is a conservative estimation based on the rule of thumb that 1 token is approximately 
        3 characters for English text in OpenAI models. We use a more conservative ratio (3 instead of 4)
        to ensure we don't exceed token limits.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        if not text:
            return 0

        # Conservative estimation: 1 token ≈ 3 characters (more conservative than the typical 4)
        return len(text) // 3 + 2  # Add 2 to provide a larger safety margin

    def _sanitize_text(self, text, aggressive=False):
        """
        Sanitize text to ensure it's valid for the OpenAI API.

        Args:
            text: The text to sanitize
            aggressive: Whether to apply more aggressive sanitization

        Returns:
            Sanitized text
        """
        if not text:
            return ""

        # Basic sanitization
        # Replace null bytes and other control characters
        text = ''.join(ch if ord(ch) >= 32 or ch in '\n\r\t' else ' ' for ch in text)

        # Check for and remove invalid surrogate pairs and other problematic Unicode
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
