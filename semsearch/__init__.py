# This file re-exports the public API for backward compatibility

# Import from the new modules
from semsearch.models import CodeUnit
from semsearch.parsers import GenericFileParser, UnifiedParser
from semsearch.embedding import CodeEmbedder
from semsearch.indexing import VectorIndex
from semsearch.search import (
    SearchEngine,
    list_available_indexes,
    search,
    display_results,
    interactive_search
)
from semsearch.utils import build_index
from semsearch.main import main

# Version information
__version__ = '0.1.0'
