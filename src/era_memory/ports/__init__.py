"""
The nine ports. All core logic depends on these interfaces only — never a backend.

Pattern mirrors era-core's existing KmsProvider ABC, generalized to all touchpoints.
"""

from .auth import Auth
from .blob_store import BlobStore
from .embedder import Embedder
from .extractor import Extractor
from .kms import KMS
from .queue import Queue
from .record_store import RecordStore, UnitOfWork
from .telemetry import Telemetry
from .vector_store import VectorStore

__all__ = [
    "Auth",
    "BlobStore",
    "Embedder",
    "Extractor",
    "KMS",
    "Queue",
    "RecordStore",
    "UnitOfWork",
    "Telemetry",
    "VectorStore",
]
