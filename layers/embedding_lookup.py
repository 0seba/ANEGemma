import numpy as np
import os
import mmap
import sys


class EmbeddingLookup:
    """
    Manages access to chunked embedding weights from a binary file using
    memory mapping (mmap).

    Use as a context manager (`with EmbeddingLookup(...) as lookup:`).
    """

    FLOAT16_SIZE = np.dtype(np.float16).itemsize

    def __init__(
        self,
        file_path: str,
        file_offset: int,
        shape: tuple[int, int],
        chunk_size: int,
        padding_bytes: int,
    ):
        """
        Initializes the mmap embedding lookup.

        Args:
            file_path: Path to the embedding weights file.
            file_offset: Byte offset where the first chunk starts.
            shape: Tuple (vocab_size, embedding_dim).
            chunk_size: Number of tokens per chunk.
            padding_bytes: Bytes between chunks.
        """
        self.file_path = file_path
        self.file_offset = file_offset
        self.vocab_size, self.embedding_dim = shape
        self.chunk_size = chunk_size
        self.padding_bytes = padding_bytes

        self.file_obj = None
        self.mapped_mmap = None
        self.map_size: int = 0
        self.map_start_offset: int = 0

        # --- Validations ---
        if not (self.vocab_size > 0 and self.embedding_dim > 0 and self.chunk_size > 0):
            raise ValueError(
                "Vocab size, embedding dim, and chunk size must be positive."
            )
        if not (self.file_offset >= 0 and self.padding_bytes >= 0):
            raise ValueError("File offset and padding bytes must be non-negative.")
        if self.vocab_size % self.chunk_size != 0:
            raise ValueError(
                f"Vocab size ({self.vocab_size}) must be divisible by chunk size ({self.chunk_size})."
            )
        if self.FLOAT16_SIZE != 2:
            raise TypeError(f"Expected float16 size 2, got {self.FLOAT16_SIZE}.")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Embedding file not found: {self.file_path}")

        # --- Calculate Derived Sizes ---
        self.row_size_in_bytes = self.embedding_dim * self.FLOAT16_SIZE
        self.chunk_size_bytes = self.chunk_size * self.row_size_in_bytes
        self.chunk_stride_bytes = self.chunk_size_bytes + self.padding_bytes
        self.num_chunks = self.vocab_size // self.chunk_size
        if self.num_chunks == 0:
            self.total_file_size_span = 0
        else:
            last_chunk_start_relative_offset = (
                self.num_chunks - 1
            ) * self.chunk_stride_bytes
            self.total_file_size_span = (
                last_chunk_start_relative_offset + self.chunk_size_bytes
            )

        # --- Map Data ---
        try:
            self._map_memory_chunked()
        except Exception as e:
            self.close()  # Ensure cleanup if init fails
            raise e

    def _map_memory_chunked(self):
        """Sets up memory mapping."""
        try:
            self.file_obj = open(self.file_path, "rb")
            fileno = self.file_obj.fileno()

            try:
                page_size = mmap.PAGESIZE
            except AttributeError:
                page_size = os.sysconf("SC_PAGESIZE")  # Fallback

            if page_size <= 0:
                raise ValueError("Invalid page size.")

            self.map_start_offset = (self.file_offset // page_size) * page_size
            buffer_internal_offset = self.file_offset - self.map_start_offset
            self.map_size = buffer_internal_offset + self.total_file_size_span

            if self.map_size <= 0:
                self.mapped_mmap = None  # Nothing to map
                # print("Warning: Map size is zero or negative. No mapping performed.")
                return

            self.mapped_mmap = mmap.mmap(
                fileno,
                self.map_size,
                access=mmap.ACCESS_READ,
                offset=self.map_start_offset,
            )

        except (OSError, ValueError) as e:
            if self.file_obj:
                self.file_obj.close()
                self.file_obj = None
            raise MemoryError(f"Memory mapping failed: {e}") from e
        except Exception as e:  # Catch other potential errors
            if self.file_obj:
                self.file_obj.close()
                self.file_obj = None
            raise MemoryError(f"Unexpected error during mmap setup: {e}") from e

    def embedding(self, token_id: int) -> np.ndarray:
        """
        Retrieves the embedding vector (as a NumPy array view) for a token ID.

        Returns a view (no copy). Call `.copy()` on the result if needed.
        """
        if not (0 <= token_id < self.vocab_size):
            raise IndexError(
                f"Token index {token_id} out of bounds (size: {self.vocab_size})."
            )

        if self.mapped_mmap is None:
            # This case occurs if map_size was <= 0 during init (e.g., vocab_size=0)
            raise RuntimeError("Memory map not initialized (vocab size might be zero).")

        # Calculate offset within the mapped region
        chunk_index = token_id // self.chunk_size
        index_in_chunk = token_id % self.chunk_size
        chunk_start_offset = chunk_index * self.chunk_stride_bytes
        row_start_in_chunk = index_in_chunk * self.row_size_in_bytes
        target_row_offset_in_file = (
            self.file_offset + chunk_start_offset + row_start_in_chunk
        )
        row_offset_in_map = target_row_offset_in_file - self.map_start_offset

        # Bounds check
        if not (
            0 <= row_offset_in_map
            and (row_offset_in_map + self.row_size_in_bytes) <= self.map_size
        ):
            raise RuntimeError(
                f"Calculated offset [{row_offset_in_map}] out of mapped bounds [0..{self.map_size}]."
            )

        # Extract view using frombuffer
        try:
            vector = np.frombuffer(
                self.mapped_mmap,
                dtype=np.float16,
                count=self.embedding_dim,
                offset=row_offset_in_map,
            )
            if vector.size != self.embedding_dim:  # Sanity check
                raise BufferError(
                    f"Extracted vector size {vector.size} != expected {self.embedding_dim}."
                )
            return vector
        except Exception as e:
            raise BufferError(
                f"Error accessing memory map for token {token_id} at offset {row_offset_in_map}: {e}"
            ) from e

    def embeddings_for_ids(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Retrieves embedding vectors for multiple token IDs.

        Args:
            token_ids: A 1D NumPy array of integer token IDs.

        Returns:
            A 2D NumPy array of shape (len(token_ids), embedding_dim)
            containing the corresponding embeddings (dtype=float16).
            This array is a *copy* of the data from the memory map.
        """
        # Input validation (optional but recommended)
        if not isinstance(token_ids, np.ndarray) or token_ids.ndim != 1:
            raise TypeError("Input token_ids must be a 1D NumPy array.")
        if not np.issubdtype(token_ids.dtype, np.integer):
            raise TypeError("Input token_ids must be of integer type.")
        # Check bounds using vectorized operations
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            # Find the first offending ID for a more informative error
            offending_id = token_ids[
                np.logical_or(token_ids < 0, token_ids >= self.vocab_size)
            ][0]
            raise IndexError(
                f"Token index {offending_id} out of bounds (size: {self.vocab_size})."
            )

        # Retrieve embeddings individually using a list comprehension
        # This leverages the existing `embedding` method which returns views
        embedding_list = [self.embedding(int(token_id)) for token_id in token_ids]

        if not embedding_list:  # Handle empty input array
            return np.empty((0, self.embedding_dim), dtype=np.float16)

        # np.stack creates a new array, copying data from the individual views
        try:
            return np.stack(embedding_list, axis=0)
        except ValueError as e:
            # This might happen if somehow embedding views had inconsistent shapes
            raise RuntimeError(f"Failed to stack embedding vectors: {e}") from e

    def close(self):
        """Closes the memory map and file handle if they are open."""
        if self.mapped_mmap is not None:
            try:
                self.mapped_mmap.close()
            except Exception:
                pass  # Ignore errors on close
            self.mapped_mmap = None
        if self.file_obj is not None:
            try:
                self.file_obj.close()
            except Exception:
                pass  # Ignore errors on close
            self.file_obj = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
