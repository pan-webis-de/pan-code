import logging
from typing import List, Optional

import numpy as np
from pydantic import BaseModel
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm import trange

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)


class SimilarityConfig(BaseModel):
    """Configuration for similarity measurement."""

    sim_model_name: str = "sentence-transformers/LaBSE"
    device: str = "cuda"
    batch_size: int = 32
    efficient_version: bool = False
    reference_weight: float = (
        0.6  # Weight for reference similarity when references are provided
    )
    original_weight: float = (
        0.4  # Weight for original similarity when references are provided
    )


class SimilarityMeasurement:
    """Class for measuring similarity between original and rewritten texts."""

    def __init__(self, config: Optional[SimilarityConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or SimilarityConfig()
        self.model = SentenceTransformer(
            self.config.sim_model_name, device=self.config.device
        )

    def _evaluate_batch_similarity(
        self, original_embeddings: np.ndarray, rewritten_embeddings: np.ndarray
    ) -> List[float]:
        """Calculate similarity scores for a batch of embeddings."""
        if self.config.efficient_version:
            similarity_matrix = np.dot(original_embeddings, rewritten_embeddings.T)
            original_norms = np.linalg.norm(original_embeddings, axis=1)
            rewritten_norms = np.linalg.norm(rewritten_embeddings, axis=1)
            return (
                1
                - similarity_matrix / (np.outer(original_norms, rewritten_norms) + 1e-9)
            ).tolist()
        else:
            return [
                1 - cosine(original_embedding, rewritten_embedding)
                for original_embedding, rewritten_embedding in zip(
                    original_embeddings, rewritten_embeddings
                )
            ]

    def _calculate_pairwise_similarity(
        self, texts_a: List[str], texts_b: List[str]
    ) -> List[float]:
        """Calculate pairwise similarity between two lists of texts."""
        similarities = []
        batch_size = min(self.config.batch_size, len(texts_a))

        for i in trange(
            0, len(texts_a), batch_size, desc="Calculating similarity scores"
        ):
            batch_a = texts_a[i : i + batch_size]
            batch_b = texts_b[i : i + batch_size]

            embeddings = self.model.encode(batch_a + batch_b)
            embeddings_a = embeddings[: len(batch_a)]
            embeddings_b = embeddings[len(batch_a) :]

            batch_similarity = self._evaluate_batch_similarity(
                embeddings_a, embeddings_b
            )
            similarities.extend(batch_similarity)

        return similarities

    def evaluate_similarity(
        self,
        original_texts: List[str],
        rewritten_texts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> List[float]:
        """
        Evaluate similarity between original and rewritten texts, optionally using references.

        Args:
            original_texts: List of original texts
            rewritten_texts: List of rewritten texts
            reference_texts: Optional list of reference texts

        Returns:
            List of similarity scores
        """
        if len(original_texts) != len(rewritten_texts):
            raise ValueError("Original and rewritten texts must have the same length")

        if reference_texts is not None and len(reference_texts) != len(rewritten_texts):
            raise ValueError("References must have the same length as rewritten texts")

        # Calculate similarity between original and rewritten texts
        original_similarity = self._calculate_pairwise_similarity(
            original_texts, rewritten_texts
        )

        if reference_texts is None:
            return original_similarity

        # Calculate similarity between references and rewritten texts
        reference_similarity = self._calculate_pairwise_similarity(
            reference_texts, rewritten_texts
        )

        # Combine scores using weighted average
        combined_similarity = (
            np.array(original_similarity) * self.config.original_weight
            + np.array(reference_similarity) * self.config.reference_weight
        )

        return combined_similarity.tolist()
