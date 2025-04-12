from typing import List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from tqdm.auto import trange
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ToxicityConfig(BaseModel):
    """Configuration for toxicity measurement"""

    tox_model_name: str = "textdetox/xlmr-large-toxicity-classifier-v2"
    tokenizer_name: str = "cardiffnlp/twitter-xlm-roberta-large-2022"
    target_label: int = 0  # 1 is toxic, 0 is neutral
    batch_size: int = 32
    max_length: int = 512
    device: str = "cuda"


class ToxicityMeasurement:
    """Class for measuring toxicity scores of texts"""

    def __init__(self, config: Optional[ToxicityConfig] = None):
        """
        Initialize toxicity measurement with config.

        Args:
            config: Configuration object. Uses defaults if None.
        """
        self.config = config if config is not None else ToxicityConfig()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model and tokenizer"""
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                self.config.tox_model_name
            )
            .to(self.config.device)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

    def classify_texts(
        self,
        texts: List[str],
        desc: Optional[str] = "Calculating toxicity scores",
    ) -> list[float]:
        """
        Classify texts and return scores for the target label.

        Args:
            texts: List of texts to classify
            desc: Description for progress bar

        Returns:
            List of scores for the target label for each text
        """
        res = []

        for i in trange(0, len(texts), self.config.batch_size, desc=desc):
            batch_texts = texts[i : i + self.config.batch_size]

            try:
                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.model.device)

                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    # Handle both binary and multi-class classification
                    if logits.shape[-1] > 1:  # Multi-class
                        scores = torch.softmax(logits, dim=-1)[
                            :, self.config.target_label
                        ]
                    else:  # Binary classification
                        scores = torch.sigmoid(logits).squeeze(-1)

                    res.extend(scores.cpu().numpy().tolist())

            except Exception as e:
                # Log error and return zeros for failed batch
                print(f"Error processing batch {i//self.config.batch_size}: {str(e)}")
                res.extend([0.0] * len(batch_texts))

        return res

    def evaluate_toxicity(
        self, texts: List[str], desc: Optional[str] = "Toxicity classifier"
    ) -> list[float]:
        """
        Evaluate toxicity scores for given texts.

        Args:
            texts: List of texts to evaluate
            desc: Description for progress bar

        Returns:
            List of toxicity scores for each text
        """
        return self.classify_texts(texts, desc=desc)

    def compare_toxicity(
        self,
        original_texts: List[str],
        rewritten_texts: List[str],
        reference_texts: Optional[List[str]] = None,
    ) -> list[float]:
        """
        Compare toxicity between rewritten, original and optionally reference texts.

        Args:
            rewritten_texts: List of detoxified texts
            original_texts: List of original toxic texts
            reference_texts: Optional list of reference neutral texts

        Returns:
            List of combined toxicity comparison scores
        """
        # Calculate scores for all text sets
        input_scores = self.evaluate_toxicity(
            original_texts, desc="Evaluating original texts"
        )

        predicted_scores = self.evaluate_toxicity(
            rewritten_texts, desc="Evaluating rewritten texts"
        )

        # Convert to numpy arrays for vector operations
        predicted_np = np.array(predicted_scores)
        input_np = np.array(input_scores)

        # First comparison: rewritten vs original
        compared_scores1 = (input_np <= predicted_np).astype(float)
        combined_scores = ((predicted_np + compared_scores1) / 2).tolist()

        # Optional second comparison: rewritten vs references
        if reference_texts is not None:
            ref_scores = np.array(
                self.evaluate_toxicity(
                    reference_texts, desc="Evaluating reference texts"
                )
            )
            compared_scores_w_ref = (ref_scores <= predicted_np).astype(float)
            combined_scores = np.maximum(
                compared_scores_w_ref, np.array(combined_scores)
            ).tolist()

        return combined_scores
