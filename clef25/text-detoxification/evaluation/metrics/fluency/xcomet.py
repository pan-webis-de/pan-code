import logging
import os
from typing import TypedDict

from metrics.fluency.deberta_encoder import XCOMETLite

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.WARNING)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SampleDict(TypedDict):
    """Dictionary structure required for each input sample.

    Attributes:
        mt: Machine-translated text to evaluate
        src: Source text (original text before translation)
        ref: Reference translation (human translation)
    """

    mt: str
    src: str
    ref: str


class CometFluency:
    XCOMET_LITE: str = "myyycroft/XCOMET-lite"

    def __init__(self, model_path: str = XCOMET_LITE):
        """
        Initialize the CometFluency scorer with a pretrained model.

        Args:
            model_path: Path to the pretrained model. Uses the default XCOMET-lite model if not specified.
        """
        self.model = XCOMETLite().from_pretrained(model_path)

    def get_scores(
        self,
        input_data: list[SampleDict],
        batch_size: int = 128,
        gpus: int = 1,
        *args,
        **kwargs
    ) -> list[float]:
        """
        Calculate fluency scores for the input data using the loaded model.

        Args:
            input_data: List of sample dictionaries where each dictionary must contain:
                        - "mt": The machine-translated text to evaluate
                        - "src": The source text (original text before translation)
                        - "ref": The reference translation (human translation)
            batch_size: Number of samples to process at once (default: 128)
            gpus: Number of GPUs to use for computation (default: 1)

        Returns:
            List of fluency scores (float values) corresponding to each input sample.
            Higher scores indicate better fluency.

        Example:
            input_data = [
                {
                    "mt": "The cat is sitting on the mat.",
                    "src": "Le chat est assis sur le tapis.",
                    "ref": "The cat sits on the mat."
                },
                ...
            ]
        """
        return self.model.predict(
            samples=input_data, batch_size=batch_size, gpus=gpus, *args, **kwargs
        ).scores
