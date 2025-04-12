from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


class RequiredColumns(str, Enum):
    TOXIC_SENTENCE = "toxic_sentence"
    NEUTRAL_SENTENCE = "neutral_sentence"
    LANG = "lang"


def validate_dataframe(
    df: pd.DataFrame, required_columns: List[str], file_path: Path
) -> None:
    """Validate DataFrame structure and content.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        file_path: Path to the file being validated (for error messages)

    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input file {file_path} is missing required columns: {missing_columns}. "
            f"Found columns: {list(df.columns)}"
        )

    # Check for NaN values
    nan_counts = df[required_columns].isna().sum()
    if nan_counts.any():
        raise ValueError(
            f"Input file {file_path} contains NaN values in required columns:\n"
            f"{nan_counts[nan_counts > 0].to_string()}"
        )


def read_and_validate_data(file_path: Path) -> pd.DataFrame:
    """Read and validate a TSV file.

    Args:
        file_path: Path to the TSV file

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If validation fails
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        df = pd.read_csv(file_path, sep="\t")
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file {file_path} is empty")

    validate_dataframe(
        df,
        required_columns=[member.value for member in RequiredColumns],
        file_path=file_path,
    )
    return df


def read_dataframes(
    submission_path: Path,
    reference_path: Path,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Read and validate dataframes from TSV files.

    Args:
        submission_path: Path to submission TSV
        reference_path: Path to reference TSV

    Returns:
        Tuple of (submission_df, reference_df)

    Raises:
        ValueError: If validation fails or files have different lengths
    """
    # Read and validate all files
    submission_df = read_and_validate_data(submission_path)
    reference_df = read_and_validate_data(reference_path)

    # Verify consistent lengths
    if len(submission_df) != len(reference_df):
        raise ValueError(
            f"Input files have different lengths: "
            f"original={len(submission_df)}, rewritten={len(reference_df)}"
        )

    return submission_df, reference_df
