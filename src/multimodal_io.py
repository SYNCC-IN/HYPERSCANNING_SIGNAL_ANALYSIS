"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import os
from typing import Optional
import joblib
from . import dataloader
from .data_structures import MultimodalData


def save_to_file(multimodal_data: MultimodalData, output_dir: str) -> None:
    """
    Save MultimodalData instance to a joblib file.

    Args:
        multimodal_data: The multimodal data instance to save.
        output_dir: Directory path where the file will be saved.

    Returns:
        None: Saves file to {output_dir}/{dyad_id}.joblib
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{multimodal_data.id}.joblib")
    joblib.dump(multimodal_data, output_path)


def load_output_data(filename: str) -> MultimodalData | None:
    """
    Load saved MultimodalData from a joblib file.

    .. warning::
        Uses ``joblib.load`` which relies on pickle under the hood.
        Never load files from untrusted sources as they may execute
        arbitrary code during deserialization.

    Args:
        filename: Path to the joblib file to load.

    Returns:
        MultimodalData or None: The loaded multimodal data instance, or None if file not found.
    """
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")
        return None

