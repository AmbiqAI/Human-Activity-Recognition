import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path

from pydantic import BaseModel, Field

class TrainParams(BaseModel):
    seed: int = Field(42, description="Random Seed")
    num_time_steps: int = Field(200, description="Number of Timestep Windows")
    time_steps: int = Field(20, description="Timestep Window Slide")
    trained_model_dir: Path = Field("trained_models/", description="Directory where trained models are stored")
    dataset_dir: Path =Field(".", description="Directory where datasets reside")
    processed_dataset: str = Field("processed_dataset.pkl", description="Name of processed baseline dataset")
    augmented_dataset: str = Field("augmented_dataset.pkl", description="Name of processed augmented dataset")
    batch_size: int = Field(400, description="Batch Size")
    augmentations: int = Field(4, description="Number of augmentation passes")
    epochs: int =Field(10, description="Number of training epochs")
    model_name: str = Field("model", description="Name of trained model")
