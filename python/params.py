import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path

from pydantic import BaseModel, Field

class TrainParams(BaseModel):
    seed: int = Field(42, description="Random Seed")
    num_time_steps: int = Field(200, description="Number of Timestep Windows")
    num_features: int = Field(6, description="Number of Features")
    sample_step: int = Field(20, description="Timestep Window Slide")
    trained_model_dir: Path = Field("trained_models/", description="Directory where trained models are stored")
    job_dir: Path = Field("artifacts/", description="Directory where artifacts are stored")
    dataset_dir: Path =Field(".", description="Directory where datasets reside")
    processed_dataset: str = Field("processed_dataset.pkl", description="Name of processed baseline dataset")
    augmented_dataset: str = Field("augmented_dataset.pkl", description="Name of processed augmented dataset")
    batch_size: int = Field(32, description="Batch Size")
    augmentations: int = Field(8, description="Number of augmentation passes")
    save_processed_dataset: bool = Field(True, description="Save processed datasets as pkls")
    epochs: int =Field(30, description="Number of training epochs")
    model_name: str = Field("model", description="Name of trained model")
    training_dataset_percent: int = Field(70, description="Percent of records used for training")
    show_training_plot: bool = Field(True, description="Show training statistics plots")
    train_model: bool = Field(False, description="Train the model, otherwise load existing model")

