# config.py
import os
import torch
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """
    Configuration settings for the model training process.
    """
    # --- Project Paths and Naming ---
    # The absolute path to your dataset folder
    dataset_dir: str = r"A:\New folder\face_detection\dataset"
    
    # Base directory for all models and checkpoints
    checkpoint_base_dir: str = r"A:\model_checkpoints"
    
    # Main project name, will be a subfolder in checkpoint_base_dir
    project_name: str = "face-detector"
    
    # --- Model hyperparameters ---
    model_type: str = 'yolov8m.pt'  # Using the nano model as requested
    image_size: int = 640
    batch_size: int = 4
    num_epochs: int = 300

    # --- optimization and learning
    patience: int = 20

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def project_dir(self) -> str:
        """Returns the full path of the project checkpoint directory."""
        return os.path.join(self.checkpoint_base_dir, self.project_name)

    @property
    def data_yaml_path(self) -> str:
        """Returns the full path to the data.yaml file."""
        return os.path.join(self.dataset_dir, 'data.yaml')
