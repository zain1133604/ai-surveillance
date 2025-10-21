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
    data_dir : str = r"A:\\images\\dataset"
    # Base directory for all models and checkpoints
    checkpoint_base_dir : str = r"A:\\model_checkpoints"
    # Main project name, will be a subfolder in checkpoint_base_dir
    project_name: str = "gun_detector"
    # Fixed run ID for all checpoints. this ensures consistent saving/loading.
    run_id: str = "gun_dectector_run35222"


    # --- Model hyperparameters --- 
    model_type : str = 'yolov8l.pt'

    image_size: int = 640
    batch_size: int = 4
    num_epochs: int = 80

    # --- optimization and learning
    freeze_backbone_epochs: int = 0
    patience: int = 10

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


    @property
    def project_dir(self) -> str:
        """Returns the full path to the project's checkpoint directory."""
        return os.path.join(self.checkpoint_base_dir, self.project_name)

    @property
    def run_dir(self) -> str:
        """Returns the full path to the current run's directory where weights are saved."""
        return os.path.join(self.project_dir, self.run_id)

    @property
    def last_checkpoint_path(self) -> str:
        """Returns the path to the last saved model checkpoint."""
        return os.path.join(self.run_dir, 'weights', 'last.pt')

    @property
    def best_checkpoint_path(self) -> str:
        """Returns the path to the best saved model checkpoint."""
        return os.path.join(self.run_dir, 'weights', 'best.pt')
