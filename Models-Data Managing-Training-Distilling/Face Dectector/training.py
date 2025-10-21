# train_model.py
import os
from ultralytics import YOLO
import logging

from config import TrainingConfig

# Set up logging for professional output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _load_model(checkpoint_dir: str) -> YOLO:
    """
    Load YOLO model from checkpoint directory.
    Priority: last.pt > best.pt > default yolov8m.pt
    """
    last_ckpt = os.path.join(checkpoint_dir, "last.pt")
    best_ckpt = os.path.join(checkpoint_dir, "best.pt")

    if os.path.exists(last_ckpt):
        logger.info(f"Resuming training from last checkpoint: {last_ckpt}")
        return YOLO(last_ckpt), True  # resume=True
    elif os.path.exists(best_ckpt):
        logger.info(f"Starting from best model weights: {best_ckpt}")
        return YOLO(best_ckpt), False  # resume=False
    else:
        logger.warning("No checkpoints found. Loading default yolov8m.pt")
        return YOLO("yolov8m.pt"), False


def main():
    try:
        # Load configuration
        config = TrainingConfig()

        # Step 1: Check for dataset.yaml
        if not os.path.exists(config.data_yaml_path):
            logger.error(
                f"Error: dataset.yaml not found at {config.data_yaml_path}. Please create it."
            )
            return

        # Step 2: Initialize YOLO model
        checkpoint_dir = r"A:\model_checkpoints\face-detector\face-detector\weights"
        model, resume_flag = _load_model(checkpoint_dir)

        # Step 3: Start training
        logger.info("Starting model training...")
        os.makedirs(config.project_dir, exist_ok=True)

        results = model.train(
            data=config.data_yaml_path,
            epochs=config.num_epochs,
            imgsz=config.image_size,
            batch=config.batch_size,
            project=config.project_dir,
            name=config.project_name,
            patience=config.patience,
            device=config.device,
            resume=resume_flag,
            amp = True,
            cache = False,
            augment=True,        # keep augmentations
            mosaic=0.7,          # slightly reduce mosaic (0.7 = less aggressive)
            mixup=0.05,           # light mixup, helps precision
            )

        logger.info("Training completed successfully!")

        # Save checkpoints info
        last_pt = os.path.join(config.project_dir, config.project_name, "weights", "last.pt")
        best_pt = os.path.join(config.project_dir, config.project_name, "weights", "best.pt")

        logger.info(f"Model checkpoints saved to: {config.project_dir}")
        logger.info(f"Last model path: {last_pt}")
        logger.info(f"Best model path: {best_pt}")

    except Exception as e:
        logger.exception(f"An error occurred during training: {e}")


if __name__ == "__main__":
    main()
