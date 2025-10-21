import torch
import torch.nn.functional as F
from ultralytics import YOLO
import os

# ==============================================================================
# == 1. CONFIGURATION - Bro, change these settings to match your setup! ==
# ==============================================================================

# --- REQUIRED PATHS ---
# Path to your dataset's .yaml file. This is super important for training.
DATA_YAML_PATH = r'A:\New folder\face_detection\dataset\data.yaml'

# Path to your pre-trained YOLOv8-L teacher model weights.
TEACHER_MODEL_PATH = r'A:\model_checkpoints\face-detector\face-detector\weights\best.pt'

# Directory where your new distilled models will be saved.
PROJECT_SAVE_DIR = r'A:\model_checkpoints\distill-face-detector'

# Name for the training run. A new folder with this name will be created inside PROJECT_SAVE_DIR.
# If you run the script again with the same name, it will resume training.
RUN_NAME = 'distilled_yolov8n_from_M'


# --- TRAINING HYPERPARAMETERS ---
EPOCHS = 200
BATCH_SIZE = 4
IMG_SIZE = 640


# --- DISTILLATION HYPERPARAMETERS ---
# These control the strength of the teacher's guidance.
# Higher temperature softens the teacher's predictions, making it easier for the student to learn.
DISTILL_TEMP = 20.0

# Weight for the classification part of the distillation loss.
DISTILL_WEIGHT_CLS = 1.0

# Weight for the bounding box part of the distillation loss.
DISTILL_WEIGHT_BOX = 0.05

# --- Global variable for the teacher model ---
# We define it here so the callback function can access it.
teacher_model = None

# ==============================================================================
# == 2. SCRIPT SETUP - No need to change anything below this line. ==
# ==============================================================================

# --- Sanity Checks ---
if not os.path.exists(TEACHER_MODEL_PATH):
    raise FileNotFoundError(f"Teacher model not found at: {TEACHER_MODEL_PATH}")
if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"Dataset YAML not found at: {DATA_YAML_PATH}. Please set this path.")
if not os.path.exists(PROJECT_SAVE_DIR):
    print(f"Creating project directory: {PROJECT_SAVE_DIR}")
    os.makedirs(PROJECT_SAVE_DIR)


# ==============================================================================
# == 3. DISTILLATION LOGIC - This is where the magic happens. ==
# ==============================================================================

def distillation_callback(trainer):
    """
    This function is the core of the distillation process.
    It's called by the Ultralytics trainer right before the backward pass
    to calculate and add the distillation loss.
    """
    # Ensure the teacher model is loaded before proceeding
    if teacher_model is None:
        return

    # Only run this logic during the training phase.
    if trainer.epoch < trainer.start_epoch:
        return

    # Get student predictions (already computed by the trainer)
    student_preds = trainer.preds[0]

    # Run the teacher model on the same batch of images to get its predictions.
    # We do this in a 'no_grad' block because we don't need to track gradients for the teacher.
    with torch.no_grad():
        teacher_preds = teacher_model.model(trainer.batch['img'])[0]

    # Get model properties from the trainer.
    nc = trainer.model.nc  # number of classes
    reg_max = trainer.model.model[-1].reg_max

    # Split the prediction tensors into box and class parts.
    # The shape of the tensor is (batch, channels, anchors), where channels = (4 * reg_max) + nc
    student_box, student_cls = student_preds.split((4 * reg_max, nc), 1)
    teacher_box, teacher_cls = teacher_preds.split((4 * reg_max, nc), 1)

    # --- Calculate Distillation Loss ---

    # 1. Classification Loss (using Kullback-Leibler Divergence)
    # This loss encourages the student model's class predictions to match the
    # "soft labels" (probability distribution) provided by the teacher.
    loss_cls = F.kl_div(
        F.log_softmax(student_cls / DISTILL_TEMP, dim=1),
        F.log_softmax(teacher_cls / DISTILL_TEMP, dim=1),
        reduction='batchmean',
        log_target=True
    ) * (DISTILL_TEMP ** 2)

    # 2. Bounding Box Regression Loss (using Mean Squared Error)
    # This loss encourages the student's bounding box predictions to match the teacher's.
    loss_box = F.mse_loss(student_box, teacher_box)

    # Combine the distillation losses with their weights.
    distillation_loss = (DISTILL_WEIGHT_CLS * loss_cls) + (DISTILL_WEIGHT_BOX * loss_box)

    # Add our custom distillation loss to the main loss computed by the trainer.
    trainer.loss += distillation_loss

    # Optional: Log the distillation loss to the progress bar for real-time monitoring.
    if trainer.progress_bar:
        trainer.progress_bar.set_postfix(distill_loss=f"{distillation_loss.item():.4f}")


# ==============================================================================
# == 4. TRAINING EXECUTION - Kicking off the training. ==
# ==============================================================================

if __name__ == '__main__':
    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load Teacher Model ---
    # This is now inside the main block to prevent it from running in worker processes.
    print("Loading teacher model...")
    teacher_model = YOLO(TEACHER_MODEL_PATH)
    teacher_model.to(device)
    teacher_model.eval()  # Set teacher to evaluation mode
    print("Teacher model loaded successfully.")

    # --- Check for Resuming ---
    # This is the key for resumable training. We check if a 'last.pt' checkpoint
    # already exists from a previous run with the same RUN_NAME.
    resume_path = os.path.join(PROJECT_SAVE_DIR, RUN_NAME, 'weights', 'last.pt')

    if os.path.exists(resume_path):
        print(f"✅ Resuming training from existing checkpoint: {resume_path}")
        student_model = YOLO(resume_path)
    else:
        print(f"🚀 Starting a new training run from scratch (yolov8n).")
        # Initialize a new student model from the standard yolov8n architecture.
        student_model = YOLO('yolov8n.yaml')

    # Attach our custom distillation function to the trainer's lifecycle.
    # It will be called automatically on the 'on_before_backward' event.
    student_model.add_callback("on_before_backward", distillation_callback)

    print("Starting model training...")
    # --- Start Training ---
    student_model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        project=PROJECT_SAVE_DIR,
        name=RUN_NAME,
        amp=True,
        cache=False,
        augment=True,        # keep augmentations
        mosaic=0.7,          # slightly reduce mosaic (0.7 = less aggressive)
        mixup=0.05,
        resume = True,
        patience = 20          # light mixup, helps precision

    )
    print("🎉 Training finished!")

