# Gradio app: Real-Time Restricted Area Monitoring + Missing Person
# Single-file app combining YOLOv8 (Human, Weapon, Face detection) and FaceNet (Missing Person).

# To run locally, you MUST have the required models (e.g., distiled_face_Detector.pt, distill_human_detector.pt, etc.) 
# and the necessary Python libraries installed.

# Requirements for a real run:
# - pip install gradio opencv-python-headless numpy pillow tqdm torch ultralytics
# - pip install keras-facenet face-alignment

import os
import tempfile
import uuid
import logging
from typing import Optional, Tuple, List, Dict

# Core dependencies
import cv2
import numpy as np
import torch, warnings
from PIL import Image
import gradio as gr
from tqdm import tqdm

# YOLO and FaceNet dependencies
from ultralytics import YOLO
import face_alignment


# --- Face Alignment Setup ---
# Initialize Face Alignment for robust face recognition/alignment
try:
    # Use 'cpu' device for face_alignment if not specified, or use the global DEVICE
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        device='cpu'
    )
    FACE_ALIGNMENT_ENABLED = True
    print("Face Alignment model loaded successfully.")
except Exception as e:
    fa = None
    FACE_ALIGNMENT_ENABLED = False
    print(f"WARNING: Face Alignment failed to load: {e}. Face recognition may be less robust.")

# NOTE ON FACE RECOGNITION:
# The keras-facenet logic is maintained. 
try:
    # Ensure 'keras_facenet' and its dependencies are correctly installed.
    from keras_facenet import FaceNet
    # FaceNet needs a small pre-run for initialization
    embedder = FaceNet() 
    # Small dummy run to ensure first call latency is removed (optional)
    _ = embedder.embeddings([np.zeros((160, 160, 3), dtype=np.uint8)]) 
    FACE_RECOGNITION_ENABLED = True
    print("Keras FaceNet model loaded successfully.")
except ImportError:
    FACE_RECOGNITION_ENABLED = False
    embedder = None
    print("WARNING: keras-facenet not found. Missing Person feature will be disabled.")

# --- Configuration for Models and Paths ---
# NOTE: These model files MUST exist in the same directory for this script to run.
FACE_DETECTOR_MODEL_PATH = "distiled_face_Detector.pt" # YOLOv8 Face Detection
HUMAN_DETECTOR_MODEL_PATH = "distill_human_detector.pt" # YOLOv8 Human Detection & Tracking
GUN_KNIFE_DETECTOR_MODEL_PATH = "distilled_gun_detector.pt" # YOLOv8 Weapon Detection

# Use GPU if available, otherwise CPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# General Constants
YOLO_IMAGE_SIZE = 720

# Weapon Class Mappings (As per your request)
WEAPON_CLASSES = {0: "Gun", 1: "Knife"}

# --- Label Drawing Constants ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.0
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255) # White text (BGR format)
ALERT_COLOR = (0, 0, 255) # Red for Alert/Restricted/Weapon
NORMAL_COLOR = (0, 255, 0) # Green for Normal
BOUNDING_BOX_THICKNESS = 3
FACE_MATCH_COLOR = (255, 255, 0) # Yellow for Found Person (BGR format)
FACE_RECOGNITION_THRESHOLD = 0.950 # Distance threshold for FaceNet

# Global model variables
human_detector = None
weapon_detector = None
face_detector = None

# ----------------------
# Model Setup Function
# ----------------------

def setup_yolo_model(model_path: str, name: str, device: str):
    """
    Loads YOLOv8 model in pure inference mode, preventing 'data.yaml not found' or training logic.
    Works across all Ultralytics 8.x versions.
    """
    print(f"Loading {name} model from {model_path} on {device}...")
    try:
        if not os.path.exists(model_path):
            print(f"WARNING: {name} model not found at {model_path}. Using mock model.")
            return type('MockYOLO', (object,), {
                'to': lambda s, d: None,
                'eval': lambda s: None,
                '__call__': lambda s, *a, **k: [type('MockResult', (object,), {'boxes': None})()],
                'track': lambda s, *a, **k: [type('MockResult', (object,), {'boxes': None})()]
            })()

        # 🧩 Step 1: Temporarily silence any trainer/config warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        # 🧩 Step 2: Use official API but patch the environment so training logic never triggers
        os.environ["YOLO_IGNORE_CONFIG"] = "1"
        os.environ["YOLO_TASK"] = "detect"

        # 🧩 Step 3: Load safely
        model = YOLO(model_path)
        model.overrides = {"mode": "predict", "task": "detect"}
        model.predictor = None  # remove trainer ref if present

        # 🧩 Step 4: Finalize
        model.to(device)
        model.model.eval()
        print(f"{name} model loaded successfully in inference mode.")
        return model

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load {name} model: {e}")
        raise

# --- Model Loading (Runs synchronously on script startup) ---
try:
    human_detector = setup_yolo_model(HUMAN_DETECTOR_MODEL_PATH, "Human Detector", DEVICE)
    weapon_detector = setup_yolo_model(GUN_KNIFE_DETECTOR_MODEL_PATH, "Gun/Knife Detector", DEVICE)
    face_detector = setup_yolo_model(FACE_DETECTOR_MODEL_PATH, "Face Detector", DEVICE)
except Exception:
    # Exit gracefully if critical models fail to load
    print("Exiting due to critical model loading failure.")
    # In a deployed environment, this is where a deployment would fail, 
    # but the script will continue with the Mock detectors if they were returned.
    pass 

# ----------------------
# Utility Functions
# ----------------------

def draw_alert_label(frame: np.ndarray, bbox_xyxy: np.ndarray, label: str, color: Tuple[int, int, int]):
    """Draws a bounding box and a label with a solid background."""
    x1, y1, x2, y2 = bbox_xyxy
    
    # Draw Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOUNDING_BOX_THICKNESS)

    # Calculate text size and position
    (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    
    # Position the label slightly above the bounding box
    text_x = x1
    text_y = y1 - 10
    
    if text_y < text_height + 10:
        text_y = y2 + text_height + 10 

    # Draw background rectangle for text
    cv2.rectangle(frame, 
                  (text_x, text_y - text_height - 5), 
                  (text_x + text_width + 5, text_y + 5), 
                  color, 
                  cv2.FILLED)
    
    # Draw the text
    cv2.putText(frame, label, 
                (text_x + 2, text_y), 
                FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

def get_face_encodings_from_bytes(image_path: str) -> Optional[np.ndarray]:
    """
    Loads image from path and computes face embeddings using keras-facenet.
    Returns the first found face embedding.
    """
    if not FACE_RECOGNITION_ENABLED:
        print("Face recognition attempted but library is disabled.")
        return None

    try:
        # Load image with PIL first as cv2.imread is less robust in some environments
        img_pil = Image.open(image_path).convert("RGB")
        rgb_img = np.array(img_pil) # Already RGB from PIL convert

        if rgb_img is None:
            print("Could not decode missing person image.")
            return None

        # FaceNet expects 160x160 aligned face for optimal performance.
        # We'll use face_detector and fa to pre-process the reference image.
        
        # 1. Detect face using YOLO
        yolo_result = face_detector(rgb_img, imgsz=YOLO_IMAGE_SIZE, verbose=False)
        face_boxes = yolo_result[0].boxes
        
        if face_boxes is None or len(face_boxes) == 0:
            print("No face detected in missing person image.")
            return None
            
        # Use the first detected face
        x1, y1, x2, y2 = face_boxes[0].xyxy[0].cpu().numpy().astype(int)
        face_crop = rgb_img[y1:y2, x1:x2]
        
        # 2. Align face using landmarks
        if FACE_ALIGNMENT_ENABLED and fa is not None:
            preds = fa.get_landmarks(face_crop)
            if preds:
                landmarks = preds[0]
                left_eye = np.mean(landmarks[36:42], axis=0)
                right_eye = np.mean(landmarks[42:48], axis=0)
                dx, dy = right_eye - left_eye
                angle = np.degrees(np.arctan2(dy, dx))
                eyes_center = tuple(np.mean([left_eye, right_eye], axis=0))
                rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
                # Face alignment needs the image to be BGR if using cv2 for alignment, 
                # but we use RGB for consistency, though cv2 functions are fine with it.
                aligned_face = cv2.warpAffine(face_crop, rot_mat, (face_crop.shape[1], face_crop.shape[0]))
            else:
                aligned_face = face_crop
        else:
            aligned_face = face_crop

        # 3. Resize to FaceNet's required input size (160x160)
        resized_face = cv2.resize(aligned_face, (160, 160), interpolation=cv2.INTER_AREA)

        # 4. Get embeddings
        embeddings = embedder.embeddings([resized_face])
        
        if len(embeddings) > 0:
            print("Missing person face embedding calculated.")
            return embeddings[0]
        else:
            print("No face detected or embedding failed after crop/align.")
            return None

    except Exception as e:
        print(f"Error calculating missing person embedding: {e}")
        return None

def extract_first_frame(video_path: str) -> Image.Image:
    """Extracts the first frame of a video for display."""
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise RuntimeError('Unable to read first frame from video')
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def get_bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Calculates the bounding box (x1, y1, x2, y2) of the non-zero area of the mask."""
    # Find all non-zero coordinates
    # Note: mask is a binary 0/1 array, convert to 8-bit for cv2.findNonZero
    mask_8bit = (mask * 255).astype(np.uint8)
    coords = cv2.findNonZero(mask_8bit)
    if coords is None:
        return None
    # Calculate bounding box (x, y, w, h)
    x, y, w, h = cv2.boundingRect(coords)
    # Return x1, y1, x2, y2
    return (x, y, x + w, y + h) 



def is_inside_mask(mask: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
    """
    Checks if the center of the bounding box lies within the painted restricted mask.
    """
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    if cx < 0 or cy < 0 or cx >= mask.shape[1] or cy >= mask.shape[0]:
        return False
    return mask[cy, cx] > 0  # >0 means painted area



def mask_from_sketch(original_frame, sketch_img):
    """
    Robust version – works with both old/new Gradio formats.
    Compares 'background' vs 'composite' to get drawn pixels.
    """
    import numpy as np
    from PIL import Image

    if sketch_img is None:
        raise ValueError("No sketch image provided.")

    # Case 1: New Gradio Sketch format (has 'background', 'composite')
    if isinstance(sketch_img, dict) and "composite" in sketch_img and "background" in sketch_img:
        # The Gradio ImageEditor dictionary contains PIL Images or NumPy arrays.
        # We explicitly convert to NumPy for arithmetic operations.
        bg = np.array(sketch_img["background"]).astype(int)
        comp = np.array(sketch_img["composite"]).astype(int)

        # Find where composite differs from background → drawn region
        diff = np.abs(comp - bg)
        mask = (diff.sum(axis=-1) > 30).astype(np.uint8) * 255  # threshold = 30
        return mask

    # Case 2: Old format (has 'image') - kept for robustness
    elif isinstance(sketch_img, dict) and "image" in sketch_img:
        arr = np.array(sketch_img["image"])
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Look for red strokes (common for restricted areas)
        red_mask = (r > 150) & (g < 100) & (b < 100)
        mask = red_mask.astype(np.uint8) * 255
        return mask
        
    # Case 3: Simple NumPy array (less common for ImageEditor output, but safe to handle)
    elif isinstance(sketch_img, np.ndarray):
        arr = sketch_img
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        red_mask = (r > 150) & (g < 100) & (b < 100)
        mask = red_mask.astype(np.uint8) * 255
        return mask

    else:
        # Handle unexpected output, provide context
        keys = list(sketch_img.keys()) if isinstance(sketch_img, dict) else str(type(sketch_img))
        raise ValueError(f"Unexpected sketch format keys: {keys}")

# ----------------------
# Processing pipeline
# ----------------------

def process_video(
    video_path: str, 
    first_frame_pil: Image.Image, # Currently unused in function body, but kept for context
    mode: str = 'normal', 
    sketch_image_pil: Optional[Dict]=None, # Should be the ImageEditor output dict
    ref_face_path: Optional[str]=None
) -> Tuple[str, str]:
    """
    The main video processing function.
    Returns (output_video_path, textual_log)
    Modes: 'normal', 'restricted', 'missing_person'
    """
    # Check for mock detectors to provide meaningful error
    if isinstance(human_detector, type) or isinstance(weapon_detector, type) or isinstance(face_detector, type):
        return None, "Error: One or more models are unavailable. Check logs for CRITICAL ERROR messages."
    
    if human_detector is None or weapon_detector is None or face_detector is None:
        return None, "Error: One or more models are unavailable."
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # --- Setup Output File ---
    out_file = os.path.join(tempfile.gettempdir(), f'processed_{uuid.uuid4().hex}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    
    # Fallback to XVID if mp4v fails (common in Linux/Hugging Face Spaces)
    if not writer.isOpened():
          fourcc_fallback = cv2.VideoWriter_fourcc(*'XVID')
          writer = cv2.VideoWriter(out_file, fourcc_fallback, fps, (width, height))
          if not writer.isOpened():
              cap.release()
              return None, "CRITICAL: Could not create video writer using mp4v or XVID codec."

    # --- 1. Handle Restricted Area (ROI) setup ---
    roi_mask = None
    
    if mode == 'restricted' and sketch_image_pil:
        # We don't need the original frame, just a placeholder for dimension, but 
        # mask_from_sketch now just needs the sketch_image_pil dict
        try:
            mask = mask_from_sketch(None, sketch_image_pil) 
        except ValueError as e:
            return None, f"Error creating restricted mask: {e}"

        if mask is None or np.sum(mask) == 0:
            return None, "Error: Restricted area mode selected, but no drawing detected."

        # Convert to binary (0/255) mask and ensure correct video frame shape
        # Interpolation is NEAREST since it's a binary mask
        mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if mask_resized.ndim == 3:
            mask_resized = cv2.cvtColor(mask_resized, cv2.COLOR_RGB2GRAY)
        roi_mask = (mask_resized > 0).astype(np.uint8) * 255  # Ensure binary mask
    else:
        roi_mask = None
    
    # --- 2. Handle Missing Person setup ---
    target_encoding = None
    missing_person_found = False # Global flag for video duration 
    if mode == 'missing_person' and ref_face_path:
        
        if not FACE_RECOGNITION_ENABLED:
            return None, "Error: Face Recognition library is disabled."
            
        target_encoding = get_face_encodings_from_bytes(ref_face_path)
        if target_encoding is None:
            return None, "Error: Missing person image did not contain a detectable face or encoding failed."


    alerts = []
    frame_idx = 0


    missing_person_found = False
    missing_person_bbox = None  # store last known position
    frames_since_last_seen = 0
    max_invisible_frames = 20

    
    # --- Frame Processing Loop ---
    # Use tqdm for progress bar in Gradio
    for _ in tqdm(range(total_frames), desc="Processing Video"):
        success, frame = cap.read()
        if not success:
            break
        
        # --- 3. Draw Restricted Area (ROI) Boundary ---
        if mode == 'restricted' and roi_mask is not None and np.sum(roi_mask) > 0:
            # Use alpha blending for a semi-transparent overlay
            colored_mask_overlay = np.zeros_like(frame)
            colored_mask_overlay[roi_mask > 0] = ALERT_COLOR
            alpha = 0.15
            cv2.addWeighted(colored_mask_overlay, alpha, frame, 1 - alpha, 0, frame)

        # --- 4. Human Detection and Tracking ---
        # Note: We enforce classes=0 for 'person' (common in YOLO COCO)
        human_results = human_detector(
            source=frame,

            stream=False,
            imgsz=YOLO_IMAGE_SIZE,
            tracker='bytetrack.yaml',  # you can use 'botsort.yaml' too
            device=DEVICE,
            conf=0.10,  # better confidence
            iou=0.65,
            verbose=False,
            show=False
        )

        # --- 5. Weapon Detection ---
        weapon_results = weapon_detector.track(
            frame,
            imgsz=YOLO_IMAGE_SIZE,
            tracker='bytetrack.yaml',
            device=DEVICE,
            verbose=False,
            conf = 0.46,
            # Force inference mode for robustness (though .__call__ implies it)
        )
            
        detections_to_draw = [] 
        
        # A. Process Human Detections
        human_boxes = human_results[0].boxes
        if human_boxes is not None and human_boxes.id is not None:
            for i in range(len(human_boxes)):
                box_data = human_boxes[i]
                # Ensure the data exists before accessing the first element
                if box_data.xyxy.numel() == 0 or box_data.id.numel() == 0: continue

                bbox_xyxy = box_data.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box_data.id[0].item())
                
                x1, y1, x2, y2 = bbox_xyxy
                if x2 <= x1 or y2 <= y1: continue 

                # Check if human center is inside the Restricted Area
                is_restricted_alert = (
                mode == 'restricted' and 
                roi_mask is not None and 
                is_inside_mask(roi_mask, (x1, y1, x2, y2))
                )

                label = f"ID: {track_id}"
                color = NORMAL_COLOR 
                alert_status = ""

                if is_restricted_alert:
                    color = ALERT_COLOR
                    alert_status = "ALERT (Restricted)"
                    label = f"{alert_status} | ID:{track_id}"
                    alerts.append(f"F{frame_idx}: Restricted Area Breach by ID:{track_id}")
                
                detections_to_draw.append({
                    'bbox': bbox_xyxy, 
                    'label': label, 
                    'color': color,
                    'is_human': True,
                    'track_id': track_id
                })
        
        # B. Process Weapon Detections
        weapon_boxes = weapon_results[0].boxes
        if weapon_boxes is not None:
            for i in range(len(weapon_boxes)):
                box_data = weapon_boxes[i]
                if box_data.xyxy.numel() == 0 or box_data.cls.numel() == 0: continue

                bbox_xyxy = box_data.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box_data.cls[0].item())
                weapon_name = WEAPON_CLASSES.get(class_id, "Weapon")
                
                # Weapons are always ALERT (Red)
                label = f"ALERT | {weapon_name}"
                alerts.append(f"F{frame_idx}: Weapon Detected ({weapon_name})")
                
                detections_to_draw.append({
                    'bbox': bbox_xyxy, 
                    'label': label, 
                    'color': ALERT_COLOR,
                    'is_human': False 
                })

        # C. Process Missing Person Detection/Recognition
        if mode == 'missing_person' and target_encoding is not None:
            face_results = face_detector.track(frame, imgsz=YOLO_IMAGE_SIZE, device=DEVICE, verbose=False, mode='predict')
            face_boxes = face_results[0].boxes

            if face_boxes is not None:
                best_match_distance = float('inf')
                best_bbox = None

                for i in range(len(face_boxes)):
                    box_data = face_boxes[i]
                    if box_data.xyxy.numel() == 0: continue
                    
                    x1, y1, x2, y2 = box_data.xyxy[0].cpu().numpy().astype(int)
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue

                    # Convert to RGB for FaceNet/Face Alignment
                    rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

                    # --- Face alignment using landmarks ---
                    aligned_face = rgb_face
                    if FACE_ALIGNMENT_ENABLED and fa is not None:
                        preds = fa.get_landmarks(rgb_face)
                        if preds:
                            landmarks = preds[0]
                            # Only align if landmarks are found
                            left_eye = np.mean(landmarks[36:42], axis=0)
                            right_eye = np.mean(landmarks[42:48], axis=0)
                            dx, dy = right_eye - left_eye
                            angle = np.degrees(np.arctan2(dy, dx))
                            eyes_center = tuple(np.mean([left_eye, right_eye], axis=0))
                            rot_mat = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
                            aligned_face = cv2.warpAffine(rgb_face, rot_mat, (rgb_face.shape[1], rgb_face.shape[0]))
                    
                    # Resize to FaceNet's required input size (160x160)
                    resized_face = cv2.resize(aligned_face, (160, 160), interpolation=cv2.INTER_AREA)

                    # --- Now get embeddings from aligned face ---
                    embeddings = embedder.embeddings([resized_face])
                    if len(embeddings) == 0:
                        continue

                    distance = np.linalg.norm(target_encoding - embeddings[0])
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_bbox = (x1, y1, x2, y2)

                # --- Decide if matched ---
                if best_bbox is not None and best_match_distance < FACE_RECOGNITION_THRESHOLD:
                    # Only register the "found" event once per continuous period
                    if not missing_person_found:
                        alerts.append(f"F{frame_idx}: !!! MISSING PERSON FOUND !!! (Dist: {best_match_distance:.3f})")
                    
                    missing_person_found = True
                    missing_person_bbox = best_bbox
                    frames_since_last_seen = 0
                else:
                    # Not matched this frame → increase invisible frame count
                    if missing_person_found:
                        frames_since_last_seen += 1
                        if frames_since_last_seen > max_invisible_frames:
                            missing_person_found = False
                            missing_person_bbox = None

            # --- If recently found, keep showing box for N frames ---
            if missing_person_found and missing_person_bbox is not None:
                # Add the matched person to detections_to_draw to ensure it gets drawn
                # The label drawing logic later will ensure the generic human box is skipped
                detections_to_draw.append({
                    'bbox': missing_person_bbox, 
                    'label': "MATCH FOUND!", 
                    'color': FACE_MATCH_COLOR,
                    'is_human': True, # It is a human, but specially identified
                    'track_id': -1 # Use -1 to mark as special match
                })


        # --- 7. Final Drawing of other detections ---
        for det in detections_to_draw:
            is_matched_person_box = (
                det.get('track_id') == -1 or 
                (mode == 'missing_person' and missing_person_found and missing_person_bbox is not None and np.array_equal(det['bbox'], missing_person_bbox))
            )

            # Skip drawing the generic human box if it overlaps with the specific matched person box
            if det['is_human'] and mode == 'missing_person' and missing_person_found and not is_matched_person_box:
                 # Check if this human box is the same as the missing person's box (simple check: center overlap)
                 x1_det, y1_det, x2_det, y2_det = det['bbox']
                 cx_det, cy_det = int((x1_det + x2_det) / 2), int((y1_det + y2_det) / 2)

                 x1_match, y1_match, x2_match, y2_match = missing_person_bbox if missing_person_bbox is not None else (-1,-1,-1,-1)
                 if x1_match <= cx_det <= x2_match and y1_match <= cy_det <= y2_match:
                     continue # Skip drawing generic human box if match box covers it

            draw_alert_label(frame, det['bbox'], det['label'], det['color'])
            
        # --- Draw Global Missing Person Alert ---
        if missing_person_found:
            global_label = "!!! MISSING PERSON FOUND !!!"
            (text_width, text_height), baseline = cv2.getTextSize(global_label, FONT, FONT_SCALE, FONT_THICKNESS)
            text_x = width - text_width - 20
            text_y = 50 
            
            # Draw background rectangle 
            cv2.rectangle(frame, 
                          (text_x - 10, text_y - text_height - 10), 
                          (width - 10, text_y + 10), 
                          FACE_MATCH_COLOR, # Yellow
                          cv2.FILLED)
            
            cv2.putText(frame, global_label, 
                        (text_x, text_y), 
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    log_text = '\n'.join(alerts) if alerts else 'No alerts.'
    return out_file, log_text

# ----------------------
# Gradio UI
# ----------------------

with gr.Blocks() as demo:
    gr.Markdown('# Intelligent Video Surveillance (Restricted Area & Missing Person Search)')
    gr.Markdown('Upload a video, then use the first frame to define a Restricted Area (draw a rectangle) or upload a Missing Person reference face. The app detects humans, weapons, and tracks IDs.')
    with gr.Row():
        video_input = gr.Video(label='1. Upload Video')
        # The user draws on this image to define the restricted area
        first_frame_image = gr.ImageEditor(
            label='2. First frame (Draw on this image to mark Restricted Area)')
    with gr.Row():
        mode = gr.Radio(['normal', 'restricted', 'missing_person'], 
                        label='3. Select Mode', 
                        value='normal')
        ref_face = gr.Image(label='4. Reference Face Image (for missing_person mode)', 
                            type='filepath', 
                            interactive=True)
    with gr.Row():
        process_btn = gr.Button('5. Start Processing', variant='primary')
        status_out = gr.Textbox(label='Alert Log', lines=5)
    
    processed_video = gr.Video(label='6. Processed Video Output')

    # when a video is uploaded: extract first frame and display
    def on_video_upload(video_path):
        if video_path is None:
            return None
        return extract_first_frame(video_path)

    video_input.change(on_video_upload, 
                        inputs=video_input, 
                        outputs=first_frame_image,
                        queue=False) # Queue=False for faster UI feedback

    # Main processing function call
    def on_process(video_path, first_frame_image_dict, mode_val, ref_face_path):
        if video_path is None:
            return None, 'No video uploaded.'
        
        # Note: first_frame_image_dict is the ImageEditor output, which contains the sketch data
        
        if mode_val == 'missing_person' and ref_face_path is None:
             return None, 'Missing reference face image for missing_person mode.'
        
        if mode_val == 'restricted' and first_frame_image_dict is None:
             return None, 'Restricted Area mode selected, but no frame to draw on or drawing output is missing.'

        # Pass the drawn image (sketch_image_pil) and original first frame to the processing pipeline
        out_path, log = process_video(
            video_path, 
            None, # We don't need the PIL image of the first frame here
            mode=mode_val, 
            sketch_image_pil=first_frame_image_dict, # This contains the sketch/drawing
            ref_face_path=ref_face_path
        )
        return out_path, log

    process_btn.click(on_process, 
                      inputs=[video_input, first_frame_image, mode, ref_face], 
                      outputs=[processed_video, status_out])

if __name__ == '__main__':
    demo.launch()
