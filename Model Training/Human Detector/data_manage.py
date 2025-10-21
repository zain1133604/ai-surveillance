# 👨‍🔧👨‍🔧👨‍🔧 sub folder counting
import os

# # Define the path to your main folder
folder_path = r'A:\\New folder\\voilence not voilence\\dataset\\not voilence'

# Define image file extensions you want to count
image_extensions = (    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',   # Images
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
    '.mpeg', '.mpg', '.3gp', '.m4v' )

count = 0
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.lower().endswith(image_extensions):
            count += 1

print(f'Total images found: {count}')




# import os
# import sys
# from tqdm import tqdm # For a nice progress bar

# def rename_images_in_folder():
#     """
#     Renames all image files in a specified folder with a new prefix
#     and sequential numbering, preserving their original file extensions.
#     """
#     print("Welcome to the Image Renamer!")
#     print("This script will rename all image files in a chosen folder.")

#     # --- USER INPUT SECTION: MODIFY THESE VARIABLES DIRECTLY IN THE CODE ---
#     # 1. Enter the FULL path to the folder containing your images.
#     #    Example for Windows: r"C:\Users\YourName\Documents\MyImages"
#     #    Example for macOS/Linux: "/home/youruser/pictures/humans"
#     FOLDER_PATH = r"A:\\images\\archive (8)\\Anomalous Action Detection Dataset( Ano-AAD)\\abnormal class\\Arrest" # <--- SET YOUR FOLDER PATH HERE

#     # 2. Enter the desired prefix for the new filenames (e.g., 'human_pic').
#     #    If left empty (e.g., NEW_PREFIX = ""), files will be named '1.jpg', '2.png', etc.
#     NEW_PREFIX = "llllo_monks" # <--- SET YOUR DESIRED PREFIX HERE

#     # 3. Confirm if you want to proceed with renaming. Set to True to proceed, False to cancel.
#     CONFIRM_RENAME = True # <--- SET TO True TO PROCEED WITH RENAMING
#     # -----------------------------------------------------------------------

#     folder_path = FOLDER_PATH
#     prefix = NEW_PREFIX
#     confirm = "yes" if CONFIRM_RENAME else "no"

#     # Validate folder path (no longer asks in loop, just checks once)
#     if not os.path.isdir(folder_path):
#         print(f"\n❌ Error: The provided FOLDER_PATH '{folder_path}' is not a valid directory.")
#         print("Please edit the 'FOLDER_PATH' variable in the code to a correct path.")
#         return

#     if not prefix:
#         print("\n❗ Warning: No prefix entered in 'NEW_PREFIX'. Files will be named '1.jpg', '2.png', etc.")
#         if not CONFIRM_RENAME: # If user didn't explicitly confirm, treat as cancellation
#             print("🚫 Renaming cancelled because 'CONFIRM_RENAME' is False and no prefix was set.")
#             return

#     # Define common image extensions (case-insensitive)
#     image_extensions = (   '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',   # Images
#     '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm',
#     '.mpeg', '.mpg', '.3gp', '.m4v')

#     # --- 3. Get list of image files ---
#     image_files = []
#     for filename in os.listdir(folder_path):
#         # Check if the file has an image extension
#         if os.path.isfile(os.path.join(folder_path, filename)) and filename.lower().endswith(image_extensions):
#             image_files.append(filename)

#     if not image_files:
#         print(f"\n😔 No image files found in '{folder_path}' with common extensions ({', '.join(image_extensions)}).")
#         print("Please ensure your images are in the folder and have standard extensions.")
#         return

#     # Sort files to ensure consistent numbering (e.g., alphabetical order)
#     image_files.sort()

#     print(f"\nFound {len(image_files)} image files in '{folder_path}'.")
#     print(f"They will be renamed with the prefix '{prefix}' (or sequentially if no prefix given).")
#     print("Example: Your first image will become '{}{}{}'".format(prefix or '', 1, os.path.splitext(image_files[0])[1]))

#     # --- Confirmation before proceeding ---
#     if confirm != 'yes': # This now relies on CONFIRM_RENAME variable
#         print("🚫 Renaming cancelled by user (CONFIRM_RENAME set to False).")
#         return

#     # --- 4. Rename the files ---
#     renamed_count = 0
#     # Use tqdm for a progress bar
#     for i, old_filename in enumerate(tqdm(image_files, desc="Renaming Images", unit="file")):
#         try:
#             # Get the original file extension
#             # os.path.splitext returns a tuple: (root, ext)
#             # e.g., ('my_image', '.jpg')
#             _, original_extension = os.path.splitext(old_filename)

#             # Construct the new filename
#             # The number 'i + 1' ensures numbering starts from 1
#             new_filename = f"{prefix}{i + 1}{original_extension}"

#             old_filepath = os.path.join(folder_path, old_filename)
#             new_filepath = os.path.join(folder_path, new_filename)

#             # Handle cases where the new filename might already exist (unlikely with sequential naming)
#             # Or if original file names contain the prefix and number.
#             if os.path.exists(new_filepath):
#                 print(f"❗ Warning: New filename '{new_filename}' already exists. Skipping '{old_filename}'.")
#                 continue

#             os.rename(old_filepath, new_filepath)
#             renamed_count += 1
#         except Exception as e:
#             print(f"\n❌ Error renaming '{old_filename}': {e}")
#             # Decide if you want to continue or stop on error
#             continue # Continue to the next file even if one fails

#     print(f"\n🎉 Renaming complete! Successfully renamed {renamed_count} out of {len(image_files)} image files.")
#     print(f"All renamed images are now in: {folder_path}")

# # Run the function when the script is executed
# if __name__ == "__main__":
#     rename_images_in_folder()










# # 👨‍🔧👨‍🔧👨‍🔧👨‍🔧 organized 50k images like: getting all the images and labels from all directories and put them in one folder with separate folder for labels and iamge
# import os
# import shutil
# from PIL import Image

# source_dir = r"A:\\human_detection"
# target_dir = r"A:\\New folder\\project5_human_detection"
# images_target = os.path.join(target_dir, "images")
# labels_target = os.path.join(target_dir, "labels")

# os.makedirs(images_target, exist_ok=True)
# os.makedirs(labels_target, exist_ok=True)

# # Common image extensions
# image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"]

# # Counter for renaming
# counter = 1

# # Helper function to find label for an image
# def find_label(image_path):
#     # Look for labels folder in the same parent directories
#     parent = os.path.dirname(image_path)
#     while parent != source_dir and parent != os.path.dirname(parent):
#         for folder in os.listdir(parent):
#             folder_path = os.path.join(parent, folder)
#             if os.path.isdir(folder_path) and "label" in folder.lower():
#                 # Check if label exists
#                 label_name = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
#                 label_path = os.path.join(folder_path, label_name)
#                 if os.path.exists(label_path):
#                     return label_path
#         parent = os.path.dirname(parent)
#     return None

# # Walk through all files
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         file_lower = file.lower()
#         file_path = os.path.join(root, file)

#         if any(file_lower.endswith(ext) for ext in image_exts):
#             # Open and convert image to JPG
#             try:
#                 img = Image.open(file_path).convert("RGB")
#             except:
#                 print(f"Skipping corrupted file: {file_path}")
#                 continue

#             new_image_name = f"{counter}.jpg"
#             new_image_path = os.path.join(images_target, new_image_name)
#             img.save(new_image_path, "JPEG")

#             # Find corresponding label
#             label_path = find_label(file_path)
#             if label_path:
#                 new_label_name = f"{counter}.txt"
#                 new_label_path = os.path.join(labels_target, new_label_name)
#                 shutil.copyfile(label_path, new_label_path)
#             else:
#                 print(f"Label not found for {file_path}")

#             counter += 1

# print("All images and labels copied, renamed, and converted successfully!")








# import os
# from ultralytics import YOLO
# from PIL import Image
# import shutil

# # Paths
# source_dir = r"A:\\New folder\\human1"
# images_target = r"A:\\New folder\\project5_human_detection\\images"
# labels_target = r"A:\\New folder\\project5_human_detection\\labels"

# # YOLO model (pretrained person detector)
# model = YOLO("yolov8n.pt")  # You can also use yolov8s.pt or yolov8m.pt for better accuracy

# # Supported image extensions
# image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif"]

# # Start counter from 50513
# counter = 50513

# # Function to convert image to jpg
# def convert_to_jpg(src_path, dst_path):
#     img = Image.open(src_path).convert("RGB")
#     img.save(dst_path, "JPEG")

# # Walk through all images in the source folder
# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         if any(file.lower().endswith(ext) for ext in image_exts):
#             img_path = os.path.join(root, file)
            
#             # New image name
#             new_image_name = f"{counter}.jpg"
#             new_image_path = os.path.join(images_target, new_image_name)
            
#             # Convert and save image as jpg
#             convert_to_jpg(img_path, new_image_path)
            
#             # Run YOLOv8 detection on image
#             results = model.predict(new_image_path, imgsz=640, conf=0.25)  # conf=0.25 can be adjusted
            
#             # Save labels in YOLO format
#             label_file = os.path.join(labels_target, f"{counter}.txt")
#             with open(label_file, "w") as f:
#                 for result in results:
#                     boxes = result.boxes.xywhn.cpu().numpy()  # normalized x_center, y_center, w, h
#                     cls = result.boxes.cls.cpu().numpy()
#                     for c, box in zip(cls, boxes):
#                         # Only keep class 0 (person) if you want
#                         if int(c) == 0:
#                             x, y, w, h = box
#                             f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
#             counter += 1

# print("All images processed and labels generated successfully!")














# 😭👨‍🔧👨‍🔧👨‍🔧 making train test split
# import os
# import random
# import shutil

# # Base directories
# base_dir = r"A:\\New folder\\project5_human_detection"
# images_dir = os.path.join(base_dir, "images")
# labels_dir = os.path.join(base_dir, "labels")

# # New subfolders
# splits = ["train", "validation", "test"]
# for split in splits:
#     os.makedirs(os.path.join(images_dir, split), exist_ok=True)
#     os.makedirs(os.path.join(labels_dir, split), exist_ok=True)

# # Get all image files
# all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
# random.shuffle(all_images)

# def move_files(file_list, target_split):
#     for img_file in file_list:
#         img_path = os.path.join(images_dir, img_file)
#         label_file = os.path.splitext(img_file)[0] + ".txt"
#         label_path = os.path.join(labels_dir, label_file)

#         # Destination paths
#         img_dest = os.path.join(images_dir, target_split, img_file)
#         label_dest = os.path.join(labels_dir, target_split, label_file)

#         # Move only if label exists
#         if os.path.exists(label_path):
#             shutil.move(img_path, img_dest)
#             shutil.move(label_path, label_dest)

# # Step 1: Move 7k to validation
# val_files = all_images[:7000]
# move_files(val_files, "validation")

# # Step 2: Move next 7k to test
# test_files = all_images[7000:14000]
# move_files(test_files, "test")

# # Step 3: Move remaining to train
# train_files = all_images[14000:]
# move_files(train_files, "train")

# print("✅ Dataset split completed successfully!")
