# import os
# import shutil
# import random

# # Input + output folders
# root_dir = r"A:\\images"
# output_dir = os.path.join(root_dir, "dataset")

# # Create dataset structure
# for split in ["train", "val", "test"]:
#     os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

# # Counters for naming
# counter = 1

# # Collect all images + labels first
# all_files = []

# for folder in os.listdir(root_dir):
#     folder_path = os.path.join(root_dir, folder)
#     if not os.path.isdir(folder_path) or folder == "dataset":
#         continue

#     # Assign class_id by folder name
#     if folder.lower().startswith("gun"):
#         class_id = 0
#     elif folder.lower().startswith("knife"):
#         class_id = 1
#     else:
#         continue

#     for split in ["train", "val", "test"]:
#         split_path = os.path.join(folder_path, split)
#         if not os.path.exists(split_path):
#             continue

#         for sub in ["images", "labels"]:
#             sub_path = os.path.join(split_path, sub)
#             if not os.path.exists(sub_path):
#                 continue

#         images_path = os.path.join(split_path, "images")
#         labels_path = os.path.join(split_path, "labels")

#         if not os.path.exists(images_path) or not os.path.exists(labels_path):
#             continue

#         for img_name in os.listdir(images_path):
#             if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
#                 img_path = os.path.join(images_path, img_name)
#                 label_path = os.path.join(labels_path, os.path.splitext(img_name)[0] + ".txt")

#                 if os.path.exists(label_path):
#                     all_files.append((img_path, label_path, class_id))

# # Shuffle before splitting
# random.shuffle(all_files)

# # Train/val/test split
# n = len(all_files)
# val_size = int(0.1 * n)
# test_size = int(0.1 * n)

# val_files = all_files[:val_size]
# test_files = all_files[val_size:val_size + test_size]
# train_files = all_files[val_size + test_size:]

# splits = {
#     "train": train_files,
#     "val": val_files,
#     "test": test_files
# }

# # Copy + rename
# counter = 1
# for split, files in splits.items():
#     for img_path, label_path, class_id in files:
#         new_img_name = f"{counter}.jpg"
#         new_label_name = f"{counter}.txt"

#         # Copy image
#         shutil.copy(img_path, os.path.join(output_dir, "images", split, new_img_name))

#         # Edit + copy label
#         with open(label_path, "r") as f:
#             lines = f.readlines()

#         new_lines = []
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) > 0:
#                 parts[0] = str(class_id)  # overwrite class id
#                 new_lines.append(" ".join(parts))

#         with open(os.path.join(output_dir, "labels", split, new_label_name), "w") as f:
#             f.write("\n".join(new_lines))

#         counter += 1

# print("✅ Dataset created at:", output_dir)



# import os

# base_path = r"A:\\images"

# # valid image extensions
# image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")

# gun_count = 0
# knife_count = 0

# for folder in os.listdir(base_path):
#     folder_path = os.path.join(base_path, folder)
#     if not os.path.isdir(folder_path):
#         continue

#     # 🚫 skip the merged dataset folder
#     if folder.lower() == "dataset":
#         continue

#     # check if it's a gun or knife dataset
#     if folder.lower().startswith("gun"):
#         target = "gun"
#     elif folder.lower().startswith("knife"):
#         target = "knife"
#     else:
#         continue

#     # walk through subfolders (train/images, val/images, test/images, or images/)
#     for root, _, files in os.walk(folder_path):
#         for f in files:
#             if f.lower().endswith(image_exts):
#                 if target == "gun":
#                     gun_count += 1
#                 else:
#                     knife_count += 1

# print(f"Total Gun Images  : {gun_count}")
# print(f"Total Knife Images: {knife_count}")
# print(f"Total Images      : {gun_count + knife_count}")



# import os
# from tqdm import tqdm

# base_path = r"A:\\images\dataset\\labels"

# gun_count = 0
# knife_count = 0
# total_files = 0

# # Loop through train/val/test folders
# for split in ["train", "val", "test"]:
#     folder = os.path.join(base_path, split)
#     if not os.path.exists(folder):
#         continue

#     # Get all .txt files in this split
#     txt_files = [f for f in os.listdir(folder) if f.endswith(".txt")]
#     total_files += len(txt_files)

#     for file in tqdm(txt_files, desc=f"Processing {split}", unit="file"):
#         file_path = os.path.join(folder, file)
#         with open(file_path, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 parts = line.strip().split()
#                 if not parts:
#                     continue
#                 class_id = int(parts[0])
#                 if class_id == 0:
#                     gun_count += 1
#                 elif class_id == 1:
#                     knife_count += 1

# print("\n===== FINAL COUNT =====")
# print(f"Total Gun Labels   : {gun_count}")
# print(f"Total Knife Labels : {knife_count}")
# print(f"Total Labels       : {gun_count + knife_count}")
# print(f"Total Files Scanned: {total_files}")










# import os
# import shutil
# import cv2

# # Paths
# BASE_DIR = r"A:\\images"
# OUTPUT_DIR = os.path.join(BASE_DIR, "all-images-folders")

# # Create output directory if not exists
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# def find_images_labels(folder):
#     """Find images and their corresponding labels in any nested structure."""
#     images_labels = []
#     for root, dirs, files in os.walk(folder):
#         if 'images' in dirs and 'labels' in dirs:
#             images_folder = os.path.join(root, 'images')
#             labels_folder = os.path.join(root, 'labels')
#             for img_file in os.listdir(images_folder):
#                 if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
#                     img_path = os.path.join(images_folder, img_file)
#                     label_file = os.path.splitext(img_file)[0] + ".txt"
#                     label_path = os.path.join(labels_folder, label_file)
#                     images_labels.append((img_path, label_path))
#     return images_labels

# def draw_boxes_on_image(img_path, label_path):
#     """Draw all YOLO bounding boxes on an image."""
#     img = cv2.imread(img_path)
#     if img is None:
#         return None
#     h, w = img.shape[:2]

#     if os.path.exists(label_path):
#         with open(label_path, 'r') as f:
#             for line in f.readlines():
#                 parts = line.strip().split()
#                 if len(parts) < 5:
#                     continue
#                 # Support multiple boxes per label line if more than 5 values
#                 num_boxes = len(parts) // 5
#                 for i in range(num_boxes):
#                     cls, x_center, y_center, width, height = map(float, parts[i*5:(i+1)*5])
#                     x1 = int((x_center - width/2) * w)
#                     y1 = int((y_center - height/2) * h)
#                     x2 = int((x_center + width/2) * w)
#                     y2 = int((y_center + height/2) * h)
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#     return img

# def process_folder(folder_path, output_base):
#     folder_name = os.path.basename(folder_path)
#     print(f"Processing folder: {folder_name}")
#     output_folder = os.path.join(output_base, folder_name)
#     os.makedirs(output_folder, exist_ok=True)
#     output_images_folder = os.path.join(output_folder, "images")
#     os.makedirs(output_images_folder, exist_ok=True)

#     images_labels = find_images_labels(folder_path)
#     if not images_labels:
#         print(f"Skipping {folder_name} because images/labels folder is missing")
#         return

#     for img_path, label_path in images_labels:
#         img_with_boxes = draw_boxes_on_image(img_path, label_path)
#         if img_with_boxes is not None:
#             output_path = os.path.join(output_images_folder, os.path.basename(img_path))
#             cv2.imwrite(output_path, img_with_boxes)

# # Iterate over all folders starting with gun or knife
# for folder in os.listdir(BASE_DIR):
#     folder_path = os.path.join(BASE_DIR, folder)
#     if os.path.isdir(folder_path) and (folder.lower().startswith("gun") or folder.lower().startswith("knife")):
#         process_folder(folder_path, OUTPUT_DIR)

# print("Done processing all folders.")






# # 😶😶😶😶 this code is for the fire dectection model dataset preparetation 
import os
import shutil
import random

# Base path
base_dir = r"A:\New folder\fire dataset"
final_dir = os.path.join(base_dir, "final-dataset")

# Create new folder structure
for sub in ["images/train", "images/val", "images/test",
            "labels/train", "labels/val", "labels/test"]:
    os.makedirs(os.path.join(final_dir, sub), exist_ok=True)

# Collect all image-label pairs
image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
pairs = []

for root, dirs, files in os.walk(base_dir):
    if "final-dataset" in root:  # skip the new folder
        continue
    for file in files:
        if file.lower().endswith(image_exts):
            img_path = os.path.join(root, file)
            # match label path
            name, _ = os.path.splitext(file)
            label_path = os.path.join(root.replace("images", "labels"), name + ".txt")
            if os.path.exists(label_path):
                pairs.append((img_path, label_path))

print(f"Total pairs found: {len(pairs)}")

# Shuffle dataset
random.shuffle(pairs)

# Split
total = len(pairs)
train_count = int(total * 0.89)
val_count = int(total * 0.10)
test_count = total - train_count - val_count

splits = {
    "train": pairs[:train_count],
    "val": pairs[train_count:train_count+val_count],
    "test": pairs[train_count+val_count:]
}

# Copy into final structure with sequential names
counter = 1
for split, data in splits.items():
    for img_path, label_path in data:
        new_img_name = f"{counter}.jpg"
        new_lbl_name = f"{counter}.txt"

        shutil.copy(img_path, os.path.join(final_dir, "images", split, new_img_name))
        shutil.copy(label_path, os.path.join(final_dir, "labels", split, new_lbl_name))

        counter += 1

print("Dataset organized successfully into:", final_dir)
