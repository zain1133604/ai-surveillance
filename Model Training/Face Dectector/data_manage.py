# import os
# import shutil
# import random

# # Base path where your datasets are located
# base_path = r"A:\New folder\face_detection"

# # New dataset output path
# output_path = os.path.join(base_path, "dataset")

# # Create output directories
# for sub in ["images/train", "images/valid", "labels/train", "labels/valid"]:
#     os.makedirs(os.path.join(output_path, sub), exist_ok=True)

# # Collect all image-label pairs
# all_pairs = []

# # Walk through all subfolders
# for root, dirs, files in os.walk(base_path):
#     if os.path.basename(root) == "images":  # only enter image folders
#         labels_folder = os.path.join(os.path.dirname(root), "labels")
#         if not os.path.exists(labels_folder):
#             continue

#         for img_file in files:
#             img_path = os.path.join(root, img_file)
#             label_file = os.path.splitext(img_file)[0] + ".txt"
#             label_path = os.path.join(labels_folder, label_file)

#             if os.path.exists(label_path):
#                 all_pairs.append((img_path, label_path))

# print(f"Found {len(all_pairs)} image-label pairs.")

# # Shuffle and split into 90% train, 10% valid
# random.shuffle(all_pairs)
# split_idx = int(len(all_pairs) * 0.9)
# train_pairs = all_pairs[:split_idx]
# valid_pairs = all_pairs[split_idx:]

# # Copy & rename function
# def copy_and_rename(pairs, split):
#     for i, (img, lbl) in enumerate(pairs, start=1):
#         new_img_name = f"{i}.jpg"
#         new_lbl_name = f"{i}.txt"

#         shutil.copy(img, os.path.join(output_path, f"images/{split}", new_img_name))
#         shutil.copy(lbl, os.path.join(output_path, f"labels/{split}", new_lbl_name))

# # Copy and rename files
# copy_and_rename(train_pairs, "train")
# copy_and_rename(valid_pairs, "valid")

# print(f"✅ Done! Train: {len(train_pairs)} | Valid: {len(valid_pairs)}")
# print(f"Dataset created at: {output_path}")




import os

# Base labels path
labels_base_path = r"A:\New folder\face_detection\dataset\labels"

# Folders to process
splits = ["train", "valid"]

for split in splits:
    folder = os.path.join(labels_base_path, split)
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)

            # Read and modify lines
            new_lines = []
            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:  # only if not empty line
                        parts[0] = "0"  # change class id to 0
                        new_lines.append(" ".join(parts))

            # Write back updated lines
            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

print("✅ All label IDs changed to 0 in train and valid folders.")
