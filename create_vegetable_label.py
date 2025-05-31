import os

# Path to your dataset root folder
dataset_path = "dataset/train"

# Get folder names and sort to match torchvision.datasets.ImageFolder order
class_names = sorted(entry.name for entry in os.scandir(dataset_path) if entry.is_dir())

# Save to a file
with open("vegetable_labels.txt", "w") as f:
    for name in class_names:
        f.write(f"{name}\n")

print("Saved vegetable_labels.txt with class names:")
print(class_names)