import kagglehub
import os
import shutil

# Download latest version of the dataset
print("Downloading historic-art dataset from Kaggle...")
path = kagglehub.dataset_download("ansonnnnn/historic-art")
print(f"Downloaded to cache: {path}")

# Target directory — data/ folder inside the project
target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "historic-art")
os.makedirs(target_dir, exist_ok=True)

# Copy everything from the kagglehub cache to data/historic-art/
print(f"Copying dataset files to: {target_dir}")
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print(f"\nDone! Dataset is at: {target_dir}")
print("Contents:")
for item in os.listdir(target_dir):
    full = os.path.join(target_dir, item)
    if os.path.isdir(full):
        count = sum(len(files) for _, _, files in os.walk(full))
        print(f"  [DIR]  {item}/ ({count} files)")
    else:
        size_mb = os.path.getsize(full) / (1024 * 1024)
        print(f"  [FILE] {item} ({size_mb:.1f} MB)")
