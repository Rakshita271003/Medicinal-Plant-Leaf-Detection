import os
from PIL import Image

def clean_images(root_folder):
    corrupted = []
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            file_path = os.path.join(subdir, file)
            try:
                img = Image.open(file_path)   # Try opening
                img.verify()                  # Verify image integrity
            except Exception as e:
                print(f"❌ Corrupted: {file_path}")
                corrupted.append(file_path)
                os.remove(file_path)          # Delete corrupted file
    print(f"\n✅ Cleaning done. Removed {len(corrupted)} corrupted images.")

if __name__ == "__main__":
    dataset_dir = r"C:\Users\raksh\OneDrive\Desktop\projects\final year project- medicinal plant leaf detection\data"
    clean_images(dataset_dir)
