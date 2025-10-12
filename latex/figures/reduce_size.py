import os
from PIL import Image


def shrink_jpeg(filepath):
    try:
        with Image.open(filepath) as img:
            # Convert to RGB to avoid issues with palettes or transparency
            img = img.convert("RGB")
            width, height = img.size
            new_size = (width // 2, height // 2)
            img = img.resize(new_size, Image.LANCZOS)

            # Save optimized JPEG
            img.save(filepath, "JPEG", quality=85,
                     optimize=True, progressive=True)
            print(
                f"✅ Shrunk: {filepath} ({width}x{height} → {new_size[0]}x{new_size[1]})")
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")


def shrink_all_jpegs(root="."):
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith((".jpg", ".jpeg")):
                filepath = os.path.join(dirpath, filename)
                shrink_jpeg(filepath)


if __name__ == "__main__":
    print("🔍 Searching for JPEG files...")
    shrink_all_jpegs(".")
    print("🎉 Done shrinking all JPEGs!")
