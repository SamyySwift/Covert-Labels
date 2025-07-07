import os
import shutil
import uuid

AUTH_DIR = "dataset/authentic"
os.makedirs(AUTH_DIR, exist_ok=True)

product_name = input("📷 Enter product name (e.g., 'aloe_vera_face_wash'): ").strip().replace(" ", "_")
batch_id = f"BATCH_{uuid.uuid4().hex[:8]}"
batch_path = os.path.join(AUTH_DIR, batch_id)
os.makedirs(batch_path, exist_ok=True)

print(f"\n🚀 Uploading images for {product_name} — batch: {batch_id}")
print("Paste the full path to each image. Type 'done' when finished.\n")

count = 0
while True:
    img_path = input("📂 Image path: ").strip()
    if img_path.lower() == 'done':
        break
    if os.path.exists(img_path):
        fname = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(batch_path, fname))
        count += 1
        print(f"✅ {fname} added.")
    else:
        print("❌ File not found. Try again.")

print(f"\n✅ Upload complete. {count} images saved to {batch_path}")