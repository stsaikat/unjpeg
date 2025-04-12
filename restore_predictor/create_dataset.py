import os
import random
import shutil

# -------- Config -------- #
def split_test():
    source_dir = "dataset/train"
    test_dir = "dataset/test"
    test_ratio = 0.1  # 10%

    # -------- Create test folder if it doesn't exist -------- #
    os.makedirs(test_dir, exist_ok=True)

    # -------- Get all image filenames -------- #
    all_images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(all_images)
    num_test = max(1, int(total_images * test_ratio))  # at least 1 image

    # -------- Randomly select 10% for test -------- #
    test_images = random.sample(all_images, num_test)

    # -------- Move files -------- #
    for img_name in test_images:
        src = os.path.join(source_dir, img_name)
        dst = os.path.join(test_dir, img_name)
        shutil.move(src, dst)
        print(f"Moved: {img_name}")

    print(f"\n✅ Moved {len(test_images)} images to {test_dir}")
