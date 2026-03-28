import cv2
import numpy as np
import os
import glob


# --- augmentation functions ---
# each takes an image (numpy array) and returns the augmented image

def add_gaussian_noise(img):
    # pick a random sigma between 5 and 20
    sigma = np.random.uniform(5, 20)
    # generate noise with same shape as image
    noise = np.random.normal(0, sigma, img.shape)
    # add noise and clip to valid range
    noisy = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy


def jpeg_compress(img):
    # pick a random quality between 20 and 80
    quality = np.random.randint(20, 81)
    # encode image as jpeg in memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', img, encode_param)
    # decode it back
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return compressed


def dpi_downsample(img, target_dpi):
    # original is 300 dpi, scale down to target dpi
    scale = target_dpi / 300.0
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)
    # resize the image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized


def random_crop(img):
    h, w = img.shape[:2]
    # randomly pick how much to remove from each border (1-3%)
    top = int(h * np.random.uniform(0.01, 0.03))
    bottom = int(h * np.random.uniform(0.01, 0.03))
    left = int(w * np.random.uniform(0.01, 0.03))
    right = int(w * np.random.uniform(0.01, 0.03))
    # crop the image
    cropped = img[top:h - bottom, left:w - right]
    return cropped


def reduce_bit_depth(img):
    # convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # quantize from 8-bit (256 levels) to 4-bit (16 levels)
    reduced = (gray // 16) * 16
    return reduced


# --- main script ---

def main():
    # three source folders (relative to ForensicsDetective/)
    source_folders = [
        "word_pdfs_png",
        "google_docs_pdfs_png",
        "python_pdfs_png",
    ]

    # base output directory
    output_base = "data/augmented_images"

    # augmentation names for folder structure
    aug_names = ["original", "gaussian", "jpeg", "dpi", "crop", "bitdepth"]

    # create all output directories
    for aug in aug_names:
        for folder in source_folders:
            os.makedirs(os.path.join(output_base, aug, folder), exist_ok=True)
    print("Created output directories.")

    # collect all image paths: list of (folder_name, filepath)
    all_images = []
    for folder in source_folders:
        pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
        for p in pngs:
            all_images.append((folder, p))

    total = len(all_images)
    print(f"Found {total} images across {len(source_folders)} folders.")
    print(f"Will produce {total * 6} augmented + {total} originals = {total * 7} total images.\n")

    augmented_count = 0
    original_count = 0

    for idx, (folder, filepath) in enumerate(all_images):
        # load the image
        img = cv2.imread(filepath)
        if img is None:
            print(f"  WARNING: could not read {filepath}, skipping.")
            continue

        # get the image name without extension
        name = os.path.splitext(os.path.basename(filepath))[0]

        # print progress
        print(f"[{idx + 1}/{total}] Augmenting {name}.png ...")

        # save the original copy
        orig_path = os.path.join(output_base, "original", folder, f"{name}.png")
        cv2.imwrite(orig_path, img)
        original_count += 1

        # apply gaussian noise
        noisy = add_gaussian_noise(img)
        cv2.imwrite(os.path.join(output_base, "gaussian", folder, f"{name}_gaussian.png"), noisy)
        augmented_count += 1

        # apply jpeg compression
        compressed = jpeg_compress(img)
        cv2.imwrite(os.path.join(output_base, "jpeg", folder, f"{name}_jpeg.png"), compressed)
        augmented_count += 1

        # apply dpi downsampling to 150 dpi
        dpi150 = dpi_downsample(img, 150)
        cv2.imwrite(os.path.join(output_base, "dpi", folder, f"{name}_dpi150.png"), dpi150)
        augmented_count += 1

        # apply dpi downsampling to 72 dpi
        dpi72 = dpi_downsample(img, 72)
        cv2.imwrite(os.path.join(output_base, "dpi", folder, f"{name}_dpi72.png"), dpi72)
        augmented_count += 1

        # apply random cropping
        cropped = random_crop(img)
        cv2.imwrite(os.path.join(output_base, "crop", folder, f"{name}_crop.png"), cropped)
        augmented_count += 1

        # apply bit depth reduction
        reduced = reduce_bit_depth(img)
        cv2.imwrite(os.path.join(output_base, "bitdepth", folder, f"{name}_bitdepth.png"), reduced)
        augmented_count += 1

    # print final summary
    print(f"\nDone!")
    print(f"Originals saved: {original_count}")
    print(f"Augmented images saved: {augmented_count}")
    print(f"Total images: {original_count + augmented_count}")


if __name__ == "__main__":
    main()
