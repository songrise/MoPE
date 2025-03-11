import os
from PIL import Image
import shutil


def is_image_pure_black(image_path):
    """Check if the image is pure black."""
    with Image.open(image_path) as img:
        img = img.convert("L")  # Convert to grayscale
        if not img.getextrema()[1]:  # Check if the maximum pixel value is 0
            return True
    return False


def create_black_image(image_path):
    """Create a black image and save it."""
    black_img = Image.new("RGB", (224, 224), (0, 0, 0))
    black_img.save(image_path, "PNG")


def process_image_folders(base_folder, processed_folder):
    """Process image folders to identify and move good images."""
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    for subdir in os.listdir(base_folder):
        # Check if the folder name ends with '_aligned'
        if subdir.endswith("_aligned"):
            aligned_folder = os.path.join(base_folder, subdir)
            if os.path.isdir(aligned_folder):
                # Remove the '_aligned' postfix from the folder name
                processed_subdir_name = subdir.replace("_aligned", "")
                processed_subdir = os.path.join(processed_folder, processed_subdir_name)
                if not os.path.exists(processed_subdir):
                    os.makedirs(processed_subdir)

                good_images = []
                for filename in os.listdir(aligned_folder):
                    if filename.endswith(".bmp"):
                        image_path = os.path.join(aligned_folder, filename)
                        if not is_image_pure_black(image_path):
                            # Open the image and save it in PNG format
                            with Image.open(image_path) as img:
                                new_filename = os.path.splitext(filename)[0] + ".png"
                                new_image_path = os.path.join(
                                    processed_subdir, new_filename
                                )
                                img.save(new_image_path, "PNG")
                                good_images.append(new_image_path)

                # Ensure there are at least 8 images in the processed folder
                if len(good_images) < 8:
                    if good_images:
                        last_image = good_images[-1]
                        while len(good_images) < 8:
                            new_image_path = os.path.join(
                                processed_subdir, f"copy_{len(good_images) + 1}.png"
                            )
                            shutil.copy(last_image, new_image_path)
                            good_images.append(new_image_path)
                        print(
                            f"{processed_subdir_name}: Added copies to reach 8 images."
                        )
                    else:
                        for i in range(8):
                            black_image_path = os.path.join(
                                processed_subdir, f"black_{i + 1}.png"
                            )
                            create_black_image(black_image_path)
                            good_images.append(black_image_path)
                        print(
                            f"{processed_subdir_name}: No good images. Added 8 black images."
                        )


if __name__ == "__main__":
    base_folder = "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/processed"
    processed_folder = "/root/autodl-tmp/mmsd/mmsd_raw_data/utterances_final/filtered"
    process_image_folders(base_folder, processed_folder)
    print("Processing complete.")
