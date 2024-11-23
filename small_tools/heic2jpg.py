import os
import pyheif
import shutil
from PIL import Image
from tqdm import tqdm


def convert_heic_to_jpg(input_path, output_path):
    """
    Convert a HEIC image to JPG format.

    Args:
        input_path (str): Path to the input HEIC file.
        output_path (str): Path to save the output JPG file.
    """
    # Load the HEIC image
    heif_file = pyheif.read(input_path)

    # Convert the HEIC image to a Pillow Image object
    image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )

    image.save(output_path, "JPEG")

heic_image_folder = "/home/jamie/Downloads/pics_campus_heic"
save_folder = "{}_labelme".format(heic_image_folder)

os.makedirs(save_folder, exist_ok=True)

print("Converting HEIC images to JPG format...")
for root, dirs, files in os.walk(heic_image_folder):
    for file in tqdm(files):
        suffix = file.split(".")[-1]
        file_path = os.path.join(root, file)
        if suffix in ['heic', 'HEIC']:
            file_path_dst = os.path.join(save_folder, file.split(".")[0] + ".jpg")
            convert_heic_to_jpg(file_path, file_path_dst)
        else:
            file_path_dst = os.path.join(save_folder, file)
            shutil.copy(file_path, file_path_dst)

print("Done.")