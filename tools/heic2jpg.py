import os
import pyheif
from PIL import Image


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

heic_image_folder = "/home/jamie/Works/Waterloo/ECE730/pics"
jpg_image_foler = "{}_jpg".format(heic_image_folder)
save_folder = "{}_labelme".format(heic_image_folder)

os.makedirs(save_folder, exist_ok=True)

print("Converting HEIC images to JPG format...")
for root, dirs, files in os.walk(heic_image_folder):
    for file in files:
        if file.endswith(".HEIC"):
            file_path = os.path.join(root, file)
            file_path_dst = os.path.join(save_folder, file.split(".")[0] + ".jpg")
            convert_heic_to_jpg(file_path, file_path_dst)

print("Done.")