import os
import csv
from PIL import Image

def add_images_to_csv(image_folder, output_csv):
    existing_data = []
    with open(output_csv, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        header = next(reader)
        existing_data = list(reader)
    
    new_data = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(image_folder, filename)
            with Image.open(img_path) as img:
                width, height = img.size
            
            new_row = [
                filename,  # Filename
                str(width),  # Width
                str(height),  # Height
                '0',  # Roi.X1
                '0',  # Roi.Y1
                '0',  # Roi.X2
                '0',  # Roi.Y2
                '27'   # ClassId
            ]
            new_data.append(new_row)
    
    all_data = existing_data + new_data
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])
        writer.writerows(all_data)

image_folder = "uw"
output_csv = "GT-final_test.csv"

add_images_to_csv(image_folder, output_csv)