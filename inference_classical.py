import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow import keras
import glob
from tqdm import tqdm

class Config:
    def __init__(self):
        self.model_path = "quantum_data/models/classical_model_20241126_001917.h5"
        self.num_classes = 43
        self.input_size = (28, 28)
        self.batch_size = 32
        self.data_dir = "inference_data"
        self.output_dir = "inference_results"
        self.classes = {
            0:'University of Waterloo', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
            3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
            6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
            9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
            12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles',
            16:'Veh > 3.5 tons prohibited', 17:'No entry', 18:'General caution',
            19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve',
            22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
            25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
            29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
            32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
            35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left',
            38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory',
            41:'End of no passing', 42:'End no passing veh > 3.5 tons'
        }

def preprocess_image(image_path: str) -> np.ndarray:
    try:
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=(28, 28),
            color_mode='grayscale'
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0
        return image_array
    except Exception as e:
        raise Exception(f"Error preprocessing image {image_path}: {str(e)}")

def load_model(config: Config):
    try:
        model = keras.models.load_model(config.model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_images(model, image_paths: list, config: Config) -> dict:
    predictions = {}
    
    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            processed_image = preprocess_image(img_path)
            input_data = np.expand_dims(processed_image, axis=0)

            output = model.predict(input_data, verbose=0)
            predicted_class = np.argmax(output[0])
            confidence = float(output[0][predicted_class])
            
            predictions[img_path] = (predicted_class, confidence)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue
    
    return predictions

def visualize_predictions(predictions: dict, config: Config):
    os.makedirs(config.output_dir, exist_ok=True)
    
    OUTPUT_SIZE = (224, 224)
    FONT_SIZE = 16
    TEXT_MARGIN = 10
    
    try:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", FONT_SIZE)
            except:
                font = ImageFont.load_default()
        
        for img_path, (pred_class, confidence) in tqdm(predictions.items(), desc="Visualizing"):
            image = Image.open(img_path).convert('RGB')
            
            aspect_ratio = image.width / image.height
            if aspect_ratio > 1:
                new_width = OUTPUT_SIZE[0]
                new_height = int(OUTPUT_SIZE[0] / aspect_ratio)
            else:
                new_height = OUTPUT_SIZE[1]
                new_width = int(OUTPUT_SIZE[1] * aspect_ratio)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            label = config.classes[pred_class]
            text = f"Class {pred_class}: {label}\nConfidence: {confidence:.2%}"
            
            text_bbox = font.getbbox(text)
            text_height = text_bbox[3] - text_bbox[1]
            total_text_height = text_height + FONT_SIZE

            new_image = Image.new('RGB', (OUTPUT_SIZE[0], OUTPUT_SIZE[1] + total_text_height + 2*TEXT_MARGIN), 'white')
            
            paste_x = (OUTPUT_SIZE[0] - new_width) // 2
            paste_y = (OUTPUT_SIZE[1] - new_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            draw = ImageDraw.Draw(new_image)
            label_text = f"Class {pred_class}: {label}"
            conf_text = f"Confidence: {confidence:.2%}"

            label_bbox = font.getbbox(label_text)
            conf_bbox = font.getbbox(conf_text)
            label_width = label_bbox[2] - label_bbox[0]
            conf_width = conf_bbox[2] - conf_bbox[0]

            text_x = (OUTPUT_SIZE[0] - label_width) // 2
            text_y = OUTPUT_SIZE[1] + TEXT_MARGIN
            draw.text((text_x, text_y), label_text, fill=(0, 0, 0), font=font)

            conf_x = (OUTPUT_SIZE[0] - conf_width) // 2
            conf_y = text_y + FONT_SIZE + 5
            draw.text((conf_x, conf_y), conf_text, fill=(0, 0, 0), font=font)

            output_path = os.path.join(config.output_dir, f"pred_{os.path.basename(img_path)}")
            new_image.save(output_path)
            
    except Exception as e:
        print(f"Error visualizing predictions: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    config = Config()
    
    try:
        image_paths = glob.glob(os.path.join(config.data_dir, "*.jpg")) + \
                     glob.glob(os.path.join(config.data_dir, "*.png")) + \
                     glob.glob(os.path.join(config.data_dir, "*.ppm"))
        
        if not image_paths:
            raise Exception(f"No images found in {config.data_dir}")
        
        print("Loading model...")
        model = load_model(config)
        
        print("Making predictions...")
        predictions = predict_images(model, image_paths, config)
        
        print("Visualizing results...")
        visualize_predictions(predictions, config)
        
        print(f"Validation complete. Results saved to {config.output_dir}")
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")

if __name__ == "__main__":
    main()