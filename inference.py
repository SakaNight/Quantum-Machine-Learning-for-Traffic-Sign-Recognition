import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates import RandomLayers
import glob
from tqdm import tqdm
from tensorflow import keras
from typing import Dict, List, Tuple


class Config:
    def __init__(self, model_type='classical'):
        self.model_type = model_type
        self.model_path = f"quantum_data/models/{model_type}_model_latest.h5"
        self.num_classes = 43
        self.input_size = (28, 28)
        self.batch_size = 32
        self.data_dir = "demo"
        self.output_dir = "inference_results"

        # Quantum specific settings
        if model_type == 'quantum':
            self.n_qubits = 4
            self.n_layers = 1
            self.dev = qml.device("default.qubit", wires=self.n_qubits)
            self.rand_params = pnp.random.uniform(high=2 * np.pi, size=(self.n_layers, 4))


class ImageProcessor:
    @staticmethod
    def quantum_circuit(inputs, config):
        @qml.qnode(config.dev)
        def enhanced_circuit(inputs):
            for i in range(config.n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            RandomLayers(config.rand_params, wires=list(range(config.n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(config.n_qubits)]

        return enhanced_circuit(inputs)

    @staticmethod
    def process_quantum_blocks(image, config):
        out = np.zeros((14, 14, 4))
        for j in range(0, 28, 2):
            for k in range(0, 28, 2):
                block = [
                    float(image[j, k, 0]),
                    float(image[j, k + 1, 0]),
                    float(image[j + 1, k, 0]),
                    float(image[j + 1, k + 1, 0])
                ]
                q_results = ImageProcessor.quantum_circuit(block, config)
                for c in range(4):
                    out[j // 2, k // 2, c] = float(q_results[c])
        return out

    @staticmethod
    def preprocess_image(image_path: str, config: Config) -> np.ndarray:
        image = tf.keras.preprocessing.image.load_img(
            image_path,
            target_size=config.input_size,
            color_mode='grayscale'
        )
        image_array = tf.keras.preprocessing.image.img_to_array(image) / 255.0

        if config.model_type == 'quantum':
            return ImageProcessor.process_quantum_blocks(image_array, config)
        return image_array


class InferenceEngine:
    def __init__(self, config: Config):
        self.config = config
        self.model = self.load_model()

    def load_model(self):
        return keras.models.load_model(self.config.model_path)

    def predict_images(self, image_paths: List[str]) -> Dict[str, Tuple[int, float]]:
        predictions = {}
        for img_path in tqdm(image_paths, desc="Predicting"):
            try:
                processed_image = ImageProcessor.preprocess_image(img_path, self.config)
                input_data = np.expand_dims(processed_image, axis=0)
                output = self.model.predict(input_data, verbose=0)
                predicted_class = np.argmax(output[0])
                confidence = float(output[0][predicted_class])
                predictions[img_path] = (predicted_class, confidence)
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        return predictions


def visualize_predictions(predictions: Dict[str, Tuple[int, float]], config: Config):
    os.makedirs(config.output_dir, exist_ok=True)

    OUTPUT_SIZE = (224, 224)
    FONT_SIZE = 16
    TEXT_MARGIN = 10

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        font = ImageFont.truetype("arial.ttf", FONT_SIZE) if os.path.exists("arial.ttf") else ImageFont.load_default()

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
        new_image = Image.new('RGB', (OUTPUT_SIZE[0], OUTPUT_SIZE[1] + TEXT_MARGIN * 2 + FONT_SIZE * 2), 'white')
        new_image.paste(image, ((OUTPUT_SIZE[0] - new_width) // 2, (OUTPUT_SIZE[1] - new_height) // 2))

        draw = ImageDraw.Draw(new_image)
        label_text = f"Class {pred_class}: {label}"
        conf_text = f"Confidence: {confidence:.2%}"

        for text, y_offset in [(label_text, 0), (conf_text, FONT_SIZE + 5)]:
            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            x_pos = (OUTPUT_SIZE[0] - text_width) // 2
            y_pos = OUTPUT_SIZE[1] + TEXT_MARGIN + y_offset
            draw.text((x_pos, y_pos), text, fill=(0, 0, 0), font=font)

        output_path = os.path.join(config.output_dir, f"pred_{os.path.basename(img_path)}")
        new_image.save(output_path)


def main():
    model_type = input("Enter model type (classical/quantum): ").lower()
    if model_type not in ['classical', 'quantum']:
        raise ValueError("Model type must be either 'classical' or 'quantum'")

    config = Config(model_type)

    try:
        image_paths = []
        for ext in ['jpg', 'png', 'ppm']:
            image_paths.extend(glob.glob(os.path.join(config.data_dir, f"*.{ext}")))

        if not image_paths:
            raise Exception(f"No images found in {config.data_dir}")

        print(f"Found {len(image_paths)} images for inference")
        print("Loading model...")
        engine = InferenceEngine(config)

        print("Making predictions...")
        predictions = engine.predict_images(image_paths)

        print("Visualizing results...")
        visualize_predictions(predictions, config)

        print(f"Inference complete. Results saved to {config.output_dir}")

    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()