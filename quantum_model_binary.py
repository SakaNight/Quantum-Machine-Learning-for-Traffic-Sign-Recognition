import cirq
import sympy
import numpy as np
from cirq.contrib.svg import SVGCircuit
import tensorflow as tf
import tensorflow_quantum as tfq
import pickle
from tqdm import tqdm
import cv2


def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_quantum_model():
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


def preprocess_image(image_path: str, size: tuple = (28, 28)) -> np.ndarray:
    # 1) transfer into gray scale, 2) resize into (28,28)
    # the output size should be (num_images, 28, 28)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize the image to the specified size
    resized_image = cv2.resize(gray_image, size)

    # Add a batch dimension (1, height, width) for consistency
    preprocessed_image = np.expand_dims(resized_image, axis=0)

    return preprocessed_image


def main():
    EPOCHS = 40
    BATCH_SIZE = 32

    print("Loading data from pickle files...")
    with open("./pkls/train_dataset_2cls_027.pkl", 'rb') as f:
        train_data = pickle.load(f)

    with open("./pkls/test_dataset_2cls_027.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print("Processing training data...")
    train_paths = [train_data['image_paths'][i] for i in range(len(train_data['image_paths']))]
    train_labels = [train_data['labels'][i] for i in range(len(train_data['labels']))]

    print("Processing test data...")
    test_paths = [test_data['image_paths'][i] for i in range(len(test_data['image_paths']))]
    test_labels = [test_data['labels'][i] for i in range(len(test_data['labels']))]

    train_features = [preprocess_image(path) for path in tqdm(train_paths)]
    test_features = [preprocess_image(path) for path in tqdm(test_paths)]

    x_train = np.concatenate(train_features, axis=0)
    x_test = np.concatenate(test_features, axis=0)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))


    # Downscale the images
    x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
    x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

    # Encode the data as quantum circuits

    THRESHOLD = 0.5

    x_train_bin = np.array(x_train_small > THRESHOLD, dtype=np.float32)
    x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

    """The qubits at pixel indices with values that exceed a threshold, are rotated through an $X$ gate."""
    x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
    x_test_circ = [convert_to_circuit(x) for x in x_test_bin]

    SVGCircuit(x_train_circ[0])

    """Compare this circuit to the indices where the image value exceeds the threshold:"""
    bin_img = x_train_bin[0, :, :, 0]
    indices = np.array(np.where(bin_img)).T

    """Convert these `Cirq` circuits to tensors for `tfq`:"""
    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    demo_builder = CircuitLayerBuilder(data_qubits=cirq.GridQubit.rect(4, 1),
                                       readout=cirq.GridQubit(-1, -1))

    circuit = cirq.Circuit()
    demo_builder.add_layer(circuit, gate=cirq.XX, prefix='xx')
    SVGCircuit(circuit)

    model_circuit, model_readout = create_quantum_model()

    # Build the Keras model.
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
    ])

    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    print(model.summary())

    # Train the quantum model
    NUM_EXAMPLES = len(x_train_tfcirc)
    x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
    y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

    qnn_history = model.fit(
        x_train_tfcirc_sub, y_train_hinge_sub,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_hinge))

    qnn_results = model.evaluate(x_test_tfcirc, y_test)
    print(":: qnn_accuracy = ", qnn_results[1])


if __name__ == '__main__':
    main()