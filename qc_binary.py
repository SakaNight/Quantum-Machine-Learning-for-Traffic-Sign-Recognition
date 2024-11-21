import os
import cirq
import sympy
import numpy as np
import collections
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
    """Create a QNN model circuit and readout operation to go along with it."""
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

def create_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(64, [3, 3], activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1))
    return model

def create_fair_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model


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
    save_model_path = "./model_zoos/qc_binary_2cls.h5"

    print("Loading data from pickle files...")
    with open("./pkls/train_dataset_2cls.pkl", 'rb') as f:
        train_data = pickle.load(f)

    with open("./pkls/test_dataset_2cls.pkl", 'rb') as f:
        test_data = pickle.load(f)

    print("Processing training data...")
    train_paths = [train_data['image_paths'][i] for i in range(len(train_data['image_paths']))]
    train_labels = [train_data['labels'][i] for i in range(len(train_data['labels']))]

    print("Processing test data...")
    test_paths = [test_data['image_paths'][i] for i in range(len(test_data['image_paths']))]
    test_labels = [test_data['labels'][i] for i in range(len(test_data['labels']))]

    train_features = [preprocess_image(path) for path in tqdm(train_paths)]
    test_features = [preprocess_image(path) for path in tqdm(test_paths)]

    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.concatenate(train_features, axis=0)
    x_test = np.concatenate(test_features, axis=0)
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    print("Number of original training examples:", len(x_train))
    print("Number of original test examples:", len(x_test))


    """### 1.2 Downscale the images

    An image size of 28x28 is much too large for current quantum computers. Resize the image down to 4x4:
    """
    x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
    x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

    # x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    """### 1.4 Encode the data as quantum circuits

    To process images using a quantum computer, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed representing each pixel with a qubit, with the state depending on the value of the pixel. The first step is to convert to a binary encoding.
    """

    THRESHOLD = 0.5

    x_train_bin = np.array(x_train_small > THRESHOLD, dtype=np.float32)
    x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

    """The qubits at pixel indices with values that exceed a threshold, are rotated through an $X$ gate."""
    x_train_circ = [convert_to_circuit(x) for x in x_train_bin]
    x_test_circ = [convert_to_circuit(x) for x in x_test_bin]

    """Here is the circuit created for the first example (circuit diagrams do not show qubits with zero gates):"""

    SVGCircuit(x_train_circ[0])

    """Compare this circuit to the indices where the image value exceeds the threshold:"""

    bin_img = x_train_bin[0, :, :, 0]
    indices = np.array(np.where(bin_img)).T

    """Convert these `Cirq` circuits to tensors for `tfq`:"""

    x_train_tfcirc = tfq.convert_to_tensor(x_train_circ)
    x_test_tfcirc = tfq.convert_to_tensor(x_test_circ)

    """## 2. Quantum neural network

    There is little guidance for a quantum circuit structure that classifies images. Since the classification is based on the expectation of the readout qubit, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> propose using two qubit gates, with the readout qubit always acted upon. This is similar in some ways to running small a <a href="https://arxiv.org/abs/1511.06464" class="external">Unitary RNN</a> across the pixels.

    ### 2.1 Build the model circuit

    This following example shows this layered approach. Each layer uses *n* instances of the same gate, with each of the data qubits acting on the readout qubit.

    Start with a simple class that will add a layer of these gates to a circuit:
    """

    demo_builder = CircuitLayerBuilder(data_qubits=cirq.GridQubit.rect(4, 1),
                                       readout=cirq.GridQubit(-1, -1))

    circuit = cirq.Circuit()
    demo_builder.add_layer(circuit, gate=cirq.XX, prefix='xx')
    SVGCircuit(circuit)

    model_circuit, model_readout = create_quantum_model()

    """### 2.2 Wrap the model-circuit in a tfq-keras model

    Build the Keras model with the quantum components. This model is fed the "quantum data", from `x_train_circ`, that encodes the classical data. It uses a *Parametrized Quantum Circuit* layer, `tfq.layers.PQC`, to train the model circuit, on the quantum data.

    To classify these images, <a href="https://arxiv.org/pdf/1802.06002.pdf" class="external">Farhi et al.</a> proposed taking the expectation of a readout qubit in a parameterized circuit. The expectation returns a value between 1 and -1.
    """

    # Build the Keras model.
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(model_circuit, model_readout),
    ])

    """Next, describe the training procedure to the model, using the `compile` method.

    Since the the expected readout is in the range `[-1,1]`, optimizing the hinge loss is a somewhat natural fit.

    Note: Another valid approach would be to shift the output range to `[0,1]`, and treat it as the probability the model assigns to class `3`. This could be used with a standard a `tf.losses.BinaryCrossentropy` loss.

    To use the hinge loss here you need to make two small adjustments. First convert the labels, `y_train_nocon`, from boolean to `[-1,1]`, as expected by the hinge loss.
    """

    y_train_hinge = 2.0 * y_train - 1.0
    y_test_hinge = 2.0 * y_test - 1.0

    """Second, use a custiom `hinge_accuracy` metric that correctly handles `[-1, 1]` as the `y_true` labels argument.
    `tf.losses.BinaryAccuracy(threshold=0.0)` expects `y_true` to be a boolean, and so can't be used with hinge loss).
    """

    model.compile(
        loss=tf.keras.losses.Hinge(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[hinge_accuracy])

    # print(model.summary())

    """### Train the quantum model

    Now train the model—this takes about 45 min. If you don't want to wait that long, use a small subset of the data (set `NUM_EXAMPLES=500`, below). This doesn't really affect the model's progress during training (it only has 32 parameters, and doesn't need much data to constrain these). Using fewer examples just ends training earlier (5min), but runs long enough to show that it is making progress in the validation logs.
    """

    # # todo Check if a saved model exists
    # if os.path.exists(save_model_path):
    #     print(f"Loading model from {save_model_path}...")
    #     model = tf.keras.models.load_model(save_model_path, custom_objects={"hinge_accuracy": hinge_accuracy})
    # else:
    #     print("No pre-trained model found. Starting training from scratch.")

    NUM_EXAMPLES = len(x_train_tfcirc)

    x_train_tfcirc_sub = x_train_tfcirc[:NUM_EXAMPLES]
    y_train_hinge_sub = y_train_hinge[:NUM_EXAMPLES]

    """Training this model to convergence should achieve >85% accuracy on the test set."""

    qnn_history = model.fit(
        x_train_tfcirc_sub, y_train_hinge_sub,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(x_test_tfcirc, y_test_hinge))

    qnn_results = model.evaluate(x_test_tfcirc, y_test)
    print(":: qnn_accuracy = ", qnn_results[1])

    # # todo: Save the trained model
    # print(f"Saving trained model to {save_model_path}...")
    # model.save(save_model_path)

    # # 保存量子电路
    # with open("./model_zoos/qc_binary_circuit.json", "w") as f:
    #     f.write(cirq.to_json(model_circuit))
    #
    # # 保存量子操作符
    # with open("./model_zoos/qc_binary_readout.json", "w") as f:
    #     f.write(cirq.to_json(model_readout))

    """Note: The training accuracy reports the average over the epoch. The validation accuracy is evaluated at the end of each epoch.


    # ## 3. Classical neural network
    # 
    # While the quantum neural network works for this simplified MNIST problem, a basic classical neural network can easily outperform a QNN on this task. After a single epoch, a classical neural network can achieve >98% accuracy on the holdout set.
    # 
    # In the following example, a classical neural network is used for for the 3-6 classification problem using the entire 28x28 image instead of subsampling the image. This easily converges to nearly 100% accuracy of the test set.
    # """
    #
    # # Start
    #
    # model = create_classical_model()
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    #
    # model.summary()
    #
    # model.fit(x_train,
    #           y_train,
    #           batch_size=128,
    #           epochs=1,
    #           verbose=1,
    #           validation_data=(x_test, y_test))
    #
    # cnn_results = model.evaluate(x_test, y_test)
    #
    # """The above model has nearly 1.2M parameters. For a more fair comparison, try a 37-parameter model, on the subsampled images:"""
    #
    # model = create_fair_classical_model()
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(),
    #               metrics=['accuracy'])
    #
    # model.summary()
    #
    # model.fit(x_train_bin,
    #           y_train,
    #           batch_size=128,
    #           epochs=20,
    #           verbose=2,
    #           validation_data=(x_test_bin, y_test))
    #
    # fair_nn_results = model.evaluate(x_test_bin, y_test)
    #
    # """## 4. Comparison
    #
    # Higher resolution input and a more powerful model make this problem easy for the CNN. While a classical model of similar power (~32 parameters) trains to a similar accuracy in a fraction of the time. One way or the other, the classical neural network easily outperforms the quantum neural network. For classical data, it is difficult to beat a classical neural network.
    # """
    #
    # qnn_accuracy = qnn_results[1]
    # cnn_accuracy = cnn_results[1]
    # fair_nn_accuracy = fair_nn_results[1]
    #
    # print("="*100)
    #
    # sns.barplot(x=["Quantum", "Classical, full", "Classical, fair"],
    #             y=[qnn_accuracy, cnn_accuracy, fair_nn_accuracy])


if __name__ == '__main__':
    main()