import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import Callback

import os
import datetime
import argparse

from preprocess_data import *


# size of input images
WIDTH = 105
HEIGHT = 105

# Parameters of siamese network
num_filters_1 = 64
kernel_size_1 = (10, 10)
pooling_size_1 = (2, 2)

num_filters_2 = 128
kernel_size_2 = (7, 7)
pooling_size_2 = (2, 2)

num_filters_3 = 128
kernel_size_3 = (4, 4)
pooling_size_3 = (2, 2)

num_filters_4 = 256
kernel_size_4 = (4, 4)

n_dense_neurons = 4096
n_classes = 1


class DistanceLayer(tf.keras.layers.Layer):
    """A custom layer to compute L1 distance between two image representations
    (both are outputs of convnet)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, vect1, vect2):
        distance = tf.math.abs(vect1-vect2)
        return distance


def build_siamese_model():
    # two functions to initialize weights and bias of the network
    def initialize_weights(shape, name=None):
        return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

    def initialize_bias(shape, name=None):
        return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

    # The first part of siamese network (convnet)
    input_shape = (HEIGHT, WIDTH, 1)
    input_image = Input(input_shape)
    # 1st conv. layer
    c1 = Conv2D(num_filters_1, kernel_size_1, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
                kernel_regularizer=l2(1e-3))(input_image)
    p1 = MaxPool2D(pooling_size_1)(c1)
    # 2nd conv. layer
    c2 = Conv2D(num_filters_2, kernel_size_2, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
                bias_initializer=RandomNormal(mean=0.5,stddev=1e-2),
                kernel_regularizer=l2(1e-3))(p1)
    p2 = MaxPool2D(pooling_size_2)(c2)
    # 3th conv. layer
    c3 = Conv2D(num_filters_3, kernel_size_3, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
                bias_initializer=RandomNormal(mean=0.5,stddev=1e-2),
                kernel_regularizer=l2(1e-3))(p2)
    p3 = MaxPool2D(pooling_size_3)(c3)
    # 4th conv. layer
    c4 = Conv2D(num_filters_4, kernel_size_4, activation='relu',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1e-2),
                bias_initializer=RandomNormal(mean=0.5,stddev=1e-2),
                kernel_regularizer=l2(1e-3))(p3)
    # flattening and full connection layers
    f = tf.keras.layers.Flatten()(c4)
    dense_output = Dense(n_dense_neurons, activation='sigmoid',
                         kernel_initializer=RandomNormal(mean=0.0, stddev=2e-1),
                         bias_initializer=RandomNormal(mean=0.5,stddev=1e-2),
                         kernel_regularizer=l2(1e-3))(f)

    convnet = tf.keras.Model(input_image, dense_output,
                             name="conv_net")

    # The second part of siamese network
    # Generate the embeddings for two images
    input_image_1 = tf.keras.layers.Input(input_shape)
    input_image_2 = tf.keras.layers.Input(input_shape)
    convnet_output_1 = convnet(input_image_1)
    convnet_output_2 = convnet(input_image_2)

    # Compute L1 distance between the two embeddings of two different images
    distance_output = DistanceLayer()(convnet_output_1, convnet_output_2)

    # prediction layer
    class_output = Dense(n_classes, activation='sigmoid',
                         kernel_initializer=RandomNormal(mean=0.0, stddev=2e-1),
                         bias_initializer=RandomNormal(mean=0.5,stddev=1e-2),
                         kernel_regularizer=l2(1e-3))(distance_output)

    # Setting up the Siamese Network
    siamese_model = tf.keras.Model([input_image_1, input_image_2], class_output,
                                   name="siamese_net")
    # complile the model
    siamese_model.compile(loss=binary_crossentropy,
                          optimizer=Adam(learning_rate=0.00005),
                          #optimizer=SGD(lr = 0.001, momentum = 0.5),
                          metrics=['accuracy'])

    return convnet, siamese_model


class CompteNWayMetrics(Callback):
    def __init__(self, n_way, n_trials):
        super().__init__()
        self.n_way = n_way
        self.n_trials = n_trials
        self.nway_val_acc = []

    def on_epoch_end(self, epoch, logs=None):
        p_correct, _ = n_way_1shot_task(self.model, n_way=self.n_way,
                                        n_trials=self.n_trials,
                                        display=False)
        self.nway_val_acc.append(p_correct)


def train_and_evaluate(model, p_train=0.9, n_samples=0, n_way=20,
                       n_trials=10, batch_size=64, num_epochs=1, 
                       output_path="./", device="/device:GPU:0"):
    cp_filename = "epoch{epoch:02d}-val_acc{val_accuracy:.2f}.weights.h5"
    cp_path = os.path.join(output_path, cp_filename)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path,
                                                     save_weights_only=True,
                                                     save_freq="epoch",
                                                     verbose=1)

    nway_callback = CompteNWayMetrics(n_way, n_trials)

    log_dir = output_path + "logs/" +\
              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                 histogram_freq=1)

    train_acc = []
    val_acc = []
    with tf.device(device):
        path_pairs, labels = generateTrainingdata(n_samples=n_samples)
        n_train = int(np.floor(p_train * len(labels)))
        print("n_train=", n_train)

        train_data = DataGenerator(path_pairs[:n_train], labels[:n_train],
                             batch_size=batch_size,
                             shuffle=True)

        val_data = DataGenerator(path_pairs[n_train:], labels[n_train:],
                             batch_size=batch_size,
                             shuffle=False)

        history = model.fit(train_data,
                        epochs=num_epochs,
                        validation_data=val_data,
                        verbose=2,
                        callbacks=[cp_callback, nway_callback, tb_callback])

        train_acc.extend(history.history['accuracy'])
        val_acc.extend(history.history['val_accuracy'])

        with open(os.path.join(output_path, 'accuracies.pickle'), 'wb') as f:
            pickle.dump((train_acc, val_acc, nway_callback.nway_val_acc), f)

        return model


def prediction_plot(pos_image, support_set, predicted_image, idx_sample, res=0):
    n_way = len(support_set)
    fig, ax = plt.subplots(3, n_way, figsize=(25,15))

    predict = 'success' if res else 'failure'
    fig.suptitle('Trial #{}, {}\n'.format(idx_sample, predict), x=0, y=0.95,
                 fontsize=22, fontweight='bold')
    ax[0, 0].imshow(pos_image, cmap='gray')
    for i in range(n_way):
        ax[0, i].axis("off")
    ax[0, 0].set_title('Test image')

    for idx, img in enumerate(support_set):
        ax[1, idx%len(support_set)].imshow(img, cmap='gray')
        ax[1,idx%len(support_set)].axis('off')
    ax[1, 2].set_title('Support set')

    ax[2, 0].imshow(predicted_image, cmap='gray')
    ax[2, 0].set_title("Prediction\n (image with highest similarity)")
    for i in range(n_way):
        ax[2, i].axis("off")
    fig, ax = plt.subplots(figsize=(25,0.2))
    ax.axhline(0, 0, 1, linewidth=2, color='k')
    ax.axis("off")


def n_way_1shot_task(model, data_file=None, n_way=2, n_trials=2,
                    display=False):
    """Load and generate data for evaluation"""

    acc = np.zeros(n_trials)
    for i in range(n_trials):
        p1, p2, labels = generateNway1shotTestingData('omniglot_data', n_way=n_way)
        probs = model.predict((p1, p2), verbose=None)

        if np.argmax(probs) == np.argmax(labels):
            acc[i] = 1
        percent_correct = 100.0*np.mean(acc[0:i+1])

        if display:
            prediction_plot(p1[0], p2, p2[np.argmax(probs)], i+1, acc[i])

        #print("Trial #{%d}, average of accuracy {%.2f}\n" % (i, percent_correct))

    return np.mean(acc)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--n_trials", type=int, default=100)
    parse.add_argument("--batch_size", type=int, default=64)
    parse.add_argument("--num_epochs", type=int, default=10)
    parse.add_argument("--output_path", type=str, default="./")
    parse.add_argument("--n_way", type=int, default=20)
    parse.add_argument("--device", type=str, default="/device:GPU:0")
    args = parse.parse_args()

    omniglot_download("omniglot_data")
    _, siamese_model = build_siamese_model()

    # load weights into the model
    output_path = "./"
    siamese_model = train_and_evaluate(siamese_model, n_trials=arg.n_trials,
                                       batch_size=arg.batch_size, 
                                       num_epochs=args.num_epochs,
                                       output_path=arg.output_path,
                                       n_way=arg.n_way,
                                       device=arg.device)
