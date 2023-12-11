import tensorflow as tf
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, MaxPooling2D, ZeroPadding2D, BatchNormalization, Input

class Vgg():
    def __init__(self,n):
        """Instancia da vgg para n classes"""
        img_input = Input(shape=(32, 32, 3))
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(img_input)
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        x = Flatten(name="flatten")(x)
        x = Dense(4096, activation="relu", name="fc1")(x)
        x = Dense(4096, activation="relu", name="fc2")(x)
        # x = BatchNormalization()(x)

        output = Dense(n, activation='softmax', name="predictions")(x)

        self.model = tf.keras.Model(inputs=img_input, outputs=output)
        