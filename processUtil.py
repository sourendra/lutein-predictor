import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def preprocess_image(filename):
    img_width, img_height = 256, 256
    img = tf.keras.utils.load_img(filename, target_size=(img_width, img_height))
    img = tf.keras.utils.img_to_array(img) / 255.0  # Rescale pixel values to [0, 1]
    return img
    pass


class Process:
    image_model = None

    def __init__(self):
        pass

    def loadCSVDataAndPreProcess(self):
        # Load the CSV data
        marigold_df = pd.read_csv("Marigold/marigoldlutein.csv")
        marigold_df.head()

        # %%
        imagedata = cv2.imread('Marigold/' + marigold_df.file[35])
        rgb = cv2.cvtColor(imagedata, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb, cmap=plt.cm.Spectral)

        # Preprocess all images
        images = np.array([preprocess_image("Marigold/" + filename) for filename in marigold_df['file']])

        # %%
        # Labels
        target = marigold_df['lutein'].values

        # %%
        # Define the model
        Process.image_model = tf.keras.Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1)  # No activation function for regression
        ])

        # %%
        # Compile the model
        Process.image_model.compile(optimizer='adam',
                                    loss='mean_squared_error',  # Use MSE for regression
                                    metrics=['mse'])

        # %%
        # Train the model
        Process.image_model.fit(images, target, epochs=20, batch_size=8, validation_split=0.25)

        pass
