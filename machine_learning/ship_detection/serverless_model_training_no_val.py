import tiledb
import tiledb.cloud as cloud
import os
import numpy as np
import tensorflow as tf

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
TILEDB_USER_NAME = os.environ.get('TILEDB_USER_NAME')
TILEDB_PASSWD = os.environ.get('TILEDB_PASSWD')

BATCH_SIZE = 32
IMAGE_SHAPE = (64, 64, 3)

TRAIN_IMAGES_ARRAY = "tiledb://TileDB-Inc/train_ship_images"
TRAIN_SEGMENTS_ARRAY = "tiledb://TileDB-Inc/train_ship_segments"


def generator(tiledb_images, tiledb_segments, data_size):
    """
    Yields the next training batch.
    """
    while True:
        for offset in range(0, data_size, BATCH_SIZE):
            # Get X
            X = tiledb_images[offset:offset + BATCH_SIZE]['rgb'].\
                view(np.float32).reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])

            # Get Y
            Y = tiledb_segments[offset:offset + BATCH_SIZE]['label']
            yield X, Y


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
        tf.keras.layers.Dense(2, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def train(gen_func, model_func):
    # Open TileDB arrays
    train_image_array = tiledb.open(TRAIN_IMAGES_ARRAY)
    train_segment_array = tiledb.open(TRAIN_SEGMENTS_ARRAY)

    # Create generator
    train_generator = gen_func(train_image_array, train_segment_array, 1000 * BATCH_SIZE)

    # Create a model
    model = model_func()

    # Train model
    model.fit(
        train_generator,
        steps_per_epoch=1000,
        epochs=10)

    return model


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

model = tiledb.cloud.udf.exec(train, gen_func=generator, model_func=create_model)