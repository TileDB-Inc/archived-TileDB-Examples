import tiledb
import tiledb.cloud as cloud
import os
import numpy as np
import boto3
import tensorflow as tf

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
TILEDB_USER_NAME = os.environ.get('TILEDB_USER_NAME')
TILEDB_PASSWD = os.environ.get('TILEDB_PASSWD')

BATCH_SIZE = 64
IMAGE_SHAPE = (128, 128, 3)
NUM_OF_CLASSES = 2

# We don't have to use all our data for this example. Just get some batches
# from training and validation datasets
TRAIN_DATA_SHAPE = 50 * BATCH_SIZE
VAL_DATA_SHAPE = 5 * BATCH_SIZE

EPOCHS = 1

TRAIN_IMAGES_ARRAY = "tiledb://gskoumas/train_ship_images"
TRAIN_LABELS_ARRAY = "tiledb://gskoumas/train_ship_segments"

VAL_IMAGES_ARRAY = "tiledb://gskoumas/val_ship_images"
VAL_LABELS_ARRAY = "tiledb://gskoumas/val_ship_segments"


def generator(tiledb_images_obj, tiledb_labels_obj, shape):
    """
    Yields the next training batch.
    """

    while True:  # Loop forever so the generator never terminates

        # Get index to start each batch
        for offset in range(0, shape, BATCH_SIZE):

            # Get the samples you'll use in this batch. We have to convert structured numpy arrays to
            # numpy arrays.

            # Avoid reshaping error in last batch
            if offset + BATCH_SIZE > shape:
                batch_size = shape - offset

            x_train = tiledb_images_obj[offset:offset + BATCH_SIZE]['rgb'].\
                view(np.uint8).reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])

            # Scale RGB
            x_train = x_train.astype(np.float32) / 255.0

            # One hot encode Y
            y_train = tiledb_labels_obj[offset:offset + BATCH_SIZE]['label']
            y_train = [np.array([1.0, 0.0]) if item == 1 else np.array([0.0, 1.0]) for item in y_train]
            y_train = np.stack(y_train, axis=0).astype(np.float32)

            yield x_train, y_train


def create_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(NUM_OF_CLASSES)
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


def train(gen_func, model_func):

    # Open TileDB arrays for training
    train_image_array = tiledb.open(TRAIN_IMAGES_ARRAY)
    train_label_array = tiledb.open(TRAIN_LABELS_ARRAY)
    val_image_array = tiledb.open(VAL_IMAGES_ARRAY)
    val_label_array = tiledb.open(VAL_LABELS_ARRAY)

    # Create generators to read batches from TileDB arrays
    train_generator = gen_func(train_image_array, train_label_array, TRAIN_DATA_SHAPE, BATCH_SIZE)
    validate_generator = gen_func(val_image_array, val_label_array, VAL_DATA_SHAPE, BATCH_SIZE)

    # Create a model and pass it to the udf
    model = model_func()

    # Print model summary
    model.summary()

    # Train model
    model.fit(
        train_generator,
        steps_per_epoch=TRAIN_DATA_SHAPE // BATCH_SIZE,
        epochs=1,
        validation_data=validate_generator,
        validation_steps=VAL_DATA_SHAPE // BATCH_SIZE)

    if not os.path.exists('models'):
        os.makedirs('models')

    model.save('models/model.h5')

    client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    client.upload_file('models/model.h5', 'tiledb-gskoumas', 'airbus_ship_detection_tiledb/models/model.h5')

    return


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

tiledb.cloud.udf.exec(train,
                      gen_func=generator,
                      model_func=create_model
                      )

print(tiledb.cloud.last_udf_task().logs)