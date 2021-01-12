import tiledb
import tiledb.cloud as cloud
import os
import numpy as np
import tensorflow as tf
import json

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
TILEDB_USER_NAME = os.environ.get('TILEDB_USER_NAME')
TILEDB_PASSWD = os.environ.get('TILEDB_PASSWD')

BATCH_SIZE = 32
IMAGE_SHAPE = (64, 64, 3)
NUM_OF_CLASSES = 2

# We don't have to use all our data for this example. Just get some batches
# from training and validation datasets
TRAIN_DATA_SHAPE = 10 * BATCH_SIZE
VAL_DATA_SHAPE = 3 * BATCH_SIZE

EPOCHS = 1

TRAIN_IMAGES_ARRAY = "tiledb://gskoumas/train_ship_images"
TRAIN_LABELS_ARRAY = "tiledb://gskoumas/train_ship_segments"

VAL_IMAGES_ARRAY = "tiledb://gskoumas/val_ship_images"
VAL_LABELS_ARRAY = "tiledb://gskoumas/val_ship_segments"

MODEL_ARRAY = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/model"


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

            x_train = tiledb_images_obj[offset:offset + BATCH_SIZE]['rgb']. \
                view(np.float32).reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])

            # One hot encode Y
            y_train = tiledb_labels_obj[offset:offset + BATCH_SIZE]['label']
            y_train = [np.array([1.0, 0.0]) if item == 1.0 else np.array([0.0, 1.0]) for item in y_train]
            y_train = np.stack(y_train, axis=0).astype(np.float32)

            yield x_train, y_train


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(NUM_OF_CLASSES, activation='sigmoid')
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
    train_generator = gen_func(train_image_array, train_label_array, TRAIN_DATA_SHAPE)
    validate_generator = gen_func(val_image_array, val_label_array, VAL_DATA_SHAPE)

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

    return model.get_weights(), model.get_config()


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

model_weights, config = tiledb.cloud.udf.exec(train,
                                              gen_func=generator,
                                              model_func=create_model
                                              )

print(tiledb.cloud.last_udf_task().logs)

# Save model weights and architecture as a 1-D TileDB array with metadata
ctx = tiledb.Ctx()

dom = tiledb.Domain(
    tiledb.Dim(name="weights_indx", domain=(0, len(model_weights) - 1), tile=len(model_weights), dtype=np.int32),
    ctx=ctx,
)

attrs = [
    tiledb.Attr(dtype=np.float32, var=True, ctx=ctx)
]

schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs, ctx=ctx)

tiledb.Array.create(MODEL_ARRAY, schema, ctx=ctx)

shapes = {}
reshaped_arrays = []

for indx, array in enumerate(model_weights):
    shapes["shape_" + str(indx)] = array.shape
    reshaped_arrays.append(array.flatten())

with tiledb.open(MODEL_ARRAY, 'w') as tf_model_tiledb:
    tf_model_tiledb[:] = np.array(reshaped_arrays, dtype='O')
    tf_model_tiledb.meta["architecture"] = json.dumps(config)

    for key in shapes:
        tf_model_tiledb.meta[key] = shapes[key]
