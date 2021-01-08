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

VAL_IMAGES_ARRAY = "tiledb://gskoumas/val_ship_images"

BATCH_SIZE = 64
IMAGE_SHAPE = (128, 128, 3)


def ask_model():

    if not os.path.exists('models'):
        os.makedirs('models')

    client = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    client.download_file('tiledb-gskoumas', 'airbus_ship_detection_tiledb/models/model.h5', 'models/model.h5')

    model = tf.keras.models.load_model('models/model.h5')

    val_image_array = tiledb.open(VAL_IMAGES_ARRAY)

    # Get data for prediction
    X = val_image_array[0:BATCH_SIZE]['rgb'].\
        view(np.uint8).reshape(BATCH_SIZE, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2])

    # Scale RGB
    X = X.astype(np.float32) / 255.0

    return model.predict(X)


tiledb.cloud.login(username=TILEDB_USER_NAME, password=TILEDB_PASSWD)

predictions = tiledb.cloud.udf.exec(ask_model)

print(tiledb.cloud.last_udf_task().logs)