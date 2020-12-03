import boto3
import pandas as pd
import numpy as np
import tiledb
from PIL import Image

s3 = boto3.client('s3')

bucket = 'tiledb-gskoumas'
prefix = 'airbus_ship_detection/train_v2'


def read_image_from_s3(bucket, key):
    """Load image file from s3.

    Parameters
    ----------
    bucket: string
        Bucket name
    key : string
        Path in s3

    Returns
    -------
    np array
        Image array
    """
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    return np.array(im)


def chunks(lst, n=1024):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n], (i, i + n)


print('Loading training segments csv to df...')
df = pd.read_csv('train_ship_segmentations_v2.csv', header=None)

image_ids = df[0][1:].to_list()

print('Creating TileDB array for training segments...')
ctx = tiledb.Ctx()

dom = tiledb.Domain(
        tiledb.Dim(name="image_id", domain=(0, len(image_ids) - 1), tile=1024, dtype=np.int32),
        tiledb.Dim(name="x_axis", domain=(0, 768 - 1), tile=768, dtype=np.int32),
        tiledb.Dim(name="y_axis", domain=(0, 768 - 1), tile=768, dtype=np.int32),
        ctx=ctx,
    )

attrs = [
        tiledb.Attr(name="rgb", dtype=[("", np.float32), ("", np.float32), ("", np.float32)], var=False, ctx=ctx),
    ]

schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs, ctx=ctx)

array = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/train_ship_images"

tiledb.Array.create(array, schema)

image_chunks = chunks(image_ids)

number_of_chunks = len(image_ids) // 1024

with tiledb.open(array, 'w') as train_images_tiledb:
    counter = 1
    for chunk, tpl in image_chunks:
        print('Working on chunk ' + str(counter) + 'of' + str(number_of_chunks))
        image_chunk = []
        for image_id in chunk:
            key = prefix + '/' + image_id
            image_chunk.append(read_image_from_s3(bucket=bucket, key=key))

        print('Inserting chunk ' + str(counter) + 'of' + str(number_of_chunks))
        train_images_tiledb[tpl[0]:tpl[1]] = np.stack(image_chunk, axis=0).view([("", np.float32), ("", np.float32), ("", np.float32)])

print('done')