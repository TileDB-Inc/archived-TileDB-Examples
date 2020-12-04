import boto3
import pandas as pd
import numpy as np
import tiledb

s3 = boto3.client('s3')

BATCH_SIZE = 32

bucket = 'tiledb-gskoumas'
prefix = 'airbus_ship_detection/train_v2'
s3_file = 'airbus_ship_detection/train_ship_segmentations_v2.csv'
to_be_downloaded_file = 'train_ship_segmentations_v2.csv'

print('Downloading training segments csv...')
s3.download_file(bucket, s3_file, to_be_downloaded_file)

print('Loading training segments csv to df...')
df = pd.read_csv(to_be_downloaded_file, header=None)

train_ship_segments = []

print('Creating training segment numpy arrays...')
for item in df[1][1:]:
    if isinstance(item, str):
        numbers = item.split(' ')
        numbers = np.array(numbers).astype(dtype=np.int32)
        train_ship_segments.append(numbers)
    else:
        train_ship_segments.append(np.array([0]).astype(dtype=np.int32))


train_ship_segments = np.array(train_ship_segments).reshape(len(train_ship_segments), 1)

print('Creating TileDB array for training segments...')
ctx = tiledb.Ctx()

dom = tiledb.Domain(
        tiledb.Dim(name="image_id", domain=(0, train_ship_segments.shape[0] - 1), tile=BATCH_SIZE, dtype=np.int32),
        ctx=ctx,
    )

attrs = [
        tiledb.Attr(name="segments", var=True, dtype=np.int32, ctx=ctx),
    ]

schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=attrs, ctx=ctx)

array = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/train_ship_segments"

tiledb.Array.create(array, schema)

data_dict = {'segments': train_ship_segments}

print('Injecting training segments to TileDB...')
with tiledb.open(array, 'w', ctx=ctx) as array:
    array[:] = data_dict

print('Done with segments!')