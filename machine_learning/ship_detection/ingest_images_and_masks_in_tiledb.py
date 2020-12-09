import tiledb
import pandas as pd
import numpy as np
import os
import boto3

s3 = boto3.client('s3')

bucket = 'tiledb-gskoumas'
prefix = 'airbus_ship_detection/train_v2'
BATCH_SIZE = 32


def chunks(lst, n=BATCH_SIZE):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n], (i, i + n)


print('Loading training segments csv to df...')
df = pd.read_csv('train_ship_segmentations_v2.csv', usecols=['ImageId', 'EncodedPixels'])

not_empty_boolean = pd.notna(df['EncodedPixels'])
unique_non_empty_images = df[not_empty_boolean].ImageId.nunique()
empty_images = (~not_empty_boolean).sum()

df['has_ship'] = df['EncodedPixels'].map(lambda x: True if type(x) is str else False)

grouped_df = df.groupby('has_ship')

df = pd.concat([grouped_df.get_group(True), grouped_df.get_group(False).sample(n=unique_non_empty_images)])

del grouped_df

# Drop not needed column
df = df.drop('has_ship', axis=1)

# Replace nans with empty strings
df = df.replace(np.nan, '', regex=True)

# Group by image ids, as some images contain more than one masks
print("Grouping by image id...")
df = df.groupby('ImageId')['EncodedPixels'].apply(np.array)

df = pd.DataFrame(df).reset_index()
df.columns = ['ImageId', 'EncodedPixels']

# # Split in train and validation
print('Split into train and validation')
split_msk = np.random.rand(len(df)) < 0.9

train_df = df[split_msk]
val_df = df[~split_msk]

del df

train_segs = train_df['EncodedPixels'].to_numpy()
val_segs = val_df['EncodedPixels'].to_numpy()

train_segs = train_segs.reshape(len(train_df), 1)
val_segs = val_segs.reshape(len(val_df), 1)


# Ingestion

ctx = tiledb.Ctx()

print('Creating training segments TileDB array...')
dom_segs_train = tiledb.Domain(
    tiledb.Dim(name="image_id", domain=(0, len(train_df) - 1), tile=BATCH_SIZE, dtype=np.int32),
        ctx=ctx,
    )

attrs = [
        tiledb.Attr(name="segments", var=True, dtype="U", ctx=ctx),
    ]

schema = tiledb.ArraySchema(domain=dom_segs_train, sparse=False, attrs=attrs, ctx=ctx)

array = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/train_ship_segments"

tiledb.Array.create(array, schema)

data_dict = {'segments': train_segs}

print('Injecting training segments to TileDB...')
with tiledb.open(array, 'w', ctx=ctx) as array:
    array[:] = data_dict

print('Creating validation segments TileDB array...')
dom_segs_val = tiledb.Domain(
    tiledb.Dim(name="image_id", domain=(0, len(val_df) - 1), tile=BATCH_SIZE, dtype=np.int32),
        ctx=ctx,
    )

schema = tiledb.ArraySchema(domain=dom_segs_val, sparse=False, attrs=attrs, ctx=ctx)

array = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/val_ship_segments"

tiledb.Array.create(array, schema)

data_dict = {'segments': val_segs}

print('Injecting validation segments to TileDB...')
with tiledb.open(array, 'w', ctx=ctx) as array:
    array[:] = data_dict

print('Done with segments!')
