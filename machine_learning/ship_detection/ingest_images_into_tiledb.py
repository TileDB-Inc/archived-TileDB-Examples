import boto3
import pandas as pd
import numpy as np
import tiledb
import cv2
import time

s3 = boto3.client('s3')

bucket = 'tiledb-gskoumas'
prefix = 'airbus_ship_detection/train_v2'
BATCH_SIZE = 32


def chunks(lst, n=BATCH_SIZE):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n], (i, i + n)


print('Loading training segments csv to df...')
df = pd.read_csv('train_ship_segmentations_v2.csv', header=None)

image_ids = df[0][1:].to_list()

print('Creating TileDB array for training segments...')
tiledb.stats_enable()

ctx = tiledb.Ctx()

dom = tiledb.Domain(
        tiledb.Dim(name="image_id", domain=(0, len(image_ids) - 1), tile=BATCH_SIZE, dtype=np.int32),
        tiledb.Dim(name="x_axis", domain=(0, 768 - 1), tile=768, dtype=np.int32),
        tiledb.Dim(name="y_axis", domain=(0, 768 - 1), tile=768, dtype=np.int32),
        ctx=ctx,
    )

attrs = [
        tiledb.Attr(name="rgb", dtype=[("", np.float32), ("", np.float32), ("", np.float32)], var=False, ctx=ctx),
    ]

schema = tiledb.ArraySchema(domain=dom,
                            sparse=False,
                            attrs=attrs,
                            cell_order='row-major',
                            tile_order='row-major',
                            capacity=10000,
                            ctx=ctx)

array = "s3://tiledb-gskoumas/airbus_ship_detection_tiledb/train_ship_images"

tiledb.Array.create(array, schema)

image_chunks = chunks(image_ids)

number_of_chunks = len(image_ids) // BATCH_SIZE

with tiledb.open(array, 'w', ctx=ctx) as train_images_tiledb:
    counter = 1
    for chunk, tpl in image_chunks:
        print('Working on chunk ' + str(counter) + ' of ' + str(number_of_chunks))
        image_chunk = []
        for image_id in chunk:
            image_path = 'train_v2/' + image_id
            image = cv2.imread(image_path)
            image_chunk.append(image.astype(np.float32))

        print('Inserting chunk ' + str(counter) + ' of ' + str(number_of_chunks))
        image_chunk = np.stack(image_chunk, axis=0)
        view = image_chunk.view([("", np.float32), ("", np.float32), ("", np.float32)])
        start = time.time()
        train_images_tiledb[tpl[0]:tpl[1], :, :] = view
        life_taken = time.time() - start
        print('Insertion took ' + str(life_taken) + 'seconds.')
        counter += 1
        del image_chunk

print('done')