## Image Classification Notebook

This example notebook presents how we can employ TileDB in order to store images (of the same size) as TileDB arrays, 
and then use them to train an image classification model with Tensorflow and Keras. You will see the following.

1. How to get an image dataset and store it as TileDB arrays (images and labels).
2. How to create a basic Python data generator in order to read image batches from TileDB arrays.
3. How to pass the created TileDB data generators to Keras and train an image classification model.