from sklearn.preprocessing import MultiLabelBinarizer
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 32

img_height = 3072
img_width = 3072

REDUCE = 8
input_height = img_height // REDUCE
input_width = img_width // REDUCE

LABELS = [
    'Nucleoplasm',
    'Nuclear membrane',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nuclear speckles',
    'Nuclear bodies',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    'Intermediate filaments',
    'Actin filaments',
    'Microtubules',
    'Mitotic spindle',
    'Centrosome',
    'Plasma membrane',
    'Mitochondria',
    'Aggresome',
    'Cytosol',
    'Vesicles and punctate cytosolic patterns',
    'Negative',
]
LABELS_LEN = len(LABELS)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=1)
    # resize the image to the desired size
    img = tf.image.resize_with_crop_or_pad(img, img_height, img_width)
    img = tf.image.resize(img, (input_height, input_width))
    img = img / 255.
    return img


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=10)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def concatenate_arrays(a1, a2, a3, a4):
    img = tf.concat([a1, a2, a3, a4], 2)
    return img


red_list_ds = tf.data.Dataset.list_files(os.path.join('train', '*_red.png'), shuffle=False)
green_list_ds = tf.data.Dataset.list_files(os.path.join('train', '*_green.png'), shuffle=False)
blue_list_ds = tf.data.Dataset.list_files(os.path.join('train', '*_blue.png'), shuffle=False)
yellow_list_ds = tf.data.Dataset.list_files(os.path.join('train', '*_yellow.png'), shuffle=False)
red_list_ds = red_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
green_list_ds = green_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
blue_list_ds = blue_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
yellow_list_ds = yellow_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

train_x_ds = tf.data.Dataset.zip((red_list_ds, green_list_ds, blue_list_ds, yellow_list_ds))
train_x_ds = train_x_ds.map(concatenate_arrays, num_parallel_calls=AUTOTUNE)

# for item in train_x_ds:
#     print(item)

train_y_data = pd.read_csv('train.csv')
train_y_data = train_y_data.sort_values('ID')


def test_str(labels, total_classes):
    binary_class = np.zeros(total_classes, dtype=np.uint8)
    labels = labels.split('|')
    return [int(item) for item in labels]


train_y_data = list(map(lambda x, y: test_str(x, y), train_y_data['Label'], [
                    LABELS_LEN for i in range(len(train_y_data))]))

mlb = MultiLabelBinarizer()
mlb.fit(train_y_data)
train_y_data = mlb.transform(train_y_data)
train_y_ds = tf.data.Dataset.from_tensor_slices(np.array(train_y_data, dtype=np.uint8))

# for item in train_y_ds:
# print(item)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(input_height, input_width, 4,)),
    # tf.keras.layers.Convolution2D(512, 3),
    # tf.keras.layers.MaxPool2D(),
    # tf.keras.layers.Convolution2D(256, 3),
    # tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Convolution2D(128, 3),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Convolution2D(64, 3),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(LABELS_LEN)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

train_ds = tf.data.Dataset.zip((train_x_ds, train_y_ds))
train_ds = configure_for_performance(train_ds)

model.fit(train_ds, epochs=5)
# model.evaluate(x_test, y_test)
