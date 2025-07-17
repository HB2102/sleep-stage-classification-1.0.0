import tensorflow as tf
import os
import json

def load_label_map(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

def get_image_label_paths(image_dir, label_map):
    paths = []
    for fname, label in label_map.items():
        img_path = os.path.join(image_dir, fname)
        paths.append((img_path, label))
    return paths

def parse_function(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [60, 76])  
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.cast(label, tf.int32)

def create_dataset(image_dir, label_file, batch_size=32, shuffle=True):
    label_map = load_label_map(label_file)
    image_label_paths = get_image_label_paths(image_dir, label_map)

    image_paths, labels = zip(*image_label_paths)
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels))

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)