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



def get_sorted_image_label_pairs(image_dir, label_map):
    image_label_pairs = [
        (os.path.join(image_dir, fname), label)
        for fname, label in label_map.items()
    ]
    # مرتب‌سازی بر اساس نام فایل
    image_label_pairs.sort(key=lambda x: x[0])
    return image_label_pairs

def create_sequence_dataset(image_dir, label_file, sequence_length=10, batch_size=2, shuffle=True):
    label_map = load_label_map(label_file)
    pairs = get_sorted_image_label_pairs(image_dir, label_map)

    # ساخت sequence‌ها با sliding window
    sequences = []
    for i in range(len(pairs) - sequence_length + 1):
        image_seq = [pairs[j][0] for j in range(i, i + sequence_length)]
        label_seq = [pairs[j][1] for j in range(i, i + sequence_length)]
        sequences.append((image_seq, label_seq))

    def load_sequence(images, labels):
        def _load_image(file_path):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.resize(image, [60, 76])
            image = tf.cast(image, tf.float32) / 255.0
            return image

        images = tf.map_fn(_load_image, images, fn_output_signature=tf.float32)
        labels = tf.cast(labels, tf.int32)
        return images, labels

    def generator():
        for image_paths, label_seq in sequences:
            yield image_paths, label_seq

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(sequence_length,), dtype=tf.string),
            tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
        )
    )

    dataset = dataset.map(load_sequence, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(sequences))

    return dataset.map(
            lambda x, y: (x, {'main_output': y, 'aux_output': y})
        ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
