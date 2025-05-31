import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from skimage.color import rgb2lab

def load_all_images_lab(directory, target_size):
    images_lab = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(directory, filename)
            img = load_img(path, target_size=(target_size, target_size))
            img_array = img_to_array(img) / 255.0
            lab_img = rgb2lab(img_array)
            images_lab.append(lab_img)
    return images_lab

def extract_patches(lab_img, patch_size):
    h, w, _ = lab_img.shape
    max_y = h - patch_size
    max_x = w - patch_size
    y = np.random.randint(0, max_y + 1)
    x = np.random.randint(0, max_x + 1)

    L_patch = lab_img[y:y+patch_size, x:x+patch_size, 0] / 100.0
    ab_patch = lab_img[y:y+patch_size, x:x+patch_size, 1:] / 128.0
    L_patch = np.expand_dims(L_patch, axis=-1)
    return L_patch.astype(np.float32), ab_patch.astype(np.float32)

def build_patch_dataset(images_lab, patch_size, batch_size, buffer_size=1000):
    def gen():
        while True:
            for img in images_lab:
                L, ab = extract_patches(img, patch_size)
                yield L, ab

    output_signature = (
        tf.TensorSpec(shape=(patch_size, patch_size, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(patch_size, patch_size, 2), dtype=tf.float32)
    )

    dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    return dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
