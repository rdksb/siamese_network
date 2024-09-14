import os
import urllib.request
import zipfile
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from skimage import transform
from skimage.util import random_noise

# size of input images
WIDTH = 105
HEIGHT = 105

def omniglot_download(root_dir):
    """Download Omniglot dataset (50 alphabets)"""
    urls = ['https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
            'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip']
    for url in urls:
        file_name = url.rpartition('/')[2]
        file_path = os.path.join(root_dir, file_name)
        #folder_path = os.path.join(root_dir, file_name.rpartition('.')[0])
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        request_url = urllib.request.urlopen(url)
        with open(file_path, 'wb') as f:
            f.write(request_url.read())
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(root_dir)
        zip_ref.close()


class DataGenerator(tf.keras.utils.PyDataset):
    """Return a generator that produces batches of input data"""
    def __init__(self, x, y, batch_size, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Return number of batches per epoch"""
        return int(np.floor(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        """Generate a batch of data at position idx"""
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.y))
        batch_x = self.x[low:high]
        batch_y = self.y[low:high]

        batch_im1, batch_im2 = [], []
        for pair_idx in range(len(batch_x)):
            if np.random.rand() > 0.5:
                batch_im1.append(affine_transform(path_to_image(batch_x[pair_idx, 0])))
                batch_im2.append(affine_transform(path_to_image(batch_x[pair_idx, 1])))
            else:
                batch_im1.append(path_to_image(batch_x[pair_idx, 0]))
                batch_im2.append(path_to_image(batch_x[pair_idx, 1]))
        return (np.array(batch_im1), np.array(batch_im2)), np.array(batch_y)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)


def affine_transform(image):
    """Apply an affine transformation with randomized parameters"""
    angle= np.random.randint(-10, 10)*np.pi/180
    shear = np.random.uniform(-0.3, 0.3, size=2)
    scale = np.random.uniform(0.8, 1.2, size=2)
    translation = np.random.uniform(-2, 2, size=2)
    atrans = transform.AffineTransform(scale=scale, rotation=angle,
                                shear=shear, translation=translation)
    # ‘constant’, ‘edge’, ‘symmetric’, ‘reflect’, ‘wrap’
    warp_image = transform.warp(image, atrans, mode="constant")

    return warp_image


def path_to_image(path):
    """Read an image from a png file, decode it into a tensor,
    then invert and normalise the pixel values"""
    image = tf.io.decode_png(tf.io.read_file(path))
    image = tf.bitwise.invert(image)
    image = tf.cast(image, tf.float64)
    image = image/255
    return np.array(image)


def get_paths_to_image_files(training=True, root_dir='omniglot_data'):
    """Get paths to image files"""
    if training:
        img_dir = os.path.join(root_dir, 'images_background')
    else:
        img_dir = os.path.join(root_dir, 'images_evaluation')

    alphabets = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f[0] != '.']
    characters = [os.path.join(a, f) for a in alphabets for f in os.listdir(a)]
    paths = [[os.path.join(c, f) for f in os.listdir(c)] for c in characters]
    return paths


def generateImagePairs(root_dir='omniglot_data', n_samples=0, out_path=None):
    """Generate image pairs and their labels for training purpose"""
    c_im_files = get_paths_to_image_files(training=True, root_dir=root_dir)
    im_pairs = []
    # sampling positive pairs (same class image pairs)
    for c in c_im_files: # all similar pairs for each class
        im_pairs.extend(itertools.combinations(c, 2))
    if n_samples == 0:
        n_samples = len(im_pairs)
    else:
        #subsampling
        im_pairs = random.sample(im_pairs, n_samples)
    #add label=1 for each pair
    im_pairs = [p + (1,) for p in im_pairs]

    # sampling negative pairs (different class image pairs)
    for _ in range(n_samples):
        c1, c2 = random.sample(c_im_files, 2)
        im_pairs.append((random.choice(c1), random.choice(c2), 0))

    # shuffle pos & neg exemples
    random.shuffle(im_pairs)
    # save in file or return data
    if out_path != None:
        with open(os.path.join(out_path, "training_image_paths.pickle"), 'wb') as f:
            pickle.dump(im_pairs, f)
    else:
        return im_pairs


def generateNway1shotTestingData(root_dir='omniglot_data', n_way=2):
    """Generate image pairs and their labels for testing purpose"""
    c_im_files = get_paths_to_image_files(training=False, root_dir=root_dir)
    n_classes = len(c_im_files)

    pairs = []
    # sampling n classes
    indexes = random.sample(range(n_classes), n_way)

    pos_images = random.sample(c_im_files[indexes[0]], 2)
    pairs.append((pos_images[0], pos_images[1], 1))

    for i in indexes[1:]:
        neg_image = random.choice(c_im_files[i])
        pairs.append((pos_images[0], neg_image, 0))

    random.shuffle(pairs)

    p1, p2, labels = map(np.array, zip(*pairs))
    im_pair1 = list(map(path_to_image, p1))
    im_pair2 = list(map(path_to_image, p2))

    return np.array(im_pair1), np.array(im_pair2), np.array(labels)


def generateTrainingdata(n_samples=0, data_file=None):
    """Load image pairs and their labels for training"""
    if data_file == None:
        data = generateImagePairs('omniglot_data', n_samples)
    else:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    p1, p2, labels = map(np.array, zip(*data))
    path_pairs = np.stack((p1, p2), axis=1)

    return path_pairs, labels
