# Author: Raphael Delhome, Oslandia

# Mapillary data set reading

import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import tensorflow as tf

DATASET = ["training", "validation", "testing"]
IMG_SIZE = (3264, 2448)
IMAGE_TYPES = ["images", "instances", "labels"]
TRAINING_IMAGE_PATH = os.path.join("data", "training", "images")
TRAINING_LABEL_PATH = os.path.join("data", "training", "labels")
TRAINING_INPUT_PATH = os.path.join("data", "training", "input")
TRAINING_OUTPUT_PATH = os.path.join("data", "training", "output")
VALIDATION_IMAGE_PATH = os.path.join("data", "validation", "images")
VALIDATION_LABEL_PATH = os.path.join("data", "validation", "labels")
VALIDATION_INPUT_PATH = os.path.join("data", "validation", "input")
VALIDATION_OUTPUT_PATH = os.path.join("data", "validation", "output")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass

def size_inventory():
    sizes = []
    widths = []
    heights = []
    types = []
    filenames = []
    datasets = []
    for dataset in DATASET:
        for img_filename in os.listdir(os.path.join("data", dataset, "images")):
            for img_type in IMAGE_TYPES:
                if dataset == "testing" and not img_type == "images":
                    continue
                complete_filename = os.path.join("data", dataset,
                                                 img_type, img_filename)
                if not img_type == "images":
                    complete_filename = complete_filename.replace("images", img_type)
                    complete_filename = complete_filename.replace(".jpg", ".png")
                image = Image.open(complete_filename)
                datasets.append(dataset)
                types.append(img_type)
                filenames.append(complete_filename.split("/")[-1].split(".")[0])
                sizes.append(image.size)
                widths.append(image.size[0])
                heights.append(image.size[1])
    return pd.DataFrame({"dataset": datasets,
                      "filename": filenames,
                      "img_type": types,
                      "size": sizes,
                      "width": widths,
                      "height": heights})

def mapillary_label_building(filtered_image, nb_labels):
    filtered_data = np.array(filtered_image)
    avlble_labels = (pd.Series(filtered_data.reshape([-1]))
                     .value_counts()
                     .index)
    return [1 if i in avlble_labels else 0 for i in range(nb_labels)]

def mapillary_data_preparation(dataset="training", nb_labels=1):
    IMAGE_PATH = os.path.join("data", dataset, "images")
    INPUT_PATH = os.path.join("data", dataset, "input")
    make_dir(INPUT_PATH)
    if dataset != "testing":
        LABEL_PATH = os.path.join("data", dataset, "labels")
        OUTPUT_PATH = os.path.join("data", dataset, "output")
        make_dir(OUTPUT_PATH)
        train_y = []
    for img_id, img_filename in enumerate(os.listdir(IMAGE_PATH)):
        img_in = Image.open(os.path.join(IMAGE_PATH, img_filename))
        new_img_name = "{:05d}.jpg".format(img_id)
        instance_name = os.path.join(INPUT_PATH, new_img_name)
        img_in.save(instance_name)
        logger.warning("""[{} set] Image {} saved as {}..."""
                       .format(dataset, img_filename, new_img_name))
        if dataset != "testing":
            label_filename = img_filename.replace(".jpg", ".png")
            img_out = Image.open(os.path.join(LABEL_PATH, label_filename))
            img_out = img_out.resize(IMG_SIZE, Image.NEAREST)
            y = mapillary_label_building(img_out, nb_labels)
            old_width, old_height = img_in.size
            img_in = img_in.resize(IMG_SIZE)
            width_ratio = IMG_SIZE[0] / old_width
            height_ratio = IMG_SIZE[1] / old_height
            y.insert(0, height_ratio)
            y.insert(0, old_height)
            y.insert(0, width_ratio)
            y.insert(0, old_width)
            y.insert(0, new_img_name)
            y.insert(0, img_filename)
            train_y.append(y)
    if dataset != "testing":
        train_y = pd.DataFrame(train_y, columns=["old_name", "new_name",
                                                 "old_width", "width_ratio",
                                                 "old_height", "height_ratio"]
                               + ["label_" + str(i) for i in range(nb_labels)])
        train_y.to_csv(os.path.join(OUTPUT_PATH, "labels.csv"), index=False)

if __name__ == "__main__":

    ##################
    # Transform the image into input data
    # Input data: np.array of shape [2448, 3264, 3]
    # 18000, 2000 and 5000 images respectively for training, validation and
    # testing set
    ##################
    # Transform the labelled image into output data
    # Output data: an array of 66 0-1 values, 1 if the i-th Mapillary object is
    # on the image, 0 otherwise
    # np.array of shape [18000, 66] for training set, and [2000, 66] for
    # validation set
        
    # read in config file
    with open('data/config.json') as config_file:
        config = json.load(config_file)
    labels = config['labels']

    mapillary_data_preparation("training", len(labels))
    mapillary_data_preparation("validation", len(labels))
    mapillary_data_preparation("testing", len(labels))
