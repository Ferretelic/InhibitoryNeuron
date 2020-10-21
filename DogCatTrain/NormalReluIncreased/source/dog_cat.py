import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
import pyprind

def prepare_datasets(image_size):
  print("Preparing Datasets")
  datasets_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/DogCat"
  train_directory = os.path.join(datasets_path, "train")
  test_directory = os.path.join(datasets_path, "test")

  train_images = []
  train_labels = []
  test_images = []
  test_labels = []

  datasets = {
    "train": {
      "directory": train_directory,
      "images": train_images,
      "labels": train_labels
    },
    "test": {
      "directory": test_directory,
      "images": test_images,
      "labels": test_labels
    }
  }

  label_data = {0: "dog", 1: "cat"}

  for data_type in ["train", "test"]:
      for category in os.listdir(datasets[data_type]["directory"]):
        for image_name in os.listdir(os.path.join(datasets[data_type]["directory"], category)):
          if image_name[0] != ".":
            image = cv2.imread(os.path.join(datasets[data_type]["directory"], category, image_name))
            image = cv2.resize(image, image_size)
            image = np.moveaxis(image, -1, 0)
            datasets[data_type]["images"].append(image)

            if category == "dogs":
              datasets[data_type]["labels"].append(0)
            else:
              datasets[data_type]["labels"].append(1)

  x_train = np.array(datasets["train"]["images"], dtype=np.float32)
  y_train = np.array(datasets["train"]["labels"], dtype=np.int8)
  x_test = np.array(datasets["test"]["images"], dtype=np.float32)
  y_test = np.array(datasets["test"]["labels"], dtype=np.int8)

  with open(os.path.join(datasets_path, "x_train.pkl"), "wb") as f:
    pickle.dump(x_train, f)

  with open(os.path.join(datasets_path, "y_train.pkl"), "wb") as f:
    pickle.dump(y_train, f)

  with open(os.path.join(datasets_path, "x_test.pkl"), "wb") as f:
    pickle.dump(x_test, f)

  with open(os.path.join(datasets_path, "y_test.pkl"), "wb") as f:
    pickle.dump(y_test, f)

def load_datasets():
  print("Loading Datasets")
  datasets_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/DogCat"

  with open(os.path.join(datasets_path, "x_train.pkl"), "rb") as f:
    x_train = pickle.load(f)

  with open(os.path.join(datasets_path, "y_train.pkl"), "rb") as f:
    y_train = pickle.load(f)

  with open(os.path.join(datasets_path, "x_test.pkl"), "rb") as f:
    x_test = pickle.load(f)

  with open(os.path.join(datasets_path, "y_test.pkl"), "rb") as f:
    y_test = pickle.load(f)

  train_index = np.arange(0, y_train.shape[0])
  np.random.shuffle(train_index)
  x_train = x_train[train_index]
  y_train = y_train[train_index]

  test_index = np.arange(0, y_test.shape[0])
  np.random.shuffle(test_index)
  x_test = x_test[test_index]
  y_test = y_test[test_index]

  return x_train, y_train, x_test, y_test