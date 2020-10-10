import os
import cv2
import numpy as np
import pickle

def prepare_datasets():
  datasets_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/StanfordDogsDataset"

  images = []
  labels = []
  label_names = {}

  label = 0

  for dog_directory in os.listdir(os.path.join(datasets_path, "Images")):
    if dog_directory[0] != ".":
      for image_name in os.listdir(os.path.join(datasets_path, "Images", dog_directory)):
        if image_name[0] != ".":
          image = cv2.imread(os.path.join(datasets_path, "Images", dog_directory, image_name))
          image = cv2.resize(image, (50, 50))

          images.append(image)
          labels.append(label)

      label_names[label] = dog_directory[10:].lower()
      label += 1
      print(dog_directory[10:].lower())

  images = np.array(images)
  images = np.moveaxis(images, -1, 1)
  labels = np.array(labels)

  with open(os.path.join(datasets_path, "images.pkl"), "wb") as f:
    pickle.dump(images, f)

  with open(os.path.join(datasets_path, "labels.pkl"), "wb") as f:
    pickle.dump(labels, f)

  with open(os.path.join(datasets_path, "label_names.pkl"), "wb") as f:
    pickle.dump(label_names, f)

def load_images():
  print("Loading Images...")
  datasets_path = "/home/shouki/Desktop/Programming/Python/AI/Datasets/ImageData/StanfordDogsDataset"
  with open(os.path.join(datasets_path, "images.pkl"), "rb") as f:
    images = pickle.load(f)

  with open(os.path.join(datasets_path, "labels.pkl"), "rb") as f:
    labels = pickle.load(f)

  with open(os.path.join(datasets_path, "label_names.pkl"), "rb") as f:
    label_names = pickle.load(f)

  indices = np.arange(0, images.shape[0])
  np.random.shuffle(indices)

  train_size = np.int(indices.shape[0] * 0.8)
  x_train = images[indices[:train_size]]
  y_train = labels[indices[:train_size]]

  x_test = images[indices[train_size:]]
  y_test= labels[indices[train_size:]]

  return x_train, y_train, x_test, y_test
