from PIL                 import Image
import os
import numpy             as np
import matplotlib.pyplot as plt

num_training = 200
num_validation = 150
num_testing = 150
filepath = "../data/mnist_digit_"

def make_dataset(digit_num):
  dataset = []
  with open(filepath + digit_num + ".csv", 'r') as f:
    for line in f:
      dataset.append(line.split(" "))

  training = np.array(dataset[:num_training])
  validation = np.array(dataset[num_training:(num_training + num_validation)])
  testing = np.array(dataset[(num_training + num_validation):(num_training + num_validation + num_testing)])

  return training, validation, testing

def make_all_datasets():
  all_datasets = {}

  for i in range(10):
    num = str(i)
    training, validation, testing = make_dataset(num)
    all_datasets[num] = [training, validation, testing]

    assert len(training) == num_training
    assert len(validation) == num_validation
    assert len(testing) == num_testing

  return all_datasets

def normalize_datasets(all_datasets):
  normalized_dict = {}

  for key in all_datasets.keys():
    normalized_dict[key] = []

    for number_datasets in all_datasets[key]:
      normalized_number_datasets = []

      for image in number_datasets:
        # normalized_image = [ 2 * float(elem) / 255. - 1 for elem in image ]
        normalized_image = [ float(elem) for elem in image ]
        normalized_number_datasets.append(normalized_image)

      normalized_dict[key].append(normalized_number_datasets)

  return normalized_dict

# all_datasets = make_all_datasets()
# normalized_dataset = normalize_datasets(all_datasets)

# for list_of_datasets in normalized_dataset.values():
#   for dataset in list_of_datasets:
#     for image in dataset:
#       for elem in image:
#         if elem > 1 or elem < -1:
#           raise Exception(elem)


# format of dictionary:
# normalized = {
# 1: [training, validation, testing],
# 2: [ [image, image, image, ...], ...],
# 3: [ [ [1,2,3,4,5,...], [1,2,3,4,5,...], ...], ...]
# ...
# }

