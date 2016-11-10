import sys
import random
import sklearn
import numpy                   as np
import pylab                   as pl
import matplotlib.pyplot       as plt
from sklearn import neural_network

##### Softmax functions #####
def softmax_fn(z):
  output = []
  denominator_sum = 0.
  for z_j in z:
    denominator_sum += np.exp(z_j)

  for z_i in z:
    output.append(np.exp(z_i) / denominator_sum)
  return np.array(output)


def softmax_derivative_fn(z, f_index, derivative_index):
  ez_sum = 0.

  for z_i in z:
    ez_sum += np.exp(z_i)
  denominator = ez_sum ** 2.

  if derivative_index == f_index:
    numerator = np.exp(z[derivative_index]) * (ez_sum - np.exp(z[derivative_index]))
  else:
    numerator = np.exp(z[derivative_index]) * np.exp(z[f_index])

  return numerator / denominator


def softmax_gradient_matrix(z):
  array = []

  for i in range(len(z)):
    subarray = []
    for j in range(len(z)):
      gradient = softmax_derivative_fn(z, i, j)
      subarray.append(gradient)
    array.append(subarray)

  return np.array(array)


##### ReLu functions #####
def relu_fn(z):
  output = []
  for z_i in z:
    output.append(max(0., z_i))
  return np.array(output)


def relu_derivative_fn(z):
  derivative = []

  for z_i in z:
    if z_i > 0:
      derivative.append(1)
    else:
      derivative.append(0)

  return np.array(derivative)

############################################


def feedforward(x, theta, num_layers):
  alphas = [x]
  z = []

  for layer_index in range(num_layers+1):
    w_l = theta[2 * (layer_index)]
    b_l = theta[2 * (layer_index) + 1]

    if layer_index < num_layers:
      fn = relu_fn
    else:
      fn = softmax_fn

    prev_alpha = alphas[layer_index]
    z_l = np.dot(np.transpose(w_l), prev_alpha) + b_l
    a_l = fn(z_l)

    alphas.append(a_l)
    z.append(z_l)

  return alphas, z


def loss_gradient_matrix(y, alphas):
  loss_gradient_array = []

  for index in range(len(alphas)):
    if int(y) == index:
      loss_gradient_array.append(-y / alphas[index])
    else:
      loss_gradient_array.append(0)

  return np.array(loss_gradient_array)


def calculate_output_error(y, z, alphas):
  loss_gradient = loss_gradient_matrix(y, alphas[-1])
  softmax_matrix = softmax_gradient_matrix(z[-1])

  return np.dot(loss_gradient, softmax_matrix)


def backprop(num_layers, output_error, theta, z):
  deltas = [output_error]

  for layer_index in reversed(range(num_layers)):
    # Building up list of deltas, so next delta (previously calculated) is at the 0th index
    next_delta = deltas[0]
    next_w = theta[2*(layer_index+1)]

    derivative = relu_derivative_fn(z[layer_index])

    delta = np.dot(np.diag(derivative), np.dot(next_w, next_delta))
    deltas.insert(0, delta)

  return deltas


def calculate_gradient(deltas, alphas, num_layers):
  gradient = []

  for layer_index in range(num_layers+1):
    # Alphas starts with a_0, deltas starts with delta_1
    weight_gradient = np.dot(np.transpose([alphas[layer_index]]), [deltas[layer_index]])
    bias_gradient = deltas[layer_index]

    gradient.append(weight_gradient)
    gradient.append(bias_gradient)

  return gradient


def calculate_overall_gradient(x, y, theta, num_layers):
  # print "Feedforward..."
  alphas, z = feedforward(x, theta, num_layers)
  # print "Calculating output error..."
  output_error = calculate_output_error(y, z, alphas)
  # print "Backprop..."
  deltas = backprop(num_layers, output_error, theta, z)
  # print "Calculating gradient..."
  gradient = calculate_gradient(deltas, alphas, num_layers)
  return gradient, output_error

##### SGD #####

def calculate_next_theta(old_theta, x, y, t, num_layers):
  t0 = 10
  k = 0.75
  n = lambda t: (t0 + t)**(-k)
  print "t: ", t, "  n: ", n(t)
  gradient, output_error = calculate_overall_gradient(x, y, old_theta, num_layers)
  print "gradient: ", gradient

  new_theta = []
  for old_elem, gradient_elem in zip(old_theta, gradient):
    step = np.dot(n(t), gradient_elem)
    print "step: ", step
    new_theta.append(old_elem - step)

  return new_theta, output_error


def sgd(x, y, theta, threshold, num_layers):
  number_of_samples = len(x)

  # differences = [False] * number_of_samples
  # old_jthetas = [0.0] * number_of_samples
  previous_values = []
  within_threshold = False
  t = 0

  while not within_threshold: # > threshold:  # t < 10 and not all(differences):
    i = t % number_of_samples
    theta, output_error = calculate_next_theta(theta, x[i], y[i], t, num_layers)
    print "theta: ", theta, "  output_error: ", output_error
    # new_jtheta = objective_f(x[i], y[i], theta)
    # difference = new_jtheta - old_jthetas[i]

    # if(abs(difference) < threshold):
    #   differences[i] = True

    previous_values.append(theta)
    # previous_values.append((theta, new_jtheta))
    # old_jthetas[i] = new_jtheta

    within_threshold = all([abs(elem) < threshold for elem in output_error])

    t += 1

  return previous_values

##### Predictions and testing #####

def predict(x, theta, num_layers):
  prev_alphas = x

  for layer_index in range(num_layers+1):
    w_l = theta[2 * layer_index]
    b_l = theta[2 * layer_index + 1]

    if layer_index < num_layers:
      fn = relu_fn
    else:
      fn = softmax_fn

    z = np.dot(np.transpose(w_l), prev_alphas) + b_l
    prev_alphas = fn(z)

  return np.argmax(prev_alphas)

def get_classification_accuracy(x, y, theta, num_layers):
  num_classified_correctly = 0

  for x_i, y_i in zip(x, y):
    prediction = predict(x_i, theta, num_layers)
    if int(y_i) == prediction:
      num_classified_correctly += 1

  return float(num_classified_correctly) / len(x)


def get_data(filename):
  data = pl.loadtxt(filename)
  x, y = [], []

  for line in data:
    x.append(line[:2])
    y.append(line[2])

  return x, y

def load_hw2_data(filenum):
  train = np.loadtxt('../hw2_data/data'+filenum+'_train.csv')
  x_training = train[:,0:2]
  y_training = train[:,2:3]

  validate = np.loadtxt('../hw2_data/data'+filenum+'_validate.csv')
  x_validate = validate[:,0:2]
  y_validate = validate[:,2:3]

  test = np.loadtxt('../hw2_data/data'+filenum+'_test.csv')
  x_testing = test[:,0:2]
  y_testing = test[:,2:3]

  x_training_validate, y_training_validate = [], []
  for data_x, data_y in zip(x_training, y_training):
    x_training_validate.append(data_x)
    y_training_validate.append(data_y)
  for data_x, data_y in zip(x_validate, y_validate):
    x_training_validate.append(data_x)
    y_training_validate.append(data_y)

  y_adjusted_training_validate = []
  for elem in y_training_validate:
    if elem < 0:
      y_adjusted_training_validate.append(0)
    else:
      y_adjusted_training_validate.append(1)
  y_adjusted_testing = []
  for elem in y_testing:
    if elem < 0:
      y_adjusted_testing.append(0)
    else:
      y_adjusted_testing.append(1)

  return x_training, y_training, x_training_validate, y_adjusted_training_validate, x_testing, y_adjusted_testing

def initialize_to_random(num):
  return [ random.uniform(0, 1) for i in range(num) ]


##### MAIN #####

if __name__ == '__main__':
  filenum = sys.argv[1]
  # filename = "../data/data_3class.csv"
  # x, y = get_data(filename)
  # size_of_training = 720
  # x_training, y_training = x[:size_of_training], y[:size_of_training]
  # x_validate, y_validate = x[size_of_training:], y[size_of_training:]
  # x_training, y_training = [[1,2], [2,3]], [0,1]

  x_training, y_training, x_training_validate, y_training_validate, x_testing, y_testing = load_hw2_data(filenum)


  num_layers = 1
  neurons_per_layer = 5
  threshold = 0.001

  # class_set = set(y)
  # num_classes = len(class_set)

  # W = np.array([ initialize_to_random(neurons_per_layer) for i in range(len(x[0])) ])
  # b = np.array(initialize_to_random(neurons_per_layer))
  # W_out = np.array([ initialize_to_random(num_classes) for i in range(neurons_per_layer) ])
  # b_out = np.array(initialize_to_random(num_classes))
  # theta = [W, b] * num_layers + [W_out, b_out]
  # print "theta: ", theta

  # previous_values = sgd(x_training, y_training, theta, threshold, num_layers)
  # opt_theta = previous_values[-1]

  # classification_accuracy = get_classification_accuracy(x_validate, y_validate, theta, num_layers)
  # print "validation accuracy: ", classification_accuracy
  # classification_accuracy = get_classification_accuracy(x_validate, y_validate, opt_theta, num_layers)
  # print "validation accuracy: ", classification_accuracy

  # architectures = [(5), (100), (5,5), (100, 100)]
  architectures = [(5), (5,5), (5,5,5), (5,5,5,5), (5,5,5,5,5)]
  learning_rates = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
  training_rates = {}
  testing_rates = {}
  num_repeats = 20
  testing_rates_list = []


  for learning_rate in learning_rates:
    total_training_score = 0
    total_testing_score = 0

    for i in range(num_repeats):
      classifier = neural_network.MLPClassifier((100), max_iter=500, early_stopping=True, solver='sgd', alpha=0, learning_rate_init=learning_rate, validation_fraction=1./3)
      classifier.fit(x_training_validate, y_training_validate)
      total_training_score += classifier.score(x_training_validate, y_training_validate)
      total_testing_score += classifier.score(x_testing, y_testing)

    testing_rates_list.append(total_testing_score / float(num_repeats))

    # training_rates[learning_rate] = total_training_score / float(num_repeats)
    # testing_rates[learning_rate] = total_testing_score / float(num_repeats)

  # print "average training: ", training_rates
  # print "average testing: ", testing_rates


  print "plotting..."
  plt.figure()
  plt.plot(learning_rates, testing_rates_list)
  plt.xscale('log')
  plt.title('Testing accuracy vs. learning rate, (100)', fontsize=16)
  plt.xlabel('Learning rate', fontsize=14)
  plt.ylabel('Testing accuracy', fontsize=14)
  plt.show()

