import sys
import numpy    as np
import pylab    as pl

##### Softmax functions #####
def softmax_fn(z):
  output = []
  denominator_sum = 0.
  for z_j in z:
    denominator_sum += np.exp(z_j)

  for z_i in z:
    output.append(np.exp(z_i) / denominator_sum)
  return output


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
  return output


def relu_derivative_fn(z):
  derivative = []

  for z_i in z:
    if z_i > 0:
      derivative.append(1)
    else:
      derivative.append(0)

  return derivative


# def loss_gradient_fn(x, y, z, w):
#   df_dz = 0
#   if z > 0:
#     df_dz = 1

#   dl_dz = 0
#   if cross_entropy_loss(y, z, softmax_fn) > 0:
#     dl_dz = -y

#   return x * df_dz * w * dl_dz


# def calculate_z(x, w, b):
#   z_sum = 0
#   for x_i, w_i in zip(x, w):
#     z_sum += float(x_i) * w_i
#   return z_sum + b


# def cross_entropy_loss(x, y, fn):
#   all_z = []

#   loss_sum = 0
#   for x_i, y_i in zip(x, y):
#     z_i = calculate_z(x_i, w, b)
#     log = np.log(fn(z_i, all_z))
#     loss_sum += y_i * log

#   return -loss_sum

# def calculate_node_output():


# def train_neural_net(inputs, num_layers, num_nodes):
#   current_inputs = inputs
#   for layer_index in range(num_layers):

#     for node_index in range(num_nodes):
#       node_output = calculate_node_output(current_inputs)

#     current_inputs = []


############################################


def feedforward(x, theta, num_layers):
  alphas = [x]
  z = []

  for layer_index in range(num_layers):
    w_l = theta[2 * (layer_index)]
    b_l = theta[2 * (layer_index) + 1]

    if layer_index < num_layers:
      fn = relu_fn
    else:
      fn = softmax_fn

    prev_alpha = alphas[layer_index - 1]
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

  for layer_index in reversed(range(1, num_layers-1)):
    print "layer_index: ", layer_index
    # Building up list of deltas, so next delta (previously calculated) is at the 0th index
    next_delta = deltas[0]
    next_w = theta[2 * (layer_index - 1)]

    derivative = relu_derivative_fn(z[layer_index])

    delta = np.dot(np.diag(derivative), np.dot(next_w, next_delta))
    deltas.insert(0, delta)

  return deltas


def calculate_gradient(deltas, alphas, num_layers):
  gradient = []

  for layer_index in range(num_layers):
    # Alphas starts with a_0, deltas starts with delta_1
    weight_gradient = np.dot(np.transpose([alphas[layer_index]]), [deltas[layer_index]])
    bias_gradient = deltas[layer_index]

    gradient.append(weight_gradient)
    gradient.append(bias_gradient)

  return gradient

def calculate_overall_gradient(x, y, theta):
  num_layers = len(theta) / 2

  print "Feedforward..."
  alphas, z = feedforward(x, theta, num_layers)
  print "Calculating output error..."
  output_error = calculate_output_error(y, z, alphas)
  print "Backprop..."
  deltas = backprop(num_layers, output_error, theta, z)
  print "Calculating gradient..."
  gradient = calculate_gradient(deltas, alphas, num_layers)

  return gradient

##### SGD #####

def calculate_next_theta(old_theta, x, y, t):
  t0 = 10**6
  k = 0.75
  n = lambda t: (t0 + t)**(-k)
  gradient = calculate_overall_gradient(x, y, old_theta)

  new_theta = []
  for old_elem, gradient_elem in zip(old_theta, gradient):
    step = np.dot(n(t), gradient_elem)
    new_theta.append(old_elem - step)

  return new_theta


def sgd(x, y, theta):  #, objective_f, threshold):
  number_of_samples = len(x)

  differences = [False] * number_of_samples
  old_jthetas = [0.0] * number_of_samples
  previous_values = []
  t = 0

  while t < 10 and not all(differences):
    i = t % number_of_samples
    theta = calculate_next_theta(theta, x[i], y[i], t)
    # new_jtheta = objective_f(x[i], y[i], theta)
    # difference = new_jtheta - old_jthetas[i]

    # if(abs(difference) < threshold):
    #   differences[i] = True

    previous_values.append(theta)
    # previous_values.append((theta, new_jtheta))
    # old_jthetas[i] = new_jtheta

    t += 1

  return previous_values


def get_data(filename):
  data = pl.loadtxt(filename)
  x, y = [], []

  for line in data:
    x.append(line[:2])
    y.append(line[2])

  return x, y

if __name__ == '__main__':
  filename = "../data/data_3class.csv"
  x, y = get_data(filename)
  neurons_per_layer = 5


  W_1 = np.array([[1.]*neurons_per_layer]*len(x[0]))
  b_1 = np.array([1.]*neurons_per_layer)
  theta = [W_1, b_1]

  print sgd(x, y, theta)



