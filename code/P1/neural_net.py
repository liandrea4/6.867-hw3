import sys
sys.path.append('../../../hw1/code/P1')
from sgd        import sgd
import numpy    as np

def softmax_fn(z):
  output = []
  denominator_sum = 0.
  for z_j in z:
    denominator_sum += np.e(z_j)

  for z_i in z:
    output.append(np.e(z_i) / denominator_sum)
  return output


def relu_fn(z):
  output = []
  for z_i in z:
    output.append(max(0., z_i))
  return output


def loss_gradient_fn(x, y, z, w):
  df_dz = 0
  if z > 0:
    df_dz = 1

  dl_dz = 0
  if cross_entropy_loss(y, z, softmax_fn) > 0:
    dl_dz = -y

  return x * df_dz * w * dl_dz


def calculate_z(x, w, b):
  z_sum = 0
  for x_i, w_i in zip(x, w):
    z_sum += float(x_i) * w_i
  return z_sum + b


def cross_entropy_loss(x, y, fn):
  all_z = []

  loss_sum = 0
  for x_i, y_i in zip(x, y):
    z_i = calculate_z(x_i, w, b)
    log = np.log(fn(z_i, all_z))
    loss_sum += y_i * log

  return -loss_sum

def calculate_node_output():


def train_neural_net(inputs, num_layers, num_nodes):
  current_inputs = inputs
  for layer_index in range(num_layers):

    for node_index in range(num_nodes):
      node_output = calculate_node_output(current_inputs)

    current_inputs = []


############################################

def feedforward(x, theta, num_layers):
  alphas = [x]
  z = [0]

  for layer_index in range(1, num_layers):
    w_l = theta[2 * (layer_index - 1)]
    b_l - theta[2 * (layer_index - 1) + 1]

    if layer_index < num_layers - 1:
      fn = relu_fn
    else:
      fn = softmax_fn

    prev_alpha = alphas[layer_index - 1]
    z_l = np.dot(np.transpose(w_l), prev_alpha) + b_l
    a_l = fn(z_l)

    alphas.append(a_l)
    z.append(z_l)

  return alphas, z

def calculate_output_error():


def backprop(num_layers, output_error, theta, z):
  deltas = [output_error]

  for layer_index in reversed(range(num_layers-1, 1)):
    next_delta = deltas[0]
    next_w = theta[2 * layer_index]

    if z[layer_index] > 0:
      derivative = 1
    else:
      derivative = 0

    delta = np.dot(np.diag(derivative), np.dot(next_w, next_delta))
    deltas.insert(0, delta)

  return deltas


def calculate_gradient(deltas, alphas, num_layers):
  gradient = []

  for layer_index in range(num_layers):
    ## TODO: why layer_index - 1? what does this mean for l = 0 or 1?
    weight_gradient = np.dot(alpha[layer_index - 1], deltas[layer_index])
    bias_gradient = deltas[layer_index]

    gradient.append(weight_gradient)
    gradient.append(bias_gradient)

  return gradient

if __name__ == '__main__'






